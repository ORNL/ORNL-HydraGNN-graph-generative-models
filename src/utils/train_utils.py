import os, random
import tqdm
import torch

from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, Any, List, Optional

from src.processes.diffusion import DiffusionProcess
from src.utils.logging_utils import ModelLoggerHandler


def diffusion_loss(outputs, targets):
    """
    Loss function for graph diffusion models with norm constraint.
    Computes positional loss with norm constraint and atom type loss.
    
    Args:
        outputs: Model outputs [atom_predictions, position_predictions]
        targets: Target values [atom_targets, position_targets]
        
    Returns:
        Tuple of (position_loss, atom_type_loss, metrics_dict)
    """
    # Basic MSE loss for positions
    pos_mse_loss = torch.nn.functional.mse_loss(outputs[1], targets[1])
    
    # Add norm constraint to ensure predictions have appropriate scale
    # Calculate average norm of predicted and target noise
    pred_norm = torch.norm(outputs[1], dim=1).mean()
    target_norm = torch.norm(targets[1], dim=1).mean()
    
    # Norm constraint: penalize deviation from expected norm
    norm_constraint_weight = 1.0  # Adjust this weight as needed
    norm_constraint_loss = torch.abs(pred_norm - target_norm)
    
    # Combined position loss
    pos_loss = pos_mse_loss + norm_constraint_weight * norm_constraint_loss
    
    # Cross entropy loss for atom types
    atom_loss = torch.nn.functional.cross_entropy(outputs[0], targets[0])
    
    # Create a dictionary of metrics to track
    metrics = {
        'pos_mse_loss': pos_mse_loss.item(),
        'norm_constraint_loss': norm_constraint_loss.item(),
        'pred_norm': pred_norm.item(),
        'target_norm': target_norm.item()
    }
    
    return pos_loss, atom_loss, metrics


def get_device():
    """
    Determine the appropriate device (MPS, CUDA, or CPU) for training.

    Returns:
    --------
        device: The torch device to use for training
    """
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda:4"
        if torch.cuda.is_available()
        else "cpu"
    )

def postprocess_model_outputs(outputs, data):   
    # The model should directly predict the noise (epsilon) that was added
    # No need to subtract positions as that would create a mismatch with our training targets
    # We only need to ensure the center of mass is zero by centering the predictions
    outputs[1] = outputs[1] - torch.mean(outputs[1], dim=0)
    return outputs

def train_epoch(model, loss_fun, optimizer, dataloader, device, logger_handler, epoch, transform_fn=None):
    """
    Train the model for one epoch.

    Args:
    -----
        model (torch.nn.Module): The model to be trained
        loss_fun (callable): The loss function to compute the training loss
        optimizer (torch.optim.Optimizer): The optimizer for updating weights
        dataloader (torch.utils.data.DataLoader): The training data loader
        device (torch.device): The device to run training on
        logger_handler (ModelLoggerHandler): Logger for tracking metrics
        epoch (int): Current epoch number
        transform_fn (callable, optional): Transform function to apply to each batch

    Returns:
    --------
        tuple: (epoch_loss, epoch_pos_loss, epoch_atom_loss, batch_count)
    """
    model.train()
    epoch_loss = 0
    epoch_pos_loss = 0  # Positional loss (MSE)
    epoch_atom_loss = 0  # Atom type loss (Cross-entropy)
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # Apply transform to add fresh noise if transform function is provided
        if transform_fn is not None:
            # For batches we need to transform each sample individually
            transformed_samples = [transform_fn(data.clone()) for data in batch.to_data_list()]
            batch = Batch.from_data_list(transformed_samples)

        # Forward pass
        batch = batch.to(device)
        outputs = postprocess_model_outputs(model(batch), batch)
        # then, compute the loss
        loss_pos, loss_atom, loss_metrics = loss_fun(outputs, [batch.y[:, :5], batch.y[:, 5:]])
        # Track both losses but only use position loss for optimization
        loss = loss_pos  # only work on positional loss for now

        # Backward pass and optimization
        loss.backward()
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        epoch_loss += loss.item()
        epoch_pos_loss += loss_pos.item()
        epoch_atom_loss += loss_atom.item()
        batch_count += 1

        # Calculate norms to monitor predicted vs actual noise scale
        with torch.no_grad():
            pred_noise_norm = torch.norm(outputs[1], dim=1).mean().item()
            actual_noise_norm = torch.norm(batch.y[:, 5:], dim=1).mean().item()
            noise_scale_ratio = pred_noise_norm / (actual_noise_norm + 1e-6)  # avoid division by zero
        
        # Log batch metrics including separate loss components and noise norms
        loss_components = {
            "position_loss": loss_pos, 
            "atom_type_loss": loss_atom,
            "pred_noise_norm": pred_noise_norm,
            "actual_noise_norm": actual_noise_norm,
            "noise_scale_ratio": noise_scale_ratio,
            "pos_mse_loss": loss_metrics['pos_mse_loss'],
            "norm_constraint_loss": loss_metrics['norm_constraint_loss']
        }
        logger_handler.log_batch(
            loss, batch_idx, epoch, len(dataloader), "train", loss_components
        )

    return epoch_loss, epoch_pos_loss, epoch_atom_loss, batch_count


def validate_epoch(model, loss_fun, dataloader, device, logger_handler, epoch, transform_fn=None):
    """
    Validate the model for one epoch.

    Args:
    -----
        model (torch.nn.Module): The model to be validated
        loss_fun (callable): The loss function to compute the validation loss
        dataloader (torch.utils.data.DataLoader): The validation data loader
        device (torch.device): The device to run validation on
        logger_handler (ModelLoggerHandler): Logger for tracking metrics
        epoch (int): Current epoch number
        transform_fn (callable, optional): Transform function to apply to each batch

    Returns:
    --------
        tuple: (epoch_val_loss, epoch_val_pos_loss, epoch_val_atom_loss, batch_count_val)
    """
    model.eval()
    epoch_val_loss = 0
    epoch_val_pos_loss = 0
    epoch_val_atom_loss = 0
    batch_count_val = 0

    for val_batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            # Apply transform to add fresh noise if transform function is provided
            if transform_fn is not None:
                # For batches we need to transform each sample individually
                transformed_samples = [transform_fn(data.clone()) for data in batch.to_data_list()]
                batch = Batch.from_data_list(transformed_samples)
            
            # Forward pass
            batch = batch.to(device)
            outputs = postprocess_model_outputs(model(batch), batch)

            loss_pos, loss_atom, loss_metrics = loss_fun(outputs, [batch.y[:, :5], batch.y[:, 5:]])
            loss = loss_pos  # matching the training loss

            # Accumulate losses
            epoch_val_loss += loss.item()
            epoch_val_pos_loss += loss_pos.item()
            epoch_val_atom_loss += loss_atom.item()
            batch_count_val += 1

            # Calculate norms to monitor predicted vs actual noise scale
            pred_noise_norm = torch.norm(outputs[1], dim=1).mean().item()
            actual_noise_norm = torch.norm(batch.y[:, 5:], dim=1).mean().item()
            noise_scale_ratio = pred_noise_norm / (actual_noise_norm + 1e-6)  # avoid division by zero
            
            # Log batch metrics including loss components and noise norms
            val_loss_components = {
                "position_loss": loss_pos,
                "atom_type_loss": loss_atom,
                "pred_noise_norm": pred_noise_norm,
                "actual_noise_norm": actual_noise_norm,
                "noise_scale_ratio": noise_scale_ratio,
                "pos_mse_loss": loss_metrics['pos_mse_loss'],
                "norm_constraint_loss": loss_metrics['norm_constraint_loss']
            }
            logger_handler.log_batch(
                loss,
                val_batch_idx,
                epoch,
                len(dataloader),
                "val",
                val_loss_components,
            )

    return epoch_val_loss, epoch_val_pos_loss, epoch_val_atom_loss, batch_count_val


def log_epoch_metrics(logger_handler, epoch, train_losses, val_losses, optimizer):
    """
    Calculate and log metrics for the epoch.

    Args:
    -----
        logger_handler (ModelLoggerHandler): Logger for tracking metrics
        epoch (int): Current epoch number
        train_losses (tuple): (epoch_loss, epoch_pos_loss, epoch_atom_loss, batch_count)
        val_losses (tuple): (epoch_val_loss, epoch_val_pos_loss, epoch_val_atom_loss, batch_count_val)
        optimizer (torch.optim.Optimizer): The optimizer used for training

    Returns:
    --------
        float: The average epoch loss
    """
    epoch_loss, epoch_pos_loss, epoch_atom_loss, batch_count = train_losses
    (
        epoch_val_loss,
        epoch_val_pos_loss,
        epoch_val_atom_loss,
        batch_count_val,
    ) = val_losses

    # Calculate average losses
    avg_epoch_val_loss = epoch_val_loss / batch_count_val
    avg_epoch_loss = epoch_loss / batch_count

    # Calculate average loss components
    avg_pos_loss = epoch_pos_loss / batch_count
    avg_atom_loss = epoch_atom_loss / batch_count
    avg_val_pos_loss = epoch_val_pos_loss / batch_count_val
    avg_val_atom_loss = epoch_val_atom_loss / batch_count_val

    # Create loss component averages dictionary
    loss_component_avgs = {
        "train_position_loss": avg_pos_loss,
        "train_atom_type_loss": avg_atom_loss,
        "val_position_loss": avg_val_pos_loss,
        "val_atom_type_loss": avg_val_atom_loss,
    }

    # Log epoch metrics with loss component averages
    logger_handler.log_epoch(
        epoch,
        avg_epoch_loss,
        avg_epoch_val_loss,
        optimizer.param_groups[0]["lr"],
        loss_component_avgs,
    )

    return avg_epoch_loss


def train_model(
    model,
    loss_fun,
    optimizer,
    train_dataloader,
    val_dataloader,
    num_epochs,
    logger=None,
    config=None,
    save_freq=1,
    save_best=True,
    model_name="model",
    scheduler=None,
    train_transform=None
):
    """
    Trains a given model using the provided loss function, optimizer, and data loader.

    Args:
    -----
        model (torch.nn.Module): The model to be trained.
        loss_fun (callable): The loss function to compute the training loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's weights.
        train_dataloader (torch.utils.data.DataLoader): An iterable over the training dataset.
        val_dataloader (torch.utils.data.DataLoader): An iterable over the validation dataset.
        num_epochs (int): The number of epochs for training.
        logger: The logger object to use for experiment tracking.
        config (dict, optional): Configuration dictionary for logger initialization.
        save_freq (int): Frequency of saving model checkpoints (in epochs).
        save_best (bool): Whether to save the best model based on loss.
        model_name (str): Base name for saved model files.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        train_transform (callable, optional): Transform function to apply to each batch.

    Returns:
    --------
        model: The trained model
    """
    device = get_device()
    model.to(device)

    # Initialize logger handler
    logger_handler = ModelLoggerHandler(
        logger=logger, model_name=model_name, save_freq=save_freq, save_best=save_best
    )
    logger_handler.setup(model, config)
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        # Train for one epoch
        train_losses = train_epoch(
            model, loss_fun, optimizer, train_dataloader, device, logger_handler, epoch, train_transform
        )

        # Validate the model
        val_losses = validate_epoch(
            model, loss_fun, val_dataloader, device, logger_handler, epoch, train_transform
        )

        # Log the epoch metrics
        avg_epoch_loss = log_epoch_metrics(
            logger_handler, epoch, train_losses, val_losses, optimizer
        )
        
        # Update learning rate with scheduler if provided
        if scheduler is not None:
            # For ReduceLROnPlateau, step needs the validation loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses[0] / val_losses[3])  # avg_val_loss
            else:
                scheduler.step()
                
        # Handle end of epoch (saving checkpoints)
        logger_handler.handle_epoch_end(epoch, model, optimizer, avg_epoch_loss)

    print("Training complete!")
    logger_handler.finish()
    return model


def get_train_transform(dp: DiffusionProcess):
    """
    Returns a training transform function for the QM9 dataset. Encompasses
    the forward noising process, time insertion, and formatting for the
    denoising model.

    Args:
    -----
    dp (DiffusionProcess):
        The diffusion process object.
    head_types (list):
        The types of heads in the denoising model.
    out_indices (list):
        The output indices of the denoising model.
    graph_feature_dim (list):
        The dimensions of the graph features.
    node_feature_dim (list):
        The dimensions of the node features.
    """

    def train_transform(data: Data):
        data.t = 0  # default

        # Only use atom type features
        data.x = data.x[:, :5].float()

        # randomly sample a t
        t = random.randint(0, dp.timesteps - 1)

        # noise the data 
        noised_data = dp.forward_process_sample(
            data, t
        )  
        # the noised data object has the following attributes:
        # - x: noised atom types 
        # - x_t: probability distribution of atom types 
        # - pos: noised positions (eg z_t^(x) from Hoogeboom et al.)
        # - ypos: epsilon for the positions, with center of gravity removed
        # - t: the time step t
        # - pos_mu: set to None
        # - pos_sigma: set to None

        # insert time t into the node features
        noised_data, time_vec = insert_t(noised_data, data.t, dp.timesteps)

        # set y_loc for hydragnn reasons
        noised_data.y_loc = torch.tensor(
            [0, 5, 8], dtype=torch.int64, device=data.x.device
        ).unsqueeze(0)

        # assign y values for "epsilon" parameterization
        noised_data.y = torch.hstack([data.x, noised_data.ypos])
        return noised_data

    return train_transform

def get_hydra_transform():
    """
    Returns a training transform function for the QM9 dataset. Encompasses
    the forward noising process, time insertion, and formatting for the
    denoising model.

    Args:
    -----
    dp (DiffusionProcess):
        The diffusion process object.
    head_types (list):
        The types of heads in the denoising model.
    out_indices (list):
        The output indices of the denoising model.
    graph_feature_dim (list):
        The dimensions of the graph features.
    node_feature_dim (list):
        The dimensions of the node features.
    """
    def hydra_transform(data: Data):
        data.y_loc = torch.tensor(
            [0, 5, 8], dtype=torch.int64, device=data.x.device
        ).unsqueeze(0)
        return data

    return hydra_transform


def insert_t(data: Data, t: int, T: int):
    """
    Insert time t into the node features of the data before passing
    to the denoising model.

    Args:
    -----
    data (Data):
        Noised sample from the diffusion process.
    t (int):
        The time step to insert into the node features.
    T (int):
        The total number of time steps in the diffusion process.
    """
    data_ins = data.clone().detach_()
    # concatenate node features and (scaled) time feature
    time_vec = (
        torch.ones((data_ins.num_nodes, 1), device=data_ins.x.device) * t / (T - 1.0)
    )
    data_ins.x = torch.hstack([data_ins.x, time_vec])  # (n_nodes, 6)
    return data_ins, time_vec
