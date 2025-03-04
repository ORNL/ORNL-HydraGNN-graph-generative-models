import os, random
import tqdm
import torch
import wandb

from torch_geometric.data import Data

from src.processes.diffusion import DiffusionProcess


class ModelLoggerHandler:
    """
    Handles model logging and saving operations during training.
    """

    def __init__(self, logger=None, model_name="model", save_freq=1, save_best=True):
        self.logger = logger
        self.model_name = model_name
        self.save_freq = save_freq
        self.save_best = save_best
        self.best_loss = float("inf")

        # Setup save directory if using wandb
        if isinstance(logger, type(wandb.run)):
            self.save_dir = os.path.join(wandb.run.dir, "checkpoints")
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = "."

    def setup(self, model, config=None):
        """Initialize logger with config and model watching."""
        if self.logger is not None and config is not None:
            if hasattr(self.logger, "config"):
                for key, value in config.items():
                    self.logger.config[key] = value

            if hasattr(self.logger, "watch"):
                self.logger.watch(model, log_freq=100)

    def log_batch(
        self, loss, batch_idx, epoch, total_batches, tag="train", loss_components=None
    ):
        """Log batch-level metrics with optional individual loss components.

        Args:
            loss: The total loss value
            batch_idx: Current batch index
            epoch: Current epoch
            total_batches: Total number of batches per epoch
            tag: Tag to differentiate train/val metrics
            loss_components: Dictionary of individual loss components to log
        """
        if self.logger is not None and hasattr(self.logger, "log"):
            metrics = {
                f"{tag} batch_loss": loss.item(),
                f"{tag} batch": batch_idx + epoch * total_batches,
            }

            # Add individual loss components if provided
            if loss_components is not None:
                for name, value in loss_components.items():
                    metrics[f"{tag} {name}"] = (
                        value.item() if torch.is_tensor(value) else value
                    )

            self.logger.log(metrics)

    def log_epoch(
        self, epoch, avg_loss, avg_val_loss, learning_rate, loss_component_avgs=None
    ):
        """Log epoch-level metrics with optional individual loss component averages.

        Args:
            epoch: Current epoch
            avg_loss: Average total loss for the epoch
            avg_val_loss: Average validation loss for the epoch
            learning_rate: Current learning rate
            loss_component_avgs: Dictionary of average individual loss components
        """
        if self.logger is not None and hasattr(self.logger, "log"):
            metrics = {
                "epoch": epoch + 1,
                "epoch_loss": avg_loss,
                "learning_rate": learning_rate,
                "avg_val_loss": avg_val_loss,
            }

            # Add individual loss component averages if provided
            if loss_component_avgs is not None:
                for name, value in loss_component_avgs.items():
                    metrics[f"avg_{name}"] = value

            self.logger.log(metrics)

    def save_checkpoint(self, epoch, model, optimizer, loss, is_best=False):
        """Save model checkpoint and upload to wandb if available."""
        if self.logger is None:
            return

        # Determine checkpoint name
        checkpoint_name = (
            f"{self.model_name}_best.pt"
            if is_best
            else f"{self.model_name}_epoch_{epoch+1}.pt"
        )
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        # Save checkpoint locally
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                "loss": loss,
            },
            checkpoint_path,
        )

        # Upload to wandb if available
        # if isinstance(self.logger, type(wandb.run)):
        #     artifact = wandb.Artifact(
        #         name=f'{self.model_name}_{"best" if is_best else "checkpoint"}',
        #         type='model',
        #         description=f'{"Best model" if is_best else "Model"} checkpoint from epoch {epoch+1}'
        #     )
        # artifact.add_file(checkpoint_path)
        # self.logger.log_artifact(artifact)
        # Clean up local checkpoint
        # os.remove(checkpoint_name)

    def handle_epoch_end(self, epoch, model, optimizer, avg_loss):
        """Handle end of epoch operations (logging and saving)."""
        # Save based on frequency
        # if (epoch + 1) % self.save_freq == 0:
        #     self.save_checkpoint(epoch, model, optimizer, avg_loss)

        # Save best model if enabled
        if self.save_best and avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_checkpoint(epoch, model, optimizer, avg_loss, is_best=True)

    def finish(self):
        """Cleanup logging resources."""
        if self.logger is not None and hasattr(self.logger, "finish"):
            self.logger.finish()


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


def train_epoch(model, loss_fun, optimizer, dataloader, device, logger_handler, epoch):
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
        # Forward pass
        outputs = model(batch.to(device))
        loss_pos, loss_atom = loss_fun(outputs, [batch.y[:, :5], batch.y[:, 5:]])
        # Track both losses but only use position loss for optimization
        loss = loss_pos  # only work on positional loss for now

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        epoch_loss += loss.item()
        epoch_pos_loss += loss_pos.item()
        epoch_atom_loss += loss_atom.item()
        batch_count += 1

        # Log batch metrics including separate loss components
        loss_components = {"position_loss": loss_pos, "atom_type_loss": loss_atom}
        logger_handler.log_batch(
            loss, batch_idx, epoch, len(dataloader), "train", loss_components
        )

    return epoch_loss, epoch_pos_loss, epoch_atom_loss, batch_count


def validate_epoch(model, loss_fun, dataloader, device, logger_handler, epoch):
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
            # Forward pass
            outputs = model(batch.to(device))
            loss_pos, loss_atom = loss_fun(outputs, [batch.y[:, :5], batch.y[:, 5:]])
            loss = loss_pos  # matching the training loss

            # Accumulate losses
            epoch_val_loss += loss.item()
            epoch_val_pos_loss += loss_pos.item()
            epoch_val_atom_loss += loss_atom.item()
            batch_count_val += 1

            # Log batch metrics including loss components
            val_loss_components = {
                "position_loss": loss_pos,
                "atom_type_loss": loss_atom,
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
            model, loss_fun, optimizer, train_dataloader, device, logger_handler, epoch
        )

        # Validate the model
        val_losses = validate_epoch(
            model, loss_fun, val_dataloader, device, logger_handler, epoch
        )

        # Log the epoch metrics
        avg_epoch_loss = log_epoch_metrics(
            logger_handler, epoch, train_losses, val_losses, optimizer
        )

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
        initial_x = data.x.clone().float()
        initial_pos = data.pos.clone()

        # randomly sample a t
        t = random.randint(0, dp.timesteps - 1)
        data = dp.forward_process_sample(
            data, t
        )  # should be attaching t to node features

        data, time_vec = insert_t(data, data.t, dp.timesteps)

        training_sample = data.clone()
        training_sample.y_loc = torch.tensor(
            [0, 5, 8], dtype=torch.int64, device=data.x.device
        ).unsqueeze(0)

        x_noised = torch.hstack([data.x_t, time_vec])
        training_sample.x = x_noised

        training_sample.pos = data.pos
        # "epsilon" parameterization
        training_sample.y = torch.hstack([data.x_t - initial_x, data.pos - initial_pos])
        return training_sample

    return train_transform


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
