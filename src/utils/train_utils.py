import random
import torch

from torch_geometric.data import Data

from src.processes.diffusion import DiffusionProcess


def train_model(model, loss_fun, optimizer, dataloader, num_epochs):
    """
    Trains a given model using the provided loss function, optimizer, and data loader.

    Args:
    -----
        model (torch.nn.Module): The model to be trained.
        loss_fun (callable): The loss function to compute the training loss.
            Should accept the model outputs and target labels as input.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's weights.
        dataloader (torch.utils.data.DataLoader): An iterable over the training dataset, providing batches of input data.
        num_epochs (int): The number of epochs for training.

    Returns:
    --------
        None
    """

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0

        for batch in dataloader:

            # Forward pass
            outputs = model(batch)
            loss = loss_fun(outputs, [batch.y[:, :6], batch.y[:, 6:]])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete!")
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

        data = dp.forward_process_sample(
            data, t
        )  # should be attaching t to node features

        data, time_vec = insert_t(data, data.t, dp.timesteps)

        # set y to the expected shape for HydraGNN. Create a hack for
        # noise data by creating a new data with noise in .x and .pos
        x_targ = torch.hstack([data.x_probs, time_vec])
        noisedata = Data(x=x_targ, pos=data.ypos)

        # update_predicted_values(
        #     head_types, out_indices, graph_feature_dim, node_feature_dim, noisedata
        # )
        # extract .y from the hack
        # data.y = noisedata.y
        # data.y_loc = noisedata.y_loc
        training_sample = data.clone()
        training_sample.y_loc = torch.tensor(
            [0, 6, 9], dtype=torch.int64, device=data.x.device
        ).unsqueeze(0)
        training_sample.x = x_targ
        training_sample.pos = data.ypos
        training_sample.y = torch.hstack([data.x, data.pos])
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
