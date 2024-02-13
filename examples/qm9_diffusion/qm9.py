import os, json

import torch
import torch_geometric
from torch_geometric.data import Data
# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn
from torch_gdl import EquivariantDiffusionProcess, DiffusionProcess
import random
from argparse import ArgumentParser

def atomic_num_to_one_hot(atom_idents: torch.Tensor) -> torch.Tensor:
    # expect 1-D tensor
    # CHONF

    atom_idents = atom_idents.where(atom_idents != 1, 0) # hydrogen
    atom_idents = atom_idents.where(atom_idents != 6, 1) # carbon
    atom_idents = atom_idents.where(atom_idents != 7, 2) # nitrogen
    atom_idents = atom_idents.where(atom_idents != 8, 3) # oxygen
    atom_idents = atom_idents.where(atom_idents != 9, 4) # fluorine
    
    return torch.nn.functional.one_hot(atom_idents, 5)

def fc_edge_index(n_nodes: int) -> torch.Tensor:
    assert isinstance(n_nodes, int)
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive int")
    n_edges = n_nodes * (n_nodes - 1)
    edge_index = torch.empty((2, n_edges), dtype=torch.long)
    c = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_index[0][c] = i
                edge_index[1][c] = j
                c += 1
    return edge_index

def qm9_pre_transform(data: Data):
    # create a fully connected graph. retain the previous edge_index
    data.edge_index_old = data.edge_index
    data.edge_index = fc_edge_index(data.num_nodes)

    # Set descriptor as element type. ONE HOT
    data.x = atomic_num_to_one_hot(data.z.long()).float() # ONE HOT
    return data



def qm9_pre_filter(data):
    return data.idx < num_samples

def get_train_transform(dp: DiffusionProcess, head_types: list, out_indices: list, graph_feature_dim: list, node_feature_dim: list):

    def train_transform(data: Data):

        data.t = 0 # default
        
        # randomly sample a t
        t = random.randint(1, dp.timesteps-1)

        data = dp.forward_process_sample(data, t) # should be attaching t to node features

        # concatenate node features and time
        time_vec = torch.ones((data.num_nodes, 1), device=data.x.device) * t / (dp.timesteps - 1.)
        data.x = torch.hstack([data.x, time_vec]) # n_nodes, 2

        # set y to the expected shape for HydraGNN. create a hack for noise data by creating a new data with noise in .x and .pos
        x_targ = torch.hstack([data.yx, time_vec])
        noisedata = Data(x=x_targ, pos=data.ypos)
        hydragnn.preprocess.utils.update_predicted_values(
            head_types, out_indices, graph_feature_dim, node_feature_dim, noisedata
        )
        # extract .y from the hack
        data.y = noisedata.y
        data.y_loc = noisedata.y_loc

        return data

    return train_transform


if __name__ == "__main__":

    # parser = ArgumentParser()

    # parser.add_argument("--lr", type=float, default=1e-5)

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    num_samples = 1000

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
    with open(filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.setup_ddp()

    log_name = "qm9_test"
    # Enable print to log file.
    hydragnn.utils.setup_log(log_name)

    voi = config["NeuralNetwork"]["Variables_of_interest"]

    # Use built-in torch_geometric dataset.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
    dp = EquivariantDiffusionProcess(100)
    train_tform = get_train_transform(dp, voi["type"], voi["output_index"], [], voi["output_dim"])
    dataset = torch_geometric.datasets.QM9(
        root="dataset/qm9", pre_transform=qm9_pre_transform, pre_filter=qm9_pre_filter, transform=train_tform
    )

    datum = dataset[0]
    print("X: ", datum.x)
    print("EDGE: ", datum.edge_index)

    print("---------Diffused Version---------")
    datum = train_tform(datum)
    print("Time: ", datum.t)
    print("X: ", datum.x)
    print("EDGE: ", datum.edge_index)
    print("Y: ", datum.y)
    print("POS: ", datum.pos)
    print("YPOS: ", datum.ypos)
    print("Y_shape: ", datum.y.shape)

    train, val, test = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # Run training with the given model and qm9 dataset.
    writer = hydragnn.utils.get_summary_writer(log_name)
    hydragnn.utils.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
    )
