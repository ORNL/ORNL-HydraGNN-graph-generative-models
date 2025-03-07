import os, json, sys, argparse, random, datetime, wandb
import torch, torch_geometric

# from torchmdnet.models import EquivariantModel
# from torchmdnet.utils import make_spherical_harmonics
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch, DataLoader

import hydragnn
from hydragnn.utils.print import setup_log
from hydragnn.utils.input_config_parsing import config_utils
from hydragnn.utils.distributed import setup_ddp, get_distributed_model
from hydragnn.preprocess.graph_samples_checks_and_updates import update_predicted_values


from src.utils import diffusion_utils as du
from src.processes.diffusion import DiffusionProcess
from src.processes.equivariant_diffusion import EquivariantDiffusionProcess
from src.processes.marginal_diffusion import MarginalDiffusionProcess
from src.utils.train_utils import train_model, get_train_transform, insert_t
from src.utils.data_utils import get_marg_dist, FullyConnectGraph


def train(args):

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Configurable run choices (JSON file that accompanies this example script).
    with open(args.config_path, "r") as f:
        config = json.load(f)

    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    # Always initialize for multi-rank training.
    world_size, world_rank = setup_ddp()
    voi = config["NeuralNetwork"]["Variables_of_interest"]

    # Create a MarginalDiffusionProcess object.
    dp = MarginalDiffusionProcess(
        args.diffusion_steps, marg_dist=get_marg_dist(root_path=args.data_path)
    )
    # dp = EquivariantDiffusionProcess(args.diffusion_steps)

    # Create a training transform function for the QM9 dataset.
    train_tform = get_train_transform(dp)

    # Load the QM9 dataset from torch with the pre-transform, pre-filter, and train transform.
    # TODO should be generalized, a la fine tuning
    dataset = torch_geometric.datasets.QM9(root=args.data_path, transform=train_tform)
    # dataset = torch_geometric.datasets.QM9(root=args.data_path)

    # Limit the number of samples if specified.
    if args.samples != None:
        dataset = dataset[: args.samples]
    else:
        print("Training on Full Dataset")

    # Make all graphs fully connected.
    dataset = [FullyConnectGraph()(data) for data in dataset]  # Apply to all graphs
    # datum = dataset[0]
    # print(datum)
    # print("X: ", datum.x)
    # print("POS: ", datum.pos)
    # print("EDGE: ", datum.edge_index)

    # print("---------Diffused Version---------")
    # datum = train_tform(datum)
    # print(datum)
    # print("Time: ", datum.t)
    # print("X: ", datum.x)
    # print("EDGE: ", datum.edge_index)
    # #print("Y: ", datum.y)
    # print("POS: ", datum.pos)
    # print("YPOS: ", datum.ypos)
    # print("Y_shape: ", datum.y.shape)

    # TODO modify config to move Training outside of Neural Network
    # Split into train, validation, and test sets.
    train, val, test = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )
    # Create dataloaders for PyTorch training
    (
        train_loader,
        val_loader,
        test_loader,
    ) = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    # Update the config with the dataloaders.
    config = config_utils.update_config(config, train_loader, val_loader, test_loader)

    # Save the config with all the updated stuff
    wandb.init(project="graph diffusion model", config=config)

    # Create the model from the config specifications
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    # Define training optimizer and scheduler
    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # TODO move this to train_utils.py, name specific
    def loss(outputs, targets):
        pos_loss = torch.nn.functional.mse_loss(outputs[1], targets[1])
        atom_loss = torch.nn.functional.cross_entropy(outputs[0], targets[0])
        return pos_loss, atom_loss

    # Run training with the given model and dataset.
    model = train_model(
        model,
        loss,
        optimizer,
        train_loader,
        val_loader,
        config["NeuralNetwork"]["Training"]["num_epoch"],
        logger=wandb.run,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Create default log name if not specified.

    parser.add_argument("-s", "--samples", type=int)
    parser.add_argument("-ds", "--diffusion_steps", type=int, default=1000)
    parser.add_argument(
        "-c", "--config_path", type=str, default="examples/qm9/qm9_marginal.json"
    )
    parser.add_argument("-d", "--data_path", type=str, default="examples/qm9/dataset")

    # Store the arguments in args.
    args = parser.parse_args()
    train(args)
