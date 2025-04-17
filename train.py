import os, json, sys, argparse, random, datetime
import torch, torch_geometric

# from torchmdnet.models import EquivariantModel
# from torchmdnet.utils import make_spherical_harmonics
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch, DataLoader

import hydragnn
from hydragnn.utils.input_config_parsing import config_utils
from hydragnn.utils.distributed import setup_ddp
from hydragnn.preprocess.graph_samples_checks_and_updates import update_predicted_values

from src.utils import diffusion_utils as du
from src.processes.diffusion import DiffusionProcess
from src.processes.equivariant_diffusion import EquivariantDiffusionProcess
from src.processes.marginal_diffusion import MarginalDiffusionProcess
from src.utils.train_utils import (
    train_model,
    get_train_transform,
    get_hydra_transform,
    get_deterministic_transform,
    insert_t,
    diffusion_loss,
)
from src.utils.logging_utils import ModelLoggerHandler, configure_wandb
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

    # Create a DiffusionProcess object.
    if args.diffusion_process == "marginal":
        dp = MarginalDiffusionProcess(
            args.diffusion_steps, marg_dist=get_marg_dist(root_path=args.data_path)
        )
    elif args.diffusion_process == "equivariant":
        dp = EquivariantDiffusionProcess(args.diffusion_steps)
    else:
        print("Only marginal and equivariant diffusion processes are supported.")

    train_tform = get_train_transform(dp, predict_x0=args.predict_x0)
    hydra_transform = get_hydra_transform()

    # Print prediction mode
    if args.predict_x0:
        print("Training in x0-prediction mode: model will predict original positions")
    else:
        print(
            "Training in epsilon-prediction mode: model will predict noise/deformation"
        )

    # Load the QM9 dataset from torch WITHOUT transform (we'll apply it during training)
    if os.path.splitext(args.data_path)[1] == ".bp":
        opt = {
            "preload": False,
            "shmem": shmem,
            "ddstore": False,
            "var_config": model.module.var_config,
        }
        comm = MPI.COMM_WORLD
        trainset = AdiosDataset(args.data_path, "trainset", comm, **opt)
        valset = AdiosDataset(args.data_path, "valset", comm, **opt)
        testset = AdiosDataset(args.data_path, "testset", comm, **opt)
    else:  # any data_path that doesn't have .bp is treated as QM9.
        dataset = torch_geometric.datasets.QM9(
            root=args.data_path, transform=hydra_transform
        )

    # Limit the number of samples if specified.
    if args.samples != None:
        dataset = dataset[: args.samples]
    else:
        print("Training on Full Dataset")

    # Make all graphs fully connected.
    dataset = [FullyConnectGraph()(data) for data in dataset]  # Apply to all graphs

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
    # I've found that the output_dim sometimes get overwritten, depending
    # on things that happen during dataloader construction. To surmount this,
    # I directly specify the output dimension.
    config = config_utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Architecture"]["output_dim"] = [5, 3]

    # Save the config with all the updated values and initialize wandb
    wandb_run = configure_wandb(project_name="graph diffusion model", config=config)

    # Create the model from the config specifications
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    # Define training optimizer and scheduler
    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Run training with the given model and dataset.
    model = train_model(
        model,
        diffusion_loss,
        optimizer,
        train_loader,
        val_loader,
        config["NeuralNetwork"]["Training"]["num_epoch"],
        logger=wandb_run,
        scheduler=scheduler,
        train_transform=train_tform,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Create default log name if not specified.

    parser.add_argument("-s", "--samples", default=100, type=int)
    parser.add_argument("-ds", "--diffusion_steps", default=100, type=int)
    parser.add_argument("-dp", "--diffusion_process", default="equivariant", type=str)
    parser.add_argument(
        "-c", "--config_path", type=str, default="examples/qm9/test.json"
    )
    parser.add_argument("-d", "--data_path", type=str, default="examples/qm9/dataset")
    parser.add_argument(
        "--predict_x0",
        action="store_true",
        help="Train model to predict original positions (x0) instead of noise",
    )

    # Store the arguments in args.
    args = parser.parse_args()
    print(args)
    train(args)
