import os, json
import logging
import sys
import argparse

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.preprocess.utils import update_predicted_values
from hydragnn.utils.print_utils import log

from zeolite_dataset import ZeoliteDataset, zeo_norm_data, base_train_path
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph

import torch_gdl

import torch
import torch.distributed as dist

import random

def get_pre_transform(head_types: list, out_indices: list, graph_feature_dim: list, node_feature_dim: list):

    rgtransform = RadiusGraph(1000., max_num_neighbors=10000)

    def pre_transform(data: Data): # hack to get y_loc set
        # normalize the data
        data = zeo_norm_data(data)
        data.t = 0 # default
        # create fully-connected adjacency matrix for HydraGNN processing. select arbitrarily large radius such that normalized .pos
        # data have Prob(fully_connected) ~ 1. Couldn't find documentation on what happens when max_num_neighbors=None, so just setting to large number
        data = rgtransform(data)

        # hack the yloc since the target is created dynamically
        data.y_loc = torch.zeros(1, len(head_types) + 1, dtype=torch.int64, device=data.x.device)
        for i in range(len(node_feature_dim)):
            data.y_loc[i+1] = data.y_loc[i] + data.x.shape[0] * node_feature_dim[i]

        return data
    
    return pre_transform




def get_train_transform(dp: torch_gdl.DiffusionProcess, head_types: list, out_indices: list, graph_feature_dim: list, node_feature_dim: list):

    rgtransform = RadiusGraph(1000., max_num_neighbors=10000)

    def train_transform(data: Data):

        # normalize the data
        data = zeo_norm_data(data) # data does not appear to be normalized here
        data.t = 0 # default

        # create fully-connected adjacency matrix for HydraGNN processing. select arbitrarily large radius such that normalized .pos
        # data have Prob(fully_connected) ~ 1. Couldn't find documentation on what happens when max_num_neighbors=None, so just setting to large number
        data = rgtransform(data)
        
        # randomly sample a t
        t = random.randint(1, dp.timesteps-1)

        data = dp.forward_process_sample(data, t) # should be attaching t to node features

        # concatenate node features and time
        time_vec = torch.ones((data.num_nodes, 1), device=data.x.device) * t / (dp.timesteps)
        data.x = torch.hstack([data.x, time_vec]) # n_nodes, 2

        # set y to the expected shape for HydraGNN. create a hack for noise data by creating a new data with noise in .x and .pos
        x_targ = torch.hstack([data.yx, time_vec])
        noisedata = Data(x=x_targ, pos=data.ypos)
        update_predicted_values(head_types, out_indices, graph_feature_dim, node_feature_dim, noisedata)

        
        # extract .y from the hack
        data.y = noisedata.y
        data.y_loc = noisedata.y_loc

        # assert hasattr(data, "edge_index")
        # print(data.edge_index)

        return data

    return train_transform


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", help="input file", type=str, default="zeolite.json")

    ### ARGUMENTS HERE ###
    parser.add_argument("--dataset_path", help="relative (or absolute) path to the dataset", type=str, default=base_train_path)
    parser.add_argument("--num_diffusion_steps", help="T parameter of diffusion process, the number of steps", type=int, default=100)

    ######################

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    hydragnn.utils.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    print("Logging set up")
    ## create the DiffusionProcess
    dp = torch_gdl.EquivariantDiffusionProcess(args.num_diffusion_steps)
    ## Prep the dataset
    datapath = args.dataset_path if os.path.isabs(args.dataset_path) else os.path.join(dirpwd, args.dataset_path)


    voi = config["NeuralNetwork"]["Variables_of_interest"]
    # pre_tform = get_pre_transform(voi["type"], voi["output_index"], [], voi["output_dim"])
    train_tform = get_train_transform(dp, voi["type"], voi["output_index"], [], voi["output_dim"])
    total = ZeoliteDataset(datapath, train_tform)

    print("Zeolite dataset loaded")

    trainset, valset, testset = split_dataset(
        dataset=total,
        perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
        stratify_splitting=config["Dataset"]["compositional_stratified_splitting"],
    )
    print(len(total), len(trainset), len(valset), len(testset))

    timer = Timer("load_data")
    timer.start()
    # create the datasets
    
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
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

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()

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
        create_plots=True,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    sys.exit(0)
