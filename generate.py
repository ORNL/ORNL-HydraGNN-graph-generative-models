import os, json, sys, argparse, random, datetime
import torch, torch_geometric
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch, DataLoader

import hydragnn
from hydragnn.utils.print import setup_log
from hydragnn.utils.input_config_parsing import config_utils
from hydragnn.utils.distributed import setup_ddp, get_distributed_model
from hydragnn.preprocess.graph_samples_checks_and_updates import update_predicted_values


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import diffusion_utils as du
from src.diffusion import DiffusionProcess
from src.equivariant_diffusion import EquivariantDiffusionProcess
from src.marginal_diffusion import MarginalDiffusionProcess
from src.utils.train_utils import train_model


def pred_fn(data: Data) -> Data:
    data_tx, _ = insert_t(data, data.t, dp.timesteps)
    model.eval()
    out = model(data_tx)
    atom_ident_noise, atom_pos_noise = out[0], out[1]
    noise_data = data.clone().detach_()
    noise_data.x = atom_ident_noise
    noise_data.pos = atom_pos_noise
    return noise_data


def generate(args):

    # setup diffusion process
    dp = MarginalDiffusionProcess(
        args.diffusion_steps, marg_dist=get_marg_dist(root_path=qm9_path)
    )
    # load in HydraGNN model
    hydragnn.utils.model.load_existing_model(args.model, args.log_name)
    device = hydragnn.utils.get_device()

    # Define prior distribution for the generative model
    prior_dist_state = dp.prior_dist(torch.randint(5, 20, (args.gen,)), 5, 3).to(
        device
    )
    # Sample from the prior distribution
    prior_samples = dp.sample_from_dist(prior_dist_state)
    # Denoise samples and generate data
    gen_data = dp.reverse_process_sample(prior_samples, pred_fn)

    gen_data_list = gen_data.to_data_list()
    # Write PDB files for generated data
    for i, gd in enumerate(gen_data_list):
        # postprocess by subtracting off CoM
        gd.pos = gd.pos - gd.pos.mean(dim=0, keepdim=True)
        out_path = f"./logs/{log_name}/structures/gen_{i}.pdb"
        # check if directory exists, if not create it
        if not os.path.exists(f"./logs/{log_name}/structures"):
            os.makedirs(f"./logs/{log_name}/structures")
        write_pdb_file(gd, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Create default log name if not specified.
    parser.add_argument("-g", "--num_gen", type=int, default=100)
    parser.add_argument("-d", "--diffusion_steps", type=int, default=100) # I think this needs to come from the previous model
    parser.add_argument("-m", "--model", type=str )

    # Store the arguments in args.
    args = parser.parse_args()
    generate(args)