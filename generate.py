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


from src.utils import diffusion_utils as du
from src.processes.diffusion import DiffusionProcess
from src.processes.equivariant_diffusion import EquivariantDiffusionProcess
from src.processes.marginal_diffusion import MarginalDiffusionProcess
from src.utils.train_utils import train_model, get_train_transform, insert_t


def pred_fn(model, data, dp):
    data_tx, _ = insert_t(data, data.t, dp.timesteps)
    model.eval()
    out = model(data_tx)
    atom_ident_noise, atom_pos_noise = out[0], out[1]
    noise_data = data.clone().detach_()
    noise_data.x = atom_ident_noise
    noise_data.pos = atom_pos_noise
    return noise_data

def load_model(args):
    # load the config
    with open(os.path.join(args.run_name,'config.json'),'r') as f:
        config = json.load(f)

    verbosity = config["Verbosity"]["level"]

    # Create the model from the config specifications
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    # Distribute the model across ranks (if necessary).
    model = get_distributed_model(model, verbosity)
    model.load_state_dict(torch.load(os.path.join(args.run_name,"model.pth")))
    return config, model

def generate(args):

    # load model, get device
    _, model = load_model(args)
    device = hydragnn.utils.distributed.get_device()

    # setup diffusion process
    dp = MarginalDiffusionProcess(
        args.diffusion_steps, marg_dist=du.get_marg_dist(root_path=args.data_path)
    )

    # Define prior distribution for the generative model
    prior_dist_state = dp.prior_dist(torch.randint(5, 20, (args.num_gen,)), 5, 3).to(
        device
    )

    # Sample from the prior distribution
    prior_samples = dp.sample_from_dist(prior_dist_state)

    # Denoise samples and generate data
    predictor = lambda data: pred_fn(model, data, dp)
    gen_data = dp.reverse_process_sample(prior_samples, predictor)

    gen_data_list = gen_data.to_data_list()
    # Write PDB files for generated data
    for i, gd in enumerate(gen_data_list):
        # postprocess by subtracting off CoM
        gd.pos = gd.pos - gd.pos.mean(dim=0, keepdim=True)
        out_path = os.path.join('models','test','structures',f"gen_{i}.pdb")
        du.write_pdb_file(gd, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Create default log name if not specified.
    parser.add_argument("-g", "--num_gen", type=int, default=100)
    parser.add_argument("-d", "--data_path", type=str, default='../examples/qm9/dataset')
    parser.add_argument("-ds", "--diffusion_steps", type=int, default=100) 
    parser.add_argument("-l", "--run_name", type=str, default='test')
    # parser.add_argument("-m", "--model", type=str, help='path to the trained model' )

    # Store the arguments in args.
    args = parser.parse_args()
    generate(args)