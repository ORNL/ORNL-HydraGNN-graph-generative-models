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

## debug imports
from debug import print_dict_differences
import copy
 
def write_pdb_file(data, output_file):
    """
    Create a PDB file from the output of the denoising model.
    Creates a RDKit molecule and writes it to a PDB file to make
    loading files in RDKit easier when analyzing.

    Args:
    -----
    data (Data):
        The output of the denoising model.
    output_file (str):
        The path to the output PDB file.
    """
    atom_map = [1, 6, 7, 8, 9] # HCNOF
    atom_type = {
        1: 'H',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F'
    }
    # Get atom types
    x_argmax = torch.argmax(data.x, dim=1).cpu().numpy()
    atoms = [atom_type[atom_map[t]] for t in x_argmax]
    # Get positions
    pos = data.pos.cpu().numpy().tolist()

    # Create Empty Mol
    mol = Chem.RWMol()

    # Add atoms to mol
    for atom in atoms:
        mol.AddAtom(Chem.Atom(atom))

    # Add a new conformer. The argument 0 means that it will have 0 atoms, but atoms will be added later.
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    # Add atom positions
    for i, p in enumerate(pos):
        mol.GetConformer().SetAtomPosition(i, p)

    # Save molecule to PDB file
    Chem.MolToPDBFile(mol, output_file)

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
    time_vec = torch.ones((data_ins.num_nodes, 1), device=data_ins.x.device) * t / (T - 1.)
    data_ins.x = torch.hstack([data_ins.x, time_vec]) # (n_nodes, 6)
    return data_ins, time_vec


def get_train_transform(
        dp: DiffusionProcess,
        head_types: list,
        out_indices: list,
        graph_feature_dim: list,
        node_feature_dim: list):
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

        data.t = 0 # default

        # Only use atom type features
        data.x = data.x[:, :5].float()
        
        # randomly sample a t
        t = random.randint(0, dp.timesteps-1)

        data = dp.forward_process_sample(data, t) # should be attaching t to node features

        data, time_vec = insert_t(data, data.t, dp.timesteps)

        # set y to the expected shape for HydraGNN. Create a hack for 
        # noise data by creating a new data with noise in .x and .pos
        x_targ = torch.hstack([data.x_probs, time_vec])
        noisedata = Data(x=x_targ, pos=data.ypos)
        update_predicted_values(
            head_types, out_indices, graph_feature_dim, node_feature_dim, noisedata
        )
        # extract .y from the hack
        data.y = noisedata.y
        data.y_loc = noisedata.y_loc

        return data

    return train_transform

def get_marg_dist(root_path: str) -> torch.Tensor:
    """
    Returns the marginal distribution of atom types in the QM9 dataset.
    Used for the noising model for discrete node features.

    Args:
    -----
    root_path (str):
        The path to the (stored) QM9 dataset.
    """
    # Load the QM9 dataset
    qm9 = torch_geometric.datasets.QM9(root=os.path.join(root_path))

    # Find the marginal distribution of atom types in the dataset
    # Sum the atom types to get the marginal distribution
    total_atom_types = qm9.x[:, :5].sum(dim=0)
    # Normalize the distribution and return
    return total_atom_types / total_atom_types.sum()

def create_time_string():
    """
    Creates a time string for the log file name. Used for
    default log name if not specified.
    """
    now = datetime.datetime.now()
    return f"run_{now.strftime('%Y%m%d')}_{now.strftime('%H')}_{now.strftime('%M')}"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Create default log name if not specified.
    default_log_name = create_time_string()

    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-g", "--gen", type=int, default=100)
    parser.add_argument("-s", "--samples", type=int)
    parser.add_argument("-d", "--diffusion_steps", type=int, default=100)
    parser.add_argument("-l", "--log_name", type=str, default=default_log_name)

    # Store the arguments in args.
    args = parser.parse_args()

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Configurable run choices (JSON file that accompanies this example script).
    file_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(file_path, "qm9_marginal.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # Always initialize for multi-rank training.
    world_size, world_rank = setup_ddp()

    log_name = args.log_name
    # Enable print to log file.
    setup_log(log_name)

    voi = config["NeuralNetwork"]["Variables_of_interest"]

    """
    Use built-in torch_geometric dataset.
    NOTE: data is moved to the device in the pre-transform.
    NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
    """
    # Set path to QM9 dataset.
    qm9_path = os.path.join(file_path, 'dataset')
    # Create a MarginalDiffusionProcess object.
    dp = MarginalDiffusionProcess(
        args.diffusion_steps,
        marg_dist=get_marg_dist(root_path=qm9_path)
    )
    # Create a training transform function for the QM9 dataset.
    train_tform = get_train_transform(dp, voi["type"], voi["output_index"], [], voi["output_dim"])
    # Load the QM9 dataset from torch with the pre-transform, pre-filter, and train transform.
    dataset = torch_geometric.datasets.QM9(
        root=qm9_path,
        transform=train_tform
    )

    # Limit the number of samples if specified.
    if args.samples != None:
        dataset = dataset[:args.samples]
    else:
        print("Training on Full Dataset")
    
    # Split into train, validation, and test sets.
    train, val, test = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )
    # Create dataloaders for PyTorch training
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    print(train[0])
    print(train[0].y_loc)
    # Update the config with the dataloaders.
    config = config_utils.update_config(config, train_loader, val_loader, test_loader)
    # Create the model from the config specifications
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    # Distribute the model across ranks (if necessary).
    model = get_distributed_model(model, verbosity)

    # Define training optimizer and scheduler
    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # Create a summary writer and save the config.
    writer = hydragnn.utils.model.get_summary_writer(log_name)
    print('***')
    for batch in train_loader:
        pred = model(batch)
        print(len(pred))
        break
    # Run training with the given model and qm9 dataset.
    if args.train:
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

    # Generate molecules if specified.
    if args.gen > 0:
        # load in HydraGNN model
        hydragnn.utils.model.load_existing_model(model, log_name)
        device = hydragnn.utils.get_device()
        
        # Define a prediction function for the denoising model.
        def pred_fn(data: Data) -> Data:
            data_tx, _ = insert_t(data, data.t, dp.timesteps)
            model.eval()
            out = model(data_tx)
            atom_ident_noise, atom_pos_noise = out[0], out[1]
            noise_data = data.clone().detach_()
            noise_data.x = atom_ident_noise
            noise_data.pos = atom_pos_noise
            return noise_data
        
        # Define prior distribution for the generative model
        prior_dist_state = dp.prior_dist(torch.randint(5, 20, (args.gen,)), 5, 3).to(device)
        # Sample from the prior distribution
        prior_samples = dp.sample_from_dist(prior_dist_state)
        # Denoise samples and generate data
        gen_data = dp.reverse_process_sample(prior_samples, pred_fn)

        print("############# GEN DATA ################")
        gen_data_list = gen_data.to_data_list()
        # Write PDB files for generated data
        for i, gd in enumerate(gen_data_list):
            # postprocess by subtracting off CoM
            gd.pos = gd.pos - gd.pos.mean(dim=0, keepdim=True)
            out_path = f'./logs/{log_name}/structures/gen_{i}.pdb'
            # check if directory exists, if not create it
            if not os.path.exists(f"./logs/{log_name}/structures"):
                os.makedirs(f"./logs/{log_name}/structures")
            write_pdb_file(gd, out_path)

