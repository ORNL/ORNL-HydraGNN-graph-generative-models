import os, json, sys

import torch
import torch_geometric
from torch_geometric.data import Data, Batch
# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import hydragnn
from torch_gdl import EquivariantDiffusionProcess, DiffusionProcess, utils as gdl_utils
import random
from argparse import ArgumentParser

import numpy as np
from rdkit import Chem
 
def write_pdb_file(data, output_file):
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

def atomic_num_to_one_hot(atom_idents: torch.Tensor) -> torch.Tensor:
    # expect 1-D tensor
    # CHONF

    atom_idents = atom_idents.where(atom_idents != 1, 0) # hydrogen
    atom_idents = atom_idents.where(atom_idents != 6, 1) # carbon
    atom_idents = atom_idents.where(atom_idents != 7, 2) # nitrogen
    atom_idents = atom_idents.where(atom_idents != 8, 3) # oxygen
    atom_idents = atom_idents.where(atom_idents != 9, 4) # fluorine
    
    return torch.nn.functional.one_hot(atom_idents, 5)

def qm9_pre_transform(data: Data):
    # create a fully connected graph. retain the previous edge_index
    data.edge_index_old = data.edge_index
    data.edge_index = gdl_utils.fc_edge_index(data.num_nodes)

    # Set descriptor as element type. ONE HOT
    data.x = atomic_num_to_one_hot(data.z.long()).float() # ONE HOT
    # data.x = data.x[:, :5].float() # CHONF
    return data

def insert_t(data: Data, t: int, T: int):
    data_ins = data.clone().detach_()
    # concatenate node features and time
    time_vec = torch.ones((data_ins.num_nodes, 1), device=data_ins.x.device) * t / (T - 1.)
    data_ins.x = torch.hstack([data_ins.x, time_vec]) # n_nodes, 6
    return data_ins, time_vec

def qm9_pre_filter(data):
    return data.idx < num_samples

def get_train_transform(dp: DiffusionProcess, head_types: list, out_indices: list, graph_feature_dim: list, node_feature_dim: list):

    def train_transform(data: Data):

        data.t = 0 # default
        
        # randomly sample a t
        t = random.randint(0, dp.timesteps-1)


        data = dp.forward_process_sample(data, t) # should be attaching t to node features


        data, time_vec = insert_t(data, data.t, dp.timesteps)


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

def create_time_string():
    now = datetime.datetime.now()
    return f"run_{now.strftime('%Y%m%d')}_{now.strftime('%H')}_{now.strftime('%M')}"


if __name__ == "__main__":

    parser = ArgumentParser()

    # Create default log name if not specified.
    default_log_name = create_time_string()

    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-n", "--n_gen", type=int, default=100)
    parser.add_argument("-ds", "--diffusion_steps", type=int, default=100)
    parser.add_argument("-ln", "--log_name", type=str, default=default_log_name)

    args = parser.parse_args()

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    num_samples = 130000 #100

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
    with open(filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.setup_ddp()

    log_name = 'cont_' + args.log_name
    # Enable print to log file.
    hydragnn.utils.setup_log(log_name)

    voi = config["NeuralNetwork"]["Variables_of_interest"]

    # Use built-in torch_geometric dataset.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
    dp = EquivariantDiffusionProcess(args.diffusion_steps)
    train_tform = get_train_transform(dp, voi["type"], voi["output_index"], [], voi["output_dim"])
    dataset = torch_geometric.datasets.QM9(
        root="dataset/qm9", pre_transform=qm9_pre_transform, pre_filter=qm9_pre_filter, transform=train_tform)
    
    #NOTE: Commenting this section, fixes bug
    # datum = dataset[0]
    # print("X: ", datum.x)
    # print("EDGE: ", datum.edge_index)

    # print("---------Diffused Version---------")
    # datum = train_tform(datum)
    # print("Time: ", datum.t)
    # print("X: ", datum.x)
    # print("EDGE: ", datum.edge_index)
    # print("Y: ", datum.y)
    # print("POS: ", datum.pos)
    # print("YPOS: ", datum.ypos)
    # print("Y_shape: ", datum.y.shape)

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
    writer = hydragnn.utils.get_summary_writer(log_name, logger='wandb', config=config)
    hydragnn.utils.save_config(config, log_name)

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

    if args.n_gen > 0:
        # load in HydraGNN model
        hydragnn.utils.model.load_existing_model(model, log_name)
        device = hydragnn.utils.get_device()
        # generate structures and put in .pdb

        def pred_fn(data: Data) -> Data:
            data_tx, _ = insert_t(data, data.t, dp.timesteps)
            model.eval()
            out = model(data_tx)
            atom_ident_noise, atom_pos_noise = out[0], out[2]
            noise_data = data.clone().detach_()
            noise_data.x = atom_ident_noise
            noise_data.pos = atom_pos_noise
            return noise_data
        
        prior_dist_state = dp.prior_dist(torch.randint(5, 20, (args.n_gen,)), 5, 3).to(device)
        # [print(pds) for pds in prior_dist_state.to_data_list()]
        prior_samples = dp.sample_from_dist(prior_dist_state)
        gen_data = dp.reverse_process_sample(prior_samples, pred_fn)

        print("############# GEN DATA ################")
        gen_data_list = gen_data.to_data_list()

        for i, gd in enumerate(gen_data_list):
            # postprocess by subtracting off CoM
            gd.pos = gd.pos - gd.pos.mean(dim=0, keepdim=True)
            filename = f'./logs/{log_name}/structures/gen_{i}.pdb'
            # check if directory exists, if not create it
            if not os.path.exists(f"./logs/{log_name}/structures"):
                os.makedirs(f"./logs/{log_name}/structures")
            write_pdb_file(gd, filename)

