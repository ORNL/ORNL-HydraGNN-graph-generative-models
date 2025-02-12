import os
import torch
import torch_geometric
import numpy as np
from typing import Optional, Callable, Union
from rdkit import Chem

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

def create_noise_schedule(
          timesteps, 
          type: Optional[str] = 'cos',
          alpha: Optional[float] = None, 
          s: Optional[float] = 1.e-3) -> torch.Tensor:
        """
        Create a noise schedule for the diffusion process

        Parameters:
        ------------
        type : str
            Type of noise schedule. Default is 'cos'
        alpha : float
            Alpha value for linear noise schedule. Only
            needed if type is 'linear'.
        s : float
            Small value to avoid division by zero
        """
        # Create an incremental tensor from 1 to timesteps
        t = torch.arange(timesteps) + 1

        if type == 'cos':
            # Return tensor of cosine noise schedule
            alphas = torch.cos(torch.pi/2 * (t / (timesteps + s)) / (1 + s))**2
            alphas[alphas<1e-3] = 1e-3
            return alphas
        elif type == 'linear':
            # Assert alpha is not non if type is linear
            assert alpha is not None, "alpha must be provided for linear noise schedule"
            # Return tensor of linear noise schedule
            return torch.cumprod(torch.pow(alpha, t), dim=0)
        else:
            raise ValueError("Noise schedule type not supported. Please choose 'cos' or 'linear'")


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
    atom_map = [1, 6, 7, 8, 9]  # HCNOF
    atom_type = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
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