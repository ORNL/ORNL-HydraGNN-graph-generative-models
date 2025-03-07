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
    type: Optional[str] = "cos",
    alpha: Optional[float] = None,
    s: Optional[float] = 1.0e-5,
) -> torch.Tensor:
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

    if type == "cos":
        # Return tensor of cosine noise schedule
        alphas = torch.cos(torch.pi / 2 * (t / (timesteps + s)) / (1 + s)) ** 2
        # alphas[alphas < 1e-3] = 1e-3
        return alphas
    if type == "edm":
        # Return tensor of cosine noise schedule
        alphas = (1-2*s) * (1-(t/timesteps)**2) + s 
        # alphas[alphas < 1e-3] = 1e-3
        return alphas
    elif type == "linear":
        # Assert alpha is not non if type is linear
        assert alpha is not None, "alpha must be provided for linear noise schedule"
        # Return tensor of linear noise schedule
        return torch.cumprod(torch.pow(alpha, t), dim=0)
    else:
        raise ValueError(
            "Noise schedule type not supported. Please choose 'cos' or 'linear'"
        )
