# Import libraries
import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Callable, Union
from torch_gdl.equivariant_diffusion import EquivariantDiffusionProcess, center_gravity
from torch_gdl.utils import create_noise_schedule

# Define global functions
def sample_atoms(noised_atoms: torch.Tensor, state_dim: int):
    """
    Sample an atom types from a marginal distribution.
    Rows of noised_atoms are the marginal distributions of atom types
    for each sample.

    Parameters:
    ------------
    noised_atoms : torch.Tensor
        Noised atoms
    state_dim : int
        State dimension
    """
    return torch.Tensor(
        [
            np.random.choice(
                np.arange(state_dim),
                p=noised_atoms[i, :].numpy().flatten()
            ) for i in range(noised_atoms.shape[0])
        ]
    )

class MarginalDiffusionProcess(EquivariantDiffusionProcess):
    """
    Diffusion process for prior distribution to approach marginal distribution
    of atom types in training population. From DiGress paper (Vignac et al 2023):
    https://arxiv.org/pdf/2209.14734
    """

    def __init__(
            self, 
            timesteps: int,
            marg_dist: Tensor, 
            noise_schedule: Optional[str] = 'cos'
        ):
        """
        Saves arguments as attributes and initializes the noise schedule

        Parameters:
        ------------
        timesteps : int
            Number of time steps for diffusion process
        marg_dist : Tensor
            Marginal distribution of atom types in training population
        noise_schedule : str (Optional)
            Type of noise schedule. Default is 'cos'
        """
        
        # Initialize super class
        super().__init__(timesteps)

        # Initialize arguments as attributes
        self.noise_schedule = noise_schedule
        self.marg_dist = torch.reshape(marg_dist, (-1, 1))
        self.state_dim = self.marg_dist.shape[0]

        # Create noise schedule
        self.alphas = create_noise_schedule(
            timesteps=self.timesteps,
            type=self.noise_schedule
        )
        # Calculate betas from alphas
        self.betas = 1. - self.alphas

    def check_dist_state(self, dist_state: Data):
        # Check atom probabilities
        assert hasattr(dist_state, 'x_probs')
        assert isinstance(dist_state['x_probs'], Tensor)
        # Check mean position
        assert hasattr(dist_state, 'pos_mu')
        assert isinstance(dist_state['pos_mu'], Tensor)
        # Check std deviation of position
        assert hasattr(dist_state, 'pos_sigma')
        # Check time step
        assert hasattr(dist_state, 't')
        assert isinstance(dist_state['t'], int)

    def markov_noising(self, x: Tensor, t: int) -> Tensor:
        """
        Noise atom type using markov process informed by marginal distribution.

        Parameters:
        ------------
        x : Tensor
            Atom types (one-hot encodings)
        t : int
            Time step somewhere between [0, T]
        """
        # Noise atom type (informed markov process)
        x_probs = torch.matmul(x, self.marginal_Qt(t))

        return x_probs
    
    def marginal_Qt(self, t: int) -> Tensor:
        """
        Compute transition matrix Q at time t using marginal distribution information (DiGress)
        Parameters:
        ------------
        t : int
            Time step somewhere between [0, T]
        """
        # Calculate Qt from alphas and betas
        return (
            self.alphas[t] * torch.eye(self.state_dim) + 
            self.betas[t] * (torch.ones_like(self.marg_dist) * self.marg_dist.T)
        )
    
    def gaussian_noising(self, pos: Tensor, t: int) -> tuple:
        """
        Noise position using standard Gaussian noising
        (follows Hoogeboom et al 2022)

        Parameters:
        ------------
        pos : Tensor
            Position of atoms
        t : int
            Time step somewhere between [0, T]
        """
        # Refer to Section 2.1 in Hoogeboom et al to follow procedure
        alpha_ts = self.alphas[t] / self.alphas[0]
        pos_mu_t = alpha_ts*pos
        sigma_t2 = (1. - self.alphas[t]**2)
        sigma_s2 = (1. - self.alphas[0]**2)
        xpos_sigma = (sigma_t2 - (alpha_ts**2)*sigma_s2)**0.5

        return (pos_mu_t, xpos_sigma)
    
    def forward_process_dist(self, state: Data, t: Tensor) -> Data:
        """
        Starting from state, compute the result of the forward process up to time t. (i.e., z_t|x)

        Args:
            state (Data): initial configuration of x. assume t=0 here
            t (Tensor): time(s) at which to compute the forward process result

        Returns:
            Data: forward process result z_t|x
        """
        # Extract atom type and pos from state
        x, pos = state.x, state.pos

        # Noise and sample new atom types (one-hot encodings)
        x_probs = self.markov_noising(x, t)

        # Noise position and get conditional distribution parameters
        pos_mu_t, pos_sigma_t = self.gaussian_noising(pos, t)

        # Create new distribution state
        dist_state = state.clone().detach_()
        dist_state.x_probs = x_probs
        dist_state.pos_mu = pos_mu_t
        dist_state.pos_sigma = pos_sigma_t
        dist_state.pos = None
        dist_state.x = None

        return dist_state
    
    def sample_from_dist(self, dist_state: Data) -> Data:
        """
        Sample from the diffusion process at a point in time captured by dist_state

        Args:
            dist_state (Data): data structure defining the distribution state

        Returns:
            Data: data sample from the distribution specified by dist_state params
        """
        # Check distribution state has necessary attributes
        self.check_dist_state(dist_state)

        # Sample new atom types from x_probs and return one-hot encodings
        x_t = F.one_hot(
            sample_atoms(dist_state['x_probs'], self.state_dim),
            num_classes=self.state_dim
        ).float()   # Cast back to float

        # Sample position noise from distribution
        # subtract center of gravity for position information
        pos_noise = center_gravity(torch.randn_like(dist_state['pos_mu']))
        pos_samples = dist_state['pos_mu'] + dist_state['pos_noise'] * dist_state['pos_sigma']

        # create a new data entry with samples, set y to preds
        data = dist_state.clone().detach_()
        data.x = x_t
        data.pos = pos_samples
        data.ypos = pos_noise
        data.t = dist_state['t']
        data.pos_mu = None
        data.pos_sigma = None

        return data
