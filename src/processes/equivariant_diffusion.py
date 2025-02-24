import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from typing import Optional, Callable, Union
import numpy as np
from src.processes.diffusion import DiffusionProcess
from src.utils.diffusion_utils import fc_edge_index

def center_gravity(x: Tensor) -> Tensor:
    x = x - x.mean(dim=0, keepdim=True) # average along the sequence
    return x

# from E3 Equivariant Diffusion github: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py#L38
############################################################
def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2

def polynomial_schedule(timesteps: int, s=1e-4, power=1.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

#################################################################

class EquivariantDiffusionProcess(DiffusionProcess):
    """
    E(3) Equivariant Diffusion process on node positions x and node hidden state h.

    x: node positions in 3d
    h: node hidden state in nd

    The forward diffusion process is given by the variance preserving process of Sohl-Dickstein et al. 2015
    with the twist that the node position Gaussian distribution is on the 3D subspace corresponding to zero center of mass
    """

    def __init__(self, timesteps: int, alphas: Optional[Tensor] = None):
        super().__init__(timesteps)
        if alphas is None:
            alphas = polynomial_schedule(timesteps, power=1.) # default schedule
        self.alphas = alphas

    def snr_t(self, t: int):
        alphat2 = self.alphas[t]**2
        vart = (1. - alphat2)
        return alphat2 / vart

    def check_state(self, state: Data):
        super().check_state(state)
        assert hasattr(state, 'pos') and isinstance(state['pos'], Tensor) # node position
        assert hasattr(state, 'x') and isinstance(state['x'], Tensor)# node hidden attributes

    def check_dist_state(self, dist_state: Data):
        assert hasattr(dist_state, 'x_mu')
        assert isinstance(dist_state['x_mu'], Tensor)
        assert hasattr(dist_state, 'x_sigma')
        # assert isinstance(dist_state['x_sigma'], Tensor)
        assert hasattr(dist_state, 'pos_mu')
        assert isinstance(dist_state['pos_mu'], Tensor)
        assert hasattr(dist_state, 'pos_sigma')
        # assert isinstance(dist_state['pos_sigma'], Tensor)
        assert hasattr(dist_state, 't')
        assert isinstance(dist_state['t'], int)

    def sample_from_dist(self, dist_state: Data):
        self.check_dist_state(dist_state)
        # get the distribution parameters
        x_mu = dist_state['x_mu']
        x_sigma = dist_state['x_sigma']
        pos_mu = dist_state['pos_mu']
        pos_sigma = dist_state['pos_sigma']
        t = dist_state['t']

        # sample noise from the distribution
        x_noise = torch.randn_like(x_mu)
        pos_noise = center_gravity(torch.randn_like(pos_mu)) # subtract center of gravity for position information
        
        x_samples = x_mu + x_noise*x_sigma
        pos_samples = pos_mu + pos_noise*pos_sigma

        # create a new data entry with samples, set y to preds
        data = dist_state.clone().detach_()
        data.x = x_samples # noised atom features
        data.pos = pos_samples # noised atom positions
        data.x_t = x_noise # noise in atom features
        data.ypos = pos_noise # noise in atom positions
        data.t = t
        data.x_mu = None
        data.x_sigma = None
        data.pos_mu = None
        data.pos_sigma = None

        return data

    def prior_dist(self, state_dims: Union[torch.Tensor, int], x_dim: int, pos_dim: int) -> Data:

        if isinstance(state_dims, int):
            assert state_dims > 0
            state_dims = torch.Tensor([state_dims], dtype=torch.int)

        assert isinstance(state_dims, torch.Tensor) and state_dims.dim() == 1
        assert state_dims.dtype in [torch.int, torch.long]

        batch = []
        for d in state_dims:
            d = d.item()
            batch.append(
                Data(
                    x_mu=torch.zeros((d, x_dim), device=state_dims.device),
                    x_sigma=torch.ones((d, x_dim), device=state_dims.device),
                    edge_index=fc_edge_index(d).to(state_dims.device),
                    pos_mu=torch.zeros((d, pos_dim), device=state_dims.device),
                    pos_sigma=torch.ones((d, pos_dim), device=state_dims.device),
                    num_nodes=d,
                    x=torch.zeros((d, x_dim), device=state_dims.device), # include these as a hack w/ Batch.from_data_list()
                    pos=torch.zeros((d, x_dim), device=state_dims.device)   # need for the eventual batch.to_data_list() call
                )
            )
        batch = Batch.from_data_list(batch)
        batch.t = self.timesteps - 1
        return batch


    def forward_process_dist(self, state: Data, t: int) -> Data:
        x, pos, s = state.x, state.pos, state.t
        # compute the SNR params from s to t
        alpha_ts = self.alphas[t] / self.alphas[s]
        x_mu_t, pos_mu_t = alpha_ts*x, alpha_ts*pos
        sigma_t2 = (1. - self.alphas[t]**2)
        sigma_s2 = (1. - self.alphas[s]**2)
        xpos_sigma = (sigma_t2 - (alpha_ts**2)*sigma_s2)**0.5
        x_sigma_t, pos_sigma_t = xpos_sigma, xpos_sigma

        ## replace
        dist_state = state.clone().detach_()
        dist_state.x_mu = x_mu_t
        dist_state.pos_mu = pos_mu_t
        dist_state.x_sigma = x_sigma_t
        dist_state.pos_sigma = pos_sigma_t
        dist_state.t = t
        dist_state.pos = None
        dist_state.x = None

        return dist_state

        

    def reverse_pred(self, state: Data, pred_fn: Callable[[Data,], Data]) -> Data:
        """
        Given noisy state with associated time t, use pred_fn to parameterize the (normalized) Gaussian noise and "remove" noise from state.
        Return new state with updated time t according to the reverse diffusion process

        Args:
            state (Data): state describing the noisy graph object
            pred_fn (Callable): function wrapping a call to the diffusion model. should take a Data and return a Data parameterizing the mean
        Returns:
            Data: denoised sample provided by the denoising process distribution
        """
        t, s = state.t, state.t - 1
        noise_state = pred_fn(state) # pred_fn should subtract positional center of mass to maintain equivariance
        # noise_state.pos = center_gravity(noise_state.pos)

        # create dist_state and decrement t
        alpha_ts = self.alphas[t]/self.alphas[s]
        var_t = 1. - self.alphas[t]**2
        var_s = 1. - self.alphas[s]**2
        var_ts = var_t - (alpha_ts**2)*var_s

        x_mu_s = state.x/alpha_ts - (var_ts/(alpha_ts*(var_t**0.5)))*noise_state.x
        pos_mu_s = state.pos/alpha_ts - (var_ts/(alpha_ts*(var_t**0.5)))*noise_state.pos

        x_sigma_s = (var_ts*var_s/var_t)**0.5
        pos_sigma_s = (var_ts*var_s/var_t)**0.5

        dist_state = state.clone().detach_()
        dist_state.x_mu = x_mu_s
        dist_state.pos_mu = pos_mu_s
        dist_state.x_sigma = x_sigma_s
        dist_state.pos_sigma = pos_sigma_s
        dist_state.t = s
        dist_state.x = None
        dist_state.pos = None

        return dist_state

