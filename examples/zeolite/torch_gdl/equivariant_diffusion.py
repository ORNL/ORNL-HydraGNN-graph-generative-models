import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Callable, Tuple
import numpy as np
from .diffusion import DiffusionProcess

def center_of_gravity(x: Tensor) -> Tensor:
    # x = x - x.mean(dim=list(range(x.dim())[1:]), keepdim=True)
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

def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
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
            alphas = polynomial_schedule(timesteps) # default schedule
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
        pos_noise = center_of_gravity(torch.randn_like(pos_mu)) # subtract center of gravity for position information
        
        x_samples = x_mu + x_noise*x_sigma
        pos_samples = pos_mu + pos_noise*pos_sigma

        # create a new data entry with samples, set y to preds
        data = Data(x=x_samples, pos=pos_samples, yx=x_noise, ypos=pos_noise, t=t, edge_index=dist_state.edge_index)
        return data


    def forward_process_dist(self, state: Data, t: int) -> Data:
        x, pos, s = state.x, state.pos, state.t
        # compute the SNR params from s to t
        alpha_ts = self.alphas[t] / self.alphas[s]
        x_mu_t, pos_mu_t = alpha_ts*x, alpha_ts*pos
        sigma_t2 = (1. - self.alphas[t]**2)
        sigma_s2 = (1. - self.alphas[s]**2)
        xpos_sigma = (sigma_t2 - (alpha_ts**2)*sigma_s2)**0.5
        x_sigma_t, pos_sigma_t = xpos_sigma, xpos_sigma

        dist_state = Data(x_mu=x_mu_t, pos_mu = pos_mu_t, x_sigma=x_sigma_t, pos_sigma=pos_sigma_t, t=t, edge_index=state.edge_index)
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
        noise_state = pred_fn(state) # subtract positional center of mass to maintain equivariance
        noise_state.pos = noise_state.pos - noise_state.pos.mean(dim=0, keepdim=True)

        # create dist_state and decrement t
        alpha_ts = self.alphas[t]/self.alphas[s]
        var_t = 1. - self.alphas[t]**2
        var_s = 1. - self.alphas[s]**2
        var_ts = var_t - (alpha_ts**2)*var_s

        x_mu_s = state.x/alpha_ts - (var_ts/(alpha_ts*(var_t**0.5)))*noise_state.x
        pos_mu_s = state.pos/alpha_ts - (var_ts/(alpha_ts*(var_t**0.5)))*noise_state.pos

        x_sigma_s = (var_ts*var_s/var_t)**0.5
        pos_sigma_s = (var_ts*var_s/var_t)**0.5

        dist_state = Data(x_mu=x_mu_s, pos_mu=pos_mu_s, x_sigma=x_sigma_s, pos_sigma=pos_sigma_s, t=s, edge_index=state.edge_index)

        return dist_state

    def get_loss_fn(self, state: Data): # roughly the "loss" function of HydraGNN

        targ_feat = torch.hstack([state.ypos, state.yx])
        t = state.t

        # weight according to time
        w_t = 1. # - self.snr_t(t-1)/self.snr_t(t)
        
        def loss_fn(pred: torch.Tensor):
            # hstack the features
           #  pred_feat = torch.hstack([pred.pos, pred.x])
            feat_loss = (targ_feat - pred).square().mean()
            return w_t * feat_loss


        return loss_fn

