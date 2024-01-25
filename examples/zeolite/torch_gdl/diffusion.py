from torch import Tensor
import torch
from typing import Callable, Optional, Tuple
from torch_geometric.data import Data



class DiffusionProcess:

    def __init__(self, timesteps: int):
        # check the arguments to init
        assert isinstance(timesteps, int)

        self.timesteps = timesteps

    def check_args(self, t: Tensor = None):
        if t is not None:
            assert isinstance(t, Tensor)
            assert t.dtype() is torch.int or t.dtype() is torch.long
            assert (t >= 0).all() and (t < self.timesteps).all()

    def check_state(self, state: Data):
        assert isinstance(state, Data)

    def sample_prior(self, samples_shape: torch.Size) -> Data:
        """
        Sample from the prior distribution. This could potentially be combined with sample_from_dist

        Args:
        sample_prior (torch.Size): shape of samples to be created
        """
        pass

    def sample_from_dist(self, dist_state: Data) -> Data:
        """
        Sample from the diffusion process at a point in time captured by dist_state

        Args:
            dist_state (Data): data structure defining the distribution state

        Returns:
            Data: data sample from the distribution specified by dist_state params
        """
        pass


    def forward_process_dist(self, state: Data, t: Tensor) -> Data:
        """
        Starting from state, compute the result of the forward process up to time t. (i.e., z_t|x)

        Args:
            state (Data): initial configuration of x. assume t=0 here
            t (Tensor): time(s) at which to compute the forward process result

        Returns:
            Data: forward process result z_t|x
        """
        pass

    def forward_process_sample(self, state: Data, t: Tensor) -> Data:
        """
        Get samples from the forward process z_t|z

        Args:
            state (Data): state before forward_step (z_t)
            t (Tensor): time(s) corresponding to state z_t
        
        Returns:
            Data: forward process result z_{t+1}|z_t
        """
        dist_state = self.forward_process_dist(state, t)
        samples = self.sample_from_dist(dist_state)
        return samples


    def reverse_process_sample(self, state: Data, pred_fn: Callable[[Data,], Data]) -> Data:
        """
        Predict z_t|z_T. Starting from a sample from p(z_T), compute the reverse by iteratively calling reverse_step

        Args:
            state (Data): state drawn from p(z_T)
            t (Tensor): time(s) at which to compute state z_t
        """
        # iterate through time in reverse, calling reverse_pred to yield dist_state
        # then sample using dist_state to construct the generative diffusion process
        # reversed_tsteps = reversed(range(self.timesteps))
        next_state = state.clone()
        for _ in range(self.timesteps):
            dist_state = self.reverse_pred(next_state, pred_fn)
            dist_samples = self.sample_from_dist(dist_state)
            next_state = dist_samples
        return next_state

    def reverse_pred(self, state: Data, pred_fn: Callable[[Data,], Data]) -> Data:
        """
        Predict q(z_{t-1} | z_t) . Calls pred_fn to construct parameters for the sampling distribution

        Args:
            state (Data): input state z_t
            t (Tensor): time corresponding to input state

        Returns:
            Data: parameters for q(z_{t-1}|z_t)
        """
        pass

        

    ########### Relevant to Training ##############

    def get_loss_fn(self, state: Data) -> Callable[[Tensor, Tensor, Optional[int]], Tuple[Tensor, Tensor]]: # roughly the "loss" function of HydraGNN
        
        def loss_fn(pred: Tensor, target: Tensor, head_index: int = None):
            raise NotImplementedError("get_loss_fn not yet implemented")

        return loss_fn




if __name__ == "__main__":

    pass