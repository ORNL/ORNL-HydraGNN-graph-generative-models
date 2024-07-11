# %% Import necessary libraries
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from torch_geometric.datasets import QM9

# %% Load the QM9 dataset
dataset = QM9(root="./qm9_dataset")

# %% Find the marginal distributino of atom types in the dataset
# Sum the atom types to get the marginal distribution
total_atom_types = dataset.x[:, :5].sum(dim=0)

# Normalize the distribution
marginal_distribution = total_atom_types / total_atom_types.sum()
# dist_4_plotting = torch.round(marginal_distribution, decimals=2)

# %% Plot the marginal distribution as a bar plot
# Create figure and plot
fig = plt.figure()
plt.bar(range(5), marginal_distribution)

# Plot the CDF of the marginal distribution as a line plot
plt.plot(range(5), marginal_distribution.cumsum(-1), color="red")
# Add a legend
plt.legend(["CDF", "Marginal Distribution"])

# Add labels to the x-axis
plt.xticks(range(5))

# Add labels and title
plt.xlabel("Atom Type")
plt.ylabel("Frequency (%)")
plt.title("Marginal Distribution of Atom Types in QM9 Dataset")
plt.show()

# %% Define parameters and class for the noise model

# Set random seed for reproducibility
# torch.manual_seed(42) # Can remove later

# Create class for the noise model
class NoiseModel():
    def __init__(self, alpha, T, eps, marg_dist, num_atoms=5):
        # Initialize arguments as class attributes
        self.alpha = alpha # Only used if noise schedule is linear
        self.T = T
        self.eps = eps
        self.marg_dist = torch.reshape(marg_dist, (-1, 1))
        self.num_atom_types = self.marg_dist.shape[0]
        self.num_atoms = num_atoms
        

        # Initialize initial molecule to a random matrix whose rows are one hot vectors
        self.molecule = F.one_hot(
            torch.argmax(torch.rand((self.num_atoms, self.num_atom_types)), dim=1),
            num_classes=self.num_atom_types
        )

        # For trajectory plotting, initialize empty list to store atom types over time
        self.noised_mols = []

        # Keep track of alpha_bar and beta_bar over time
        self.alpha_track = []
        self.beta_track = []
    
    def noiseSchedule(self, t, type='cos'):
        """
        Compute alpha at time t
        -----------------------------------
        Parameters:
        t : int
            Time step
        """
        # Compute alpha_bar and beta_bar
        if type == 'cos':
            return np.cos(np.pi/2 * (t/(self.T+self.eps))/ (1 + self.eps)) ** 2
        else:
            return torch.prod(torch.pow(self.alpha, torch.arange(t)+1))
    
    def marginalQ(self, t):
        """
        Computing transition matrix Q at time t using marginal distribution information (DiGress)
        -----------------------------------
        Parameters:
        t : int
            Time step
        """
        # Compute alpha_bar and beta_bar
        self.alpha_bar = self.noiseSchedule(t)
        self.beta_bar = 1 - self.alpha_bar
        # Store alpha_bar and beta_bar
        self.alpha_track.append(self.alpha_bar)
        self.beta_track.append(self.beta_bar)
        
        # Compute transition matrix Q at time t
        Qt = (
            self.alpha_bar * torch.eye(self.num_atom_types) + 
            self.beta_bar * (torch.ones_like(self.marg_dist) * self.marg_dist.T)
        )

        return Qt
    
    def snr_t(self, t=None):
        """
        Compute the noised input at time t
        -----------------------------------
        Parameters:
        t : int
            Time step
        """
        # If None, randomly sample a time step t from a uniform distribution ranging from 0 to T
        if t is None:
            t = torch.randint(0, self.T, (1,))
        # Create transition matrix at time t    
        Qt_bar = self.marginalQ(t)
        # Compute noised molecule and time t
        noised_mol = torch.matmul(self.molecule.to(torch.float32), Qt_bar)
        # Update the molecule to the noised molecule at time t
        self.molecule = F.one_hot(
            torch.tensor(
                [
                    np.random.choice(
                        np.arange(self.num_atom_types),
                        p=noised_mol[i, :].numpy().flatten()
                    ) for i in range(noised_mol.shape[0])
                ]
            ),
            num_classes=self.num_atom_types
        )
    
    def create_noise_trajectory(self):
        """
        Create a trajectory of noised inputs over time
        """
        # Loop through the time steps
        for t in range(self.T):
            # Compute the noised input at time t
            self.snr_t(t+1)
            # Append the noised input to the list
            # self.noised_mols.append(self.molecule.argmax()) #Change to this later: self.molecule.argmax(dim=1).reshape(-1, 1)
            self.noised_mols.append(self.molecule.argmax(dim=1).reshape(-1, 1))

# %% Create class instatiation and plot the trajectory of noised inputs
# Define number of atoms in molecule
num_atoms = 8
# Instantiate the NoiseModel class
noise_model = NoiseModel(
    alpha=0.9, 
    T=100, 
    eps=0.01, 
    marg_dist=marginal_distribution,
    num_atoms=num_atoms
)

# Create the trajectory of noised inputs
noise_model.create_noise_trajectory()

# Extract trajectories for plotting
trajectories = torch.hstack(noise_model.noised_mols)

# Plot the trajectory of noised inputs
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
colors = cm.rainbow(np.linspace(0, 1, num_atoms))
for i in range(num_atoms):
    ax[0].plot(trajectories[i, :], color=colors[i])
# ax[0].plot(trajectories)

# Plot alpha_bar and beta_bar over time
ax[1].plot(noise_model.alpha_track, label=r"$\alpha_{bar}$")
ax[1].plot(noise_model.beta_track, label=r"$\beta_{bar}$")

# Add labels and legends
ax[0].set_title("Trajectory of Noised Inputs")
ax[0].set_ylabel("Atom Type")
ax[1].set_title(r"$\alpha_{bar}$ and $\beta_{bar}$ over time")
ax[1].legend()
ax[1].set_xlabel("Time")

# %%
