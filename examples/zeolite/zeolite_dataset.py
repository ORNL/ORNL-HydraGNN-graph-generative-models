import torch
import os
import glob
# from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Tuple, List
# from hydragnn.utils.abstractbasedataset import AbstractBaseDataset as Dataset
from torch_geometric.data import Dataset

base_train_path = "/home/ericyeats/proj/data/zeolite_train/pointcloud_t0.7"
base_test_path = "/home/ericyeats/proj/data/zeolite_test/pointcloud_t0.7"

train_mean = torch.tensor([261.3444, 271.9661, 265.6199])
train_std = torch.tensor([159.5063, 164.1799, 162.2383])

def get_mu_sig(xdim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if xdim == 2: # not batched
        mu, sig = train_mean[None, :], train_std[None, :]
    elif xdim == 3: # batched
        mu, sig = train_mean[None, None, :], train_std[None, None, :]
    else:
        raise NotImplementedError("Expect x dim to be 2 or 3 but got {}".format(xdim))
    return mu, sig

def zeo_norm(x: torch.Tensor) -> torch.Tensor:
    xdim = x.dim()
    mu, sig = get_mu_sig(xdim)
    return (x - mu) / sig

def zeo_inorm(x: torch.Tensor) -> torch.Tensor:
    xdim = x.dim()
    mu, sig = get_mu_sig(xdim)
    return (sig * x) + mu

def zeo_norm_data(data: Data) -> Data:
    data.pos = zeo_norm(data.pos)
    return data

def zeo_inorm_data(data: Data) -> Data:
    data.pos = zeo_inorm(data.pos)
    return data

def pc_from_data(data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    msk = (data.x < 0).squeeze_()
    o_pc = data.pos[msk]
    si_pc = data.pos[torch.logical_not(msk)]
    return o_pc, si_pc

def ten_from_data(data: Data) -> torch.Tensor:
    return torch.hstack([data.pos, data.x])

def data_from_ten(o_pc, si_pc, t):
    pos = torch.vstack([o_pc, si_pc])
    x = torch.vstack([torch.ones((o_pc.shape[0], 1)) * -1, torch.ones((si_pc.shape[0], 1))])
    return Data(pos=pos, x=x, t=t)

class ZeoliteDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.path = root

        # create a registry of .O and .si
        self.ofnames = sorted(glob.glob(os.path.join(self.path, r"*_o.pt")))
        self.sifnames = sorted(glob.glob(os.path.join(self.path, r"*_si.pt")))

        # self.raw_files_names = [of[:-3] for of in self.ofnames] + [sif[:-3] for sif in self.sifnames]

        assert len(self.ofnames) == len(self.sifnames), "Should have the same number of Oxygen files as Si files"

    def len(self):
        return len(self.ofnames)

    def get(self, idx):
        o_pc = torch.load(self.ofnames[idx]).type(torch.float)
        si_pc = torch.load(self.sifnames[idx]).type(torch.float)
        # create a Data object
        pos = torch.vstack([o_pc, si_pc])
        x = torch.vstack([torch.ones((o_pc.shape[0], 1)) * -1., torch.ones((si_pc.shape[0], 1))])
        data = Data(x=x, pos=pos)
        return data
    
############### VERIFICATION #######################

def plot_sample(data: Data, loc='./material.png'):
    with torch.no_grad():
        import matplotlib.pyplot as plt
        data = data.cpu()
        o_pc, si_pc = pc_from_data(data)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(o_pc[:, 0], o_pc[:, 1], o_pc[:, 2], c='r')
        ax.scatter(si_pc[:, 0], si_pc[:, 1], si_pc[:, 2], c='y')

        plt.savefig(loc)
        plt.close()

        print("Material Saved")

###################################################


if __name__ == "__main__":

    with torch.no_grad():
        zd = ZeoliteDataset(base_train_path)

        data_slice = zd[:10]

        print(data_slice)

        data = zeo_norm_data(zd[2])

        plot_sample(data)
        

        # iterate through the dataset to calculate the mean, std of all the pointclouds in X, Y, Z

        # try calculating the mean, std of each point cloud (Si & O), then take average of each
        agg_mean = torch.zeros(3)
        agg_std = torch.zeros(3)
        for i, data in enumerate(zd):
            osi_pc = data.pos
            agg_mean += osi_pc.mean(dim=0)
            agg_std += osi_pc.std(dim=0)

            if (i+1) % 100 == 0:
                print(i+1, " of ", len(zd))

        print("Ave Mean: ", agg_mean / len(zd))
        print("Ave Std: ", agg_std / len(zd))