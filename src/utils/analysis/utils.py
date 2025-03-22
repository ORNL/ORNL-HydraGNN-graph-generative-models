import numpy as np
import rdkit

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA

def ECFP_from_smiles(smiles, R=2, L=2**10, use_features=False, use_chirality=False):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    molecule = AllChem.MolFromSmiles(smiles)
    feature_list = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=R,
        nBits=L,
        useFeatures=use_features,
        useChirality=use_chirality,
    )
    return feature_list

def getcolordensity_contour(xdata, ydata):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)

    return BCTX, BCTY, hist2d #plt contourf


def draw_molecules(molecules, prefix, molsPerRow=3, maxMols=100):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    best_molecules = [Chem.MolFromSmiles(smiles) for smiles in molecules]
    best_svg = rdkit.Chem.Draw.MolsToGridImage(
        best_molecules,
        molsPerRow=molsPerRow,
        subImgSize=(300, 300),
        useSVG=True,
        maxMols=maxMols,
    )
    with open(f"{prefix}.svg", "w") as f:
        f.write(best_svg.data)

    best_png = rdkit.Chem.Draw.MolsToGridImage(
        best_molecules,
        molsPerRow=molsPerRow,
        subImgSize=(300, 300),
        returnPNG=True,
        maxMols=maxMols,
    )
    with open(f"{prefix}.png", "wb") as f:
        f.write(best_png.data)

def run_pca(data: np.array, n_components: int = 2):
    """
    Run PCA on data with n_components. Returns transformed data
    and the PCA object.

    Args:
    -----
    data (np.array):
        Data to run PCA on
    n_components (int): 
        Number of components to keep
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca