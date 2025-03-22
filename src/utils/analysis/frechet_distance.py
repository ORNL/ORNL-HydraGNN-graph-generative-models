import fcd
import numpy as np
import torch

from torch.utils.data import DataLoader

class FCD():
    def __init__(
            self,
            gen_smiles,
            ref_smiles,
            model_path=None
        ):
        # set smiles as atribute
        self.gen_smiles = gen_smiles
        self.ref_smiles = ref_smiles
        self.model_path = model_path

        # If smiles list is less than 10,000 samples warn user
        if len(self.gen_smiles) < 10000 or len(self.ref_smiles) < 10000:
            print('Warning: FCD metric is most accurate with 10,000 samples or more. FCD varys with sample size.')
            
        # Get model (if None default ChemNet model from fcd lib is used)
        self.model = fcd.load_ref_model(model_path=self.model_path)

        # Get CHEBMLNET activations
        self.gen_act, self.ref_act = self.get_activations()
        # Get mean and std dev of activations
        self.gen_params, self.ref_params = self.get_dist_params(
            self.gen_act, self.ref_act
        )
        # Calculate FCD score
        self.fcd_score = fcd.calculate_frechet_distance(
            mu1=self.gen_params[0],
            mu2=self.ref_params[0],
            sigma1=self.gen_params[1],
            sigma2=self.ref_params[1]
        )
            
    def get_predictions(
            self,
            smiles,
            batch_size,
            num_workers
        ):
        """
        Overwrite the FCD get_predictions method which was causing errors
        when trying to get CHEBMLNET activations of generated and reference
        smile strings. This method excludes the device management during inference
        and uses the CPU device.
        """
        # define dataloader from smiles
        dataloader = DataLoader(
            fcd.utils.SmilesDataset(smiles),
            batch_size=batch_size,
            num_workers=num_workers
        )

        # Define device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Get CHEBMLNET activations
        with torch.no_grad():
            chemnet_activations = []
            for batch in dataloader:
                chemnet_activations.append(
                    self.model(
                        batch.transpose(1, 2).float().to(device)
                    ).to("cpu").detach().numpy().astype(np.float32)
                )
        
        return np.row_stack(chemnet_activations)

    def get_activations(
            self,
            batch_size=128,
            num_workers=0
        ):
        """
        Get CHEBMLNET activations of generated and reference
        smile strings.
        """

        # Get CHEBMLNET activations of generated smile strings
        gen_act = self.get_predictions(
            self.gen_smiles, batch_size, num_workers
        )
        # Get CHEBMLNET activations of reference smile strings
        ref_act = self.get_predictions(
            self.ref_smiles, batch_size, num_workers
        )

        return gen_act, ref_act
    
    def get_dist_params(self, gen_act, ref_act):
        """
        Get mean and covariance of activations of generated and
        reference smile strings.

        Args:
        -----
        gen_act (np.ndarray):
            activations of generated smile strings
        ref_act (np.ndarray):
            activations of reference smile strings
        """
        # Calculate mean and covariance of gen activations
        gen_mu = np.mean(gen_act, axis=0)
        gen_sigma = np.cov(gen_act.T)

        # Calculate mean and covariance of ref activations
        ref_mu = np.mean(ref_act, axis=0)
        ref_sigma = np.cov(ref_act.T)

        return (gen_mu, gen_sigma), (ref_mu, ref_sigma)

            
        

    