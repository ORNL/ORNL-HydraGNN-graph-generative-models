# Import necessary libraries
import pandas as pd
import numpy as np
import os, json
import pickle as pkl

from rdkit import Chem
from analysis.frechet_distance import FCD

# Define Class
class PDBDataset():
    """
    Class to create a dataset from pdb files containing
    generated molecules. The class calculates various metrics
    to evaluate the quality of the generated molecules.

    NOTE: This class assumes that the pdb files are generated
    from the rdkit Chem.MolToPDBFile function. If the PDB files
    are generated from a different source, check the parse_pdb
    function to ensure that the atom type and position are parsed
    correctly.

    Args:
    -----
    pdb_path (str or list):
        Path to folder containing pdb files. If multiple paths
        are given, the class will concatenate the dataframes
        of each path into a single dataframe.
    
    Attributes:
    -----------
    pdb_path (str or list):
        Path to folder containing pdb files
    data (pd.DataFrame):
        Dataframe containing atom type, positions, SMILES
        strings, boolean valid value, and rdkit molecule 
        objects for each generated molecule.
    qm9_smiles (list):
        List of smiles strings from the QM9 dataset.
    metrics (dict):
        Dictionary containing the following metrics:
        - validity: Percent of valid molecules in dataset
        - uniqueness: Percent of unique molecules in dataset
        - novelty: Percent of novel molecules in dataset
        - gen_atom_dist: Distribution of generated atoms
        - fcd_score: Frechet ChemNet Distance score
    """
    def __init__(self, pdb_path):
        # set folder path as attribute
        self.pdb_path = pdb_path
        # Define variable to store number of errors when extracting PDB files
        self.pdb_errors = 0
        
        # Create molecule dataframe
        # Check if multiple paths are given
        if isinstance(pdb_path, list) and len(pdb_path) > 1:
            # Create list to store dataframes
            df_list = []
            # Loop through paths and create dataframes
            for path in pdb_path:
                df_list.append(self.create_dataframe(path))
            
            # Concatenate dataframes
            self.data = pd.concat(df_list, axis=0, ignore_index=True)
        else:
            # Create single dataframe
            # Check if path is still given as list
            if isinstance(pdb_path, list):
                pdb_path = pdb_path[0]
            self.data = self.create_dataframe(pdb_path)

        # Print number of errors
        print(f'Number of PDB errors: {self.pdb_errors}')
        # Store QM9 smile strings
        self.qm9_smiles = self.get_qm9_smiles()

        # Get metrics of dataset
        print('\nCalculating RDkit metrics...')
        self.metrics = {
            'validity': self.calc_validity(),
            'uniqueness': self.calc_uniqueness(),
            'novelty': self.calc_novelty(),
            'gen_atom_dist': self.calc_gen_atom_dist()
        }

        # Calculate FCD score
        print('\nCalculating FCD score...')
        self.metrics['fcd_score'] = self.calc_fcd_score()

        # Print summary of dataset and store in run folder
        self.get_dataset_summary()

    def get_dataset_summary(self):
        """
        Print a summary of the dataset metrics
        """
        # Print summary of metrics
        print(f'\nDataset Metrics:')
        print('--------------------')
        print(f'Percent Validity: {100*self.metrics["validity"]}%')
        print(f'Percent Uniqueness: {100*self.metrics["uniqueness"]}%')
        print(f'Percent Novelty: {100*self.metrics["novelty"]}%')
        print(f'FCD score: {self.metrics["fcd_score"]}')

        # Print atom distribution
        print(f'\nAtom Distribution of Dataset Generation:')
        print('---------------------------------------------')
        for atom, dist in self.metrics['gen_atom_dist'].items():
            print(f'{atom}: {100*dist:.2f}%')

        # Save summary to json file
        if isinstance(self.pdb_path, list):
            # Store path to multi-run results
            save_path = os.path.join(self.pdb_path[0], '..', '..', 'multi_run_results.json')
            # Create key for new multi-run result by concatenating run names
            run_key = '_'.join([path.split('/')[-2] for path in self.pdb_path])
            # If the json file exists, load and update. If not, create new file
            try:
                with open(save_path, 'r') as f:
                    run_dict = json.load(f)
                
                # Update dictionary
                run_dict[run_key] = self.metrics
            except:
                run_dict = {}
                run_dict[run_key] = self.metrics

            # Save run dictionary
            with open(save_path, 'w') as f:
                json.dump(run_dict, f, indent=4)
        else:
            save_path = os.path.join(self.pdb_path, '..', 'gen_metrics.json')
            # Create json file and save metrics
            with open(save_path, 'w') as f:
                # Save metrics
                json.dump(self.metrics, f, indent=4)

    
    def get_qm9_smiles(self):
        """
        Get rdkit molecules from QM9 dataset
        """
        # Path to smiles
        smiles_path = os.path.join(
            os.path.dirname(__file__), '..', 'dataset', 'smiles_qm9.pkl'
        )

        # Check if qm9 smiles file exists
        if os.path.exists(smiles_path):
            # return smiles list
            with open(smiles_path, 'rb') as f:
                return pkl.load(f)
        else: 
            # Define path to QM9 data
            qm9_path = os.path.join(
                os.path.dirname(__file__), '..', 'dataset', 'raw', 'gdb9.sdf')
            # Load QM9 dataset
            qm9_data = Chem.SDMolSupplier(qm9_path)

            # Get smiles strings of all valid molecules
            smile_strings = [Chem.MolToSmiles(mol) for mol in qm9_data if mol is not None]

            # Store smiles strings in pickle file
            print('\n Saving to pickle file for faster reload...')
            with open(smiles_path, 'wb') as f:
                pkl.dump(smile_strings, f)

            return smile_strings
        
                            
    def get_pdb_files(self, path):
        """
        Returns all .pdb files in path given at initialization.

        Args:
        -----
        path (str):
            Path to folder containing pdb files
        """
        return [f for f in os.listdir(path) if f.endswith('.pdb')]
    
    def parse_pdb(self, file_path):
        """
        Parses a pdb file and returns a dictionary containing
        atom types and positions.

        Args:
            file_path (str or os.path): Path to pdb file
        """

        # Read content of pdb file and store in variable
        with open(file_path, 'r') as f:
            pdb_content = f.readlines()

        # Initialize dictionary to store relevant information
        atom_dict = {}
        # Create empty list keys for atom type and atom positions
        atom_dict['atom_type'] = []
        atom_dict['atom_pos'] = []

        try:
            for atom in pdb_content:
                # Split the line by whitespace
                atom_info = atom.split()
                # Check if the line is an atom line
                if atom_info[0] == 'HETATM':
                    # Store atom information in dictionary
                    atom_dict['atom_type'].append(atom_info[-1])
                    atom_dict['atom_pos'].append(
                        [atom_info[5], atom_info[6], atom_info[7]]
                    )

            # Concatenate atom type and atom positions into separate arrays
            atom_dict['atom_type'] = np.array(atom_dict['atom_type'])
            atom_dict['atom_pos'] = np.vstack(atom_dict['atom_pos']).astype(np.float32)
        except:
            print(f'Error processing file: {file_path}')
            self.pdb_errors += 1

        return atom_dict
    
    def create_dataframe(self, path):
        """
        Creates a pandas dataframe from the pdb files in the
        folder path given at initialization. The dataframe contains
        atom type, positions, and rdkit molecule objects for each 
        generated molecule.
        """

        # Get pdb files
        pdb_files = self.get_pdb_files(path)

        # Initialize dictionary for molecule dataframe
        self.mol_dict = {}

        # Loop through pdb files and fill dictionaries
        for i, file in enumerate(pdb_files):
            # Set file path
            file_path = os.path.join(path, file)

            # Parse pdb file and extract atom type and positions
            atom_dict = self.parse_pdb(file_path)

            # Create rdkit molecule object
            atom_dict['rdkit_mol'] = Chem.MolFromPDBFile(
                file_path,
                sanitize=False,
                removeHs=False,
                proximityBonding=True
            )

            # Add H atoms to molecule
            atom_dict['rdkit_mol'] = Chem.AddHs(atom_dict['rdkit_mol'])

            # Add SMILES string of molecule to dictionary
            try:
                atom_dict['smiles'] = Chem.MolToSmiles(
                    atom_dict['rdkit_mol'], canonical=True)
            except:
                atom_dict['smiles'] = None

            # TODO: Add specified features to molecule dictionary
            atom_dict['valid'] = self.is_valid_mol(atom_dict['rdkit_mol'])

            # Add molecule dictionary to mol_df
            self.mol_dict[i] = atom_dict

        # Create dataframe from dictionary
        return pd.DataFrame.from_dict(self.mol_dict).T
    
    def is_valid_mol(self, mol):
        """
        Check if molecule is valid by checking if it violates
        valency properties. Returns True if valid, False otherwise.

        Args:
        -----
        mol (rdkit.Chem.rdchem.Mol):
            RDKit molecule object
        """
        try:
            Chem.SanitizeMol(mol)
            return True
        except:
            return False
        
    def calc_validity(self):
        """
        Calculate the percent of valid molecules in the dataset.

        NOTE: In DiGress, Vignac et al. also use a relaxed validity
        matric that takes into account molecule polarity to make a more
        fair calculation (May be useful to implement later).
        """
        # Return percent of valid molecules of total molecules generated
        return self.data['valid'].sum() / len(self.data['valid'])
    
    def calc_uniqueness(self, return_unique=False):
        """
        Calculate the uniqueness of molecules in the dataset.
        """
        # Get all valid molecules in dataset
        valid_smiles = list(
            self.data.loc[self.data.valid == True].smiles.values
        )

        # return percent of unique generated molecules
        # and the unique molecules if specified
        if return_unique:
            return len(set(valid_smiles))/len(valid_smiles), list(set(valid_smiles))
        else:
            return len(set(valid_smiles))/len(valid_smiles)
        
    def calc_novelty(self):
        """
        Calculate the novelty of the generated molecules
        """

        # Start count of novel molecules
        num_novel = 0
        # Get unique molecules
        _, unique = self.calc_uniqueness(return_unique=True)
        # Loop through unique molecules and check if they are in QM9 dataset
        for smiles in unique:
            if smiles not in self.qm9_smiles:
                num_novel += 1
        
        # Return percent of novel molecules of unique generated molecules
        return num_novel / len(unique)
    
    def calc_gen_atom_dist(self):
        """
        Calculate the distribution of generated atoms in the dataset
        """
        # Get atom types
        atom_types = self.data.atom_type.values
        # combine into single list
        atom_types = np.concatenate(atom_types)
        # Count each atom type
        types, atom_count = np.unique(atom_types, return_counts=True)

        # Normalize the distribution
        dist_values = atom_count / atom_count.sum()

        atom_dist = {}
        for i in range(len(types)):
            atom_dist[types[i]] = dist_values[i]

        return atom_dist
    
    def calc_fcd_score(self):
        """
        Calculate the Frechet ChemNet Distance (FCD) of the generated
        molecules compared to the QM9 dataset.
        """
        # Get unique molecules
        _, unique = self.calc_uniqueness(return_unique=True)
        # Get 10,000 random samples from molecules if greater than 10,000
        if len(unique) > 10000:
            unique = list(
                np.random.choice(unique, 10000, replace=False)
            )
        # Get 10,000 random QM9 molecules
        qm9_mols = list(
            np.random.choice(self.qm9_smiles, 10000, replace=False)
        )
        # Calculate FCD score
        fcd_obj = FCD(
            gen_smiles=unique,
            ref_smiles=qm9_mols
        )

        return fcd_obj.fcd_score


            

        