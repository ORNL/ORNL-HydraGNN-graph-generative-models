# Import necessary libraries
import pandas as pd
import numpy as np
import os

from rdkit import Chem

# Define Class
class MolDataset():
    def __init__(self, pdb_path):
        # set folder path as attribute
        self.pdb_path = pdb_path
        
        # Create dictionary of rdkit molecule objects
        # and molecule dataframe
        self.create_dataframe()


    def get_pdb_files(self):
        """
        Returns all .pdb files in path given at initialization.
        """
        return [f for f in os.listdir(self.pdb_path) if f.endswith('.pdb')]
    
    def separate_pos(self, pos):
        """
        Separates concatenated positions by number and returns
        a list of separated positions. Each coordinate position will
        always have 3 decimal places, therefore we can separate
        them accordingly.

        Args:
            pos (str): Concatenated position string
        """

        # Separate concatenated positions by number
        sep_pos = list(pos)
        # Remove any spaces
        sep_pos = [p for p in sep_pos if p != ' ']
        # Get a list of decimal locations
        dec_loc = [i for i, dec in enumerate(sep_pos) if dec == '.']

        # Define list to store separated positions
        correct_pos = []
        # Define start index
        start = 0
        # Loop through decimal locations and separate positions
        for loc in dec_loc:
            # Define end of position
            end = loc+4
            # Avoid out of bounds error
            if loc == dec_loc[-1]:
                correct_pos.append(''.join(sep_pos[start:]))
            else:
                correct_pos.append(''.join(sep_pos[start:end]))
            start = end

        return correct_pos
    
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
                    atom_dict['atom_pos'].append(self.separate_pos(atom_info[5]))
            
            # Concatenate atom type and atom positions into separate arrays
            atom_dict['atom_type'] = np.array(atom_dict['atom_type'])
            atom_dict['atom_pos'] = np.vstack(atom_dict['atom_pos']).astype(np.float32)
        except:
            print(f'Error processing file: {file_path}')

        return atom_dict
    
    def create_dataframe(self):
        """
        Creates a pandas dataframe from the pdb files in the
        folder path given at initialization. The dataframe contains
        atom type, positions, and rdkit molecule objects for each 
        generated molecule.
        """

        # Get pdb files
        pdb_files = self.get_pdb_files()

        # Initialize dictionary to store rdkit molecules
        self.rdkit_mols = {}

        # Initialize dictionary for molecule dataframe
        self.mol_df = {}

        # Loop through pdb files and fill dictionaries
        for i, file in enumerate(pdb_files):
            # Set file path
            file_path = os.path.join(self.pdb_path, file)

            # Parse pdb file and extract atom type and positions
            atom_dict = self.parse_pdb(file_path)

            # Create rdkit molecule object
            atom_dict['rdkit_mol'] = Chem.MolFromPDBFile(
                file_path,
                sanitize=False,
                removeHs=False,
                proximityBonding=True
            )

            # TODO: Add specified features to molecule dictionary

            # Add molecule dictionary to mol_df
            self.mol_df[i] = atom_dict

        # Create dataframe from dictionary
        self.mol_df = pd.DataFrame.from_dict(self.mol_df).T


            

        
