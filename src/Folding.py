import esm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from biotite.structure import AtomArray
from typing import List, Union
import numpy as np
import torch
from jax.tree_util import tree_map
from openfold.np.residue_constants import atom_order
import pandas as pd
from io import StringIO
from biotite.structure.io.pdb import PDBFile



"""
Second stage of the algorithm.
The randomely generated population from Individual module will be given plddt, pae, and ptm scores thorugh ESM language model. 
The folding score results will be in pandas dataframe format.
"""
@dataclass
class FoldResult:
    ptm_df : pd.DataFrame
    mean_plddt_df : pd.DataFrame
    plddt_df : pd.DataFrame
    pae_df : pd.DataFrame
    atoms_df : pd.DataFrame

class Predictor(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def load(self, dir_path : str) -> None:
        pass
    @abstractmethod
    def fold(self, Population : List[str] ) -> FoldResult:
        pass


def pdb_file_to_atomarray(pdb_path = Union[str,StringIO]) -> AtomArray:
    return PDBFile.read(pdb_path).get_structure(model = 1)


class ESMFold(Predictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def load(self, dir_path : str) -> None:
        """Loading ESMFold model
        Arguments:
            - dir_path : A path to load the model in 
        Returns: The loaded model will be put on a gpu and will be in an evaluation mode
        """
        torch.hub.set_dir(dir_path)
        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval().cuda() #switch the model to an evaluation mode and move it to an available GPU
        return self.model

    def fold(self, Population : List[str]) -> FoldResult:
        """Using ESMFold model to predict 3d atomic coorrdinates of protein sequences with their uncertainty scores
        Arguments:
            - Population : List of protein sequences for structure prediction by ESMFold
        Returns: A dataframe of protein sequences with their atomic strructuer and their scores
        """
        
        assert self.model is not None
        
        ptm_ =[]
        plddt_ =[]
        pae_ = []
        mean_plddt_ = []
        atoms_ = []
        for sequence in Population:
            with torch.no_grad():
                output = self.model.infer(sequence, chain_linker="X"*0, residue_index_offset=0) #prerdict 3D sructure of sequences using ESMfold model.
                output = tree_map(lambda x: x.to("cpu"), output) 
                pdb_string = esm.esmfold.v1.misc.output_to_pdb(output)[0]
                atoms : AtomArray = pdb_file_to_atomarray(StringIO(pdb_string))
                plddt = output["plddt"] #second output of the ESMfold model, plddt score demonstrates the model's per residue confidence.
                plddt = plddt[0, ...].numpy()
                plddt = plddt.transpose()
                plddt_.append(plddt[atom_order["CA"], :])
                mean_plddt_.append((output["mean_plddt"]).numpy()) #obtain a mean plddt giving model's confidence on the whole sequence.
                ptm_.append(float(output["ptm"])) #third output, ptm score represents the model's confidence in the overall predicted structure.
                pae_.append(output["predicted_aligned_error"].numpy()) #fourth output, Pae score is a measure of how confident the model is in relative position of two residues within the predicted structure.
                atoms_.append(atoms)
        #Store the obtained values from ESM model in a dataframe 
        atoms_df = pd.DataFrame({"sequence" : Population, "atoms": atoms_})
        plddt_df = pd.DataFrame({"sequence" : Population, "plddt" : plddt_})
        ptm_df = pd.DataFrame({"sequence" : Population, "ptm" : ptm_})
        pae_df = pd.DataFrame({"sequence" : Population, "pae" : pae_})
        mean_plddt_df = pd.DataFrame({"sequence" : Population, "mean_plddt" : mean_plddt_})
        return FoldResult(atoms_df = atoms_df, ptm_df = ptm_df, plddt_df = plddt_df, mean_plddt_df = mean_plddt_df, pae_df = pae_df)



class ESMmodel(Predictor):
    
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.alphabet = None

    def load(self, dir_path : str, ) -> None:
        """Loading ESM2 model, the model will predict masked amino acids in the Mutation module
        Arguments: 
            - dir_path : A directory path to load the model in
        Returns : The model and its tokenization module, named as alphabet, will be put on GPU for inference tasks
        """
        torch.hub.set_dir(dir_path)
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D() #Load the model and alphabet module. alphabt module will be used to tokenize squence inputs
        self.model.eval() #Set the model to evaluation mode
        return self.model, self.alphabet

    def fold(self, sequence : str) -> FoldResult:
        pass
