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


class ESMFold(Predictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def load(self, dir_path : str) -> None:
        """
        loading ESMFold model
        """
        torch.hub.set_dir(dir_path)
        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval().cuda()
        return self.model

    def fold(self, Population : List[str]) -> FoldResult:
        assert self.model is not None

        def pdb_file_to_atomarray(pdb_path = Union[str,StringIO]) -> AtomArray:
            return PDBFile.read(pdb_path).get_structure(model = 1)


        ptm_ =[]
        plddt_ =[]
        pae_ = []
        mean_plddt_ = []
        atoms_ = []
        for sequence in Population:
            def pdb_file_to_atomarray(pdb_path : Union[str, StringIO]) -> AtomArray:
                return PDBFile.read(pdb_path).get_structure(model = 1)
            with torch.no_grad():
                output = self.model.infer(sequence, chain_linker="X"*0, residue_index_offset=0)

                output = tree_map(lambda x: x.to("cpu"), output)
                pdb_string = esm.esmfold.v1.misc.output_to_pdb(output)[0]
                atoms : AtomArray = pdb_file_to_atomarray(StringIO(pdb_string))

                plddt = output["plddt"]
                plddt = plddt[0, ...].numpy()
                plddt = plddt.transpose()
                plddt_.append(plddt[atom_order["CA"], :])
                mean_plddt_.append((output["mean_plddt"]).numpy())
                ptm_.append(float(output["ptm"]))
                pae_.append(output["predicted_aligned_error"].numpy())
                atoms_.append(atoms)
        #turn the folding results into a dataframe 
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
        """
        loading ESM2 model, used in masked amino acid prediction
        """

        torch.hub.set_dir(dir_path)
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model.eval()
        return self.model, self.alphabet

    def fold(self, sequence : str) -> FoldResult:
        pass
