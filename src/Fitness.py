from abc import abstractmethod, ABC
from dataclasses import dataclass
import pandas as pd
import numpy as np
from biotite.structure import AtomArray
from sklearn.preprocessing import MinMaxScaler

@dataclass
class StructureScore:
    FoldingScore_df : pd.DataFrame

@dataclass
class GlobularityScore:
    Globularity_df : pd.DataFrame

@dataclass
class FilamentDimer:
    translational_sym_df : pd.DataFrame

@dataclass
class OptimizationFitness:
    OptimizationFitness_df : pd.DataFrame

    
class Fitness(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def Score(self, foldingresult : FoldResult) -> StructureScore:
        pass

    @abstractmethod
    def FitnessScore(self, *scores) -> OptimizationFitness:
        pass


class FoldingFitness(Fitness):
    def __init__(self):
        super().__init__()
        

    def Score(self, foldingresult: FoldResult) -> StructureScore:
        """
        Storing folding result scores from ESM in dataframe format and define a general fitness score as a linear sum of mean plddt, mean pae, and ptm scores.
        """   
        assert FoldResult is not None
        FoldingScore_df = pd.DataFrame()
        FoldingScore_df["sequence"] = foldingresult.ptm_df["sequence"]   
        FoldingScore_df["plddt"] = foldingresult.plddt_df["plddt"]
        FoldingScore_df["ptm"] = foldingresult.ptm_df["ptm"]*100
        FoldingScore_df["pae"] = foldingresult.pae_df["pae"]
        FoldingScore_df["mean_plddt"] = foldingresult.mean_plddt_df['mean_plddt']
        FoldingScore_df["GeneralScore"] = foldingresult.mean_plddt_df["mean_plddt"] + foldingresult.ptm_df["ptm"]*100 + -1*foldingresult.pae_df.apply(lambda row: np.mean(row["pae"]), axis=1) 
       
        return StructureScore(FoldingScore_df)
    

    def FitnessScore(self) -> OptimizationFitness:
        pass

  
def _is_Nx3(array : np.ndarray) -> bool:
    return len(array.shape) == 2 and array.shape[1] == 3
    
def get_center_of_mass(coordinates : np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates)

    return coordinates.mean(axis = 0).reshape(1, 3)

def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    return atoms[(atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")]
        
def distance_to_centroid(coordinates : np.ndarray) -> np.ndarray:
    """
    Computing the distances from each of the coordinates to the
    centroid of all coordinates.
    """
    assert _is_Nx3(coordinates)
    center_of_mass = get_center_of_mass(coordinates)
    dist = coordinates - center_of_mass
    return np.linalg.norm(dist, axis=-1)  



class MaximizedGlobularity(Fitness):
    """
Calculating the std of the backbone distances from the center of mass,
lower std indicates a more compact and globular structure.
    """

    def __init__(self) -> None:
        super().__init__()
        

    def Score(self, foldingresult : FoldResult) -> GlobularityScore:

               
        Globularity_df = pd.DataFrame
        distance = []
        for i in range(len(foldingresult.atoms_df)):

          if ":" in foldingresult.atoms_df["sequence"][0]:


             monomer_identifiers = foldingresult.atoms_df["sequence"].iloc[i].split(":")
             start1, end1 = (1, len(monomer_identifiers[0]))
             backbone1 = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id  >= start1, foldingresult.atoms_df["atoms"].iloc[i].res_id < end1)]).coord

             start2, end2 = (len(monomer_identifiers[0]) +1, len(monomer_identifiers[0])*2)
             backbone2 = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id  >= start2, foldingresult.atoms_df["atoms"].iloc[i].res_id < end2)]).coord


             distance.append(np.mean([float(np.std(distance_to_centroid(backbone1))), float(np.std(distance_to_centroid(backbone2)))]))


          else:

            start,end = (1, len(foldingresult.atoms_df["sequence"].iloc[i]))
            backbone = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id >= start, foldingresult.atoms_df["atoms"].iloc[i].res_id < end)]).coord
            
            distance.append(float(np.std(distance_to_centroid(backbone))))
        
        Globularity_df = pd.DataFrame({"sequence" : foldingresult.ptm_df["sequence"], "distance_to_centroid" : distance})
        
        return GlobularityScore(Globularity_df)

    
    def FitnessScore(self) -> OptimizationFitness:
        pass


class Filament(Fitness):

  def __init__(self) -> None:
    super().__init__()

  #translational movement of monomers
  #being centered in the origin 


  def Score(self, foldingresult : FoldResult) -> FilamentDimer:

    translational_sym = []
    for i in range(len(foldingresult.atoms_df)):
    #translational symmetry
    #backbones distance:

      monomer_identifiers = foldingresult.atoms_df["sequence"].iloc[i].split(":")
      start1, end1 = (1, len(monomer_identifiers[0]))
      backbone1 = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id  >= start1, foldingresult.atoms_df["atoms"].iloc[i].res_id < end1)]).coord

      start2, end2 = (len(monomer_identifiers[0]) + 1, len(monomer_identifiers[1])*2)
      backbone2 = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id  >= start2, foldingresult.atoms_df["atoms"].iloc[i].res_id < end2)]).coord

      backbone_diff = (np.abs(backbone2 - backbone1))

      #com distance:
      center_of_mass1 = get_center_of_mass(backbone1)
      center_of_mass2 = get_center_of_mass(backbone2)
      com_diff = np.abs(center_of_mass2 - center_of_mass1)

      translational = float(np.std(com_diff - backbone_diff))
      translational_sym.append(translational)

    translational_sym_df = pd.DataFrame({"Sequence": foldingresult.atoms_df["sequence"], "TranslationalSummetry" : translational_sym })

    return FilamentDimer(translational_sym_df = translational_sym_df)

    
  def FitnessScore(self) -> OptimizationFitness:
    pass


class TotalFitness(Fitness):

    def __init__(self) -> None:
        super().__init__()
        self.StructureScore = None
        self.GlobularityScore = None
        self.FilamentDimer = None 

        
    def Score(self, foldingresult : FoldResult) -> StructureScore:
        pass


    def FitnessScore(self, fold_score = StructureScore, globularity_score = GlobularityScore, filament_score = FilamentDimer ) -> OptimizationFitness:
        #currently coefficient weights of 1 and 2 for folding and globularity are defined.
        
        OptimizationFitness_df = pd.DataFrame()
        OptimizationFitness_df["sequence"] = fold_score.FoldingScore_df["sequence"]
        OptimizationFitness_df["plddt"] = fold_score.FoldingScore_df["plddt"]
        OptimizationFitness_df["ptm"] = fold_score.FoldingScore_df["ptm"]
        OptimizationFitness_df["pae"] = fold_score.FoldingScore_df["pae"]
        OptimizationFitness_df["mean_plddt"] = fold_score.FoldingScore_df['mean_plddt']

        #In the case of filaments keeping the first monomer for the rest of optimization!
        if ":" in OptimizationFitness_df["sequence"][0]:
        
           def downsample_data(row):
             seq_length = len(row["sequence"])
             row['plddt'] = row['plddt'][:seq_length]
             row["pae"] = row["pae"][0][:seq_length, :seq_length]
             return row


           OptimizationFitness_df["sequence"] = OptimizationFitness_df["sequence"].str.split(':').str.get(0)[0]
   
          
           df_downsampled = OptimizationFitness_df.apply(downsample_data, axis=1)
           OptimizationFitness_df["pae"] = df_downsampled["pae"]
           OptimizationFitness_df["plddt"] = df_downsampled["plddt"]

           OptimizationFitness_df["FitnessScore"] = ((fold_score.FoldingScore_df["GeneralScore"])/100) + (2 * 1/((filament_score.translational_sym_df["TranslationalSummetry"]) + ((np.e) ** -6)))+( 2 * 1/((globularity_score.Globularity_df["distance_to_centroid"]) + ((np.e) ** -6)))
        
        else:
           OptimizationFitness_df["FitnessScore"] = ((fold_score.FoldingScore_df["GeneralScore"])/100) + 2 * 1/((globularity_score.Globularity_df["distance_to_centroid"]) + ((np.e) ** -6))


        return OptimizationFitness(OptimizationFitness_df = OptimizationFitness_df)    
