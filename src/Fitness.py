from abc import abstractmethod, ABC
from dataclasses import dataclass
import pandas as pd
import numpy as np
from biotite.structure import AtomArray
from sklearn.preprocessing import MinMaxScaler
from Folding import FoldResult



"""
Third stage of the algorithm.
Various fitness functions such as globularity, folding and filament have been defiend using the obtained structural scores from Folding module.
The defined fitness scores drive the population towards desired structural properties over successive generations.

Each individual sequence in the population is evaluated using these fitness criteria.
A total fitness score is calculated for each sequence, which determines its survival
probability in the subsequent evolution process.
"""
        
   
def score_folding(foldingresult: FoldResult) -> dict:
    """Storing folding result scores from ESM in dataframe format and define a general fitness score as a linear weighted sum of mean plddt, mean pae, and ptm scores.
    Arguments: 
        - foldingresult : The strucural scores obtained by ESMFold by Folding module
    Returns: A dataframe of protein sequences with their fitness scores
    """ 
    assert FoldResult is not None
    FoldingScore_df = pd.DataFrame()
    FoldingScore_df["sequence"] = foldingresult.ptm_df["sequence"]   
    FoldingScore_df["plddt"] = foldingresult.plddt_df["plddt"]
    FoldingScore_df["ptm"] = foldingresult.ptm_df["ptm"] * 100
    FoldingScore_df["pae"] = foldingresult.pae_df["pae"]
    FoldingScore_df["mean_plddt"] = foldingresult.mean_plddt_df['mean_plddt']
    FoldingScore_df["GeneralScore"] = (
        foldingresult.mean_plddt_df["mean_plddt"]
        + foldingresult.ptm_df["ptm"] * 100
        + -1 * foldingresult.pae_df.apply(lambda row: np.mean(row["pae"]), axis=1)
    )
    return FoldingScore_df
    

  
def _is_Nx3(array : np.ndarray) -> bool:
    """Checks if the input NumPy array is a 2D array with exactly 3 columns.
    Arguments: 
        -array : A NumPy array to be checked.
    Returns : bool; True if the input array is a 2D array with 3 columns, otherwise False.
    """
    return len(array.shape) == 2 and array.shape[1] == 3

    
def get_center_of_mass(coordinates : np.ndarray) -> np.ndarray:
    """Calculate the center of mass of coordinates in each dimension.
    Arguments:
        - coordinates: Atomic 3d coordinates from PDB data.
    Returns: Returns a 3D vector representing the center of mass of the input coordinates, calculated as the mean value of each coordinate dimension (x, y, z).
    """
    assert _is_Nx3(coordinates)
    return coordinates.mean(axis = 0).reshape(1, 3)

def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    """Extracts backbone atoms (CA, N, and C) from an AtomArray.
    Arguments:
        - atoms (AtomArray): An array containing atoms of a protein structure.
    Returns: A filtered array containing only the backbone atoms
        (CA - alpha carbon, N - nitrogen, and C - carbonyl carbon) from the input.
    """
    return atoms[(atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")]
        
def distance_to_centroid(coordinates : np.ndarray) -> np.ndarray:
    """Computing the euclidean distances of the backbone coordinates to the centroid of all coordinates.
    Argument:
        - coordinates: Backbone coordinates, distance betweem CA-CA atoms of residues.
    Returns: A 1D array of Euclidean distances of each point in 'coordinates' to the centroid. Each value represents
            the distance of a residue from the center of mass, providing a measure of its spatial spread relative to the centroid.
    """
    assert _is_Nx3(coordinates)
    center_of_mass = get_center_of_mass(coordinates)
    dist = coordinates - center_of_mass
    return np.linalg.norm(dist, axis=-1) #calculate the distance along each row of the coordinates  


def maximizedglobularity_score(foldingresult : FoldResult) -> dict:
        """Calculating the globularity of proteins based on their spread value from their cwnter of mass.
        Arguments:
            - foldingresult: The atomic coordinates prerdicted by ESMFold model
        Returns: A dataframe consisting of protein sequences and their corresponding compactness values.
        """
        Globularity_df = pd.DataFrame
        distance = []
        for i in range(len(foldingresult.atoms_df)):

          if ":" in foldingresult.atoms_df["sequence"][0]:


             monomer_identifiers = foldingresult.atoms_df["sequence"].iloc[i].split(":")
             start1, end1 = (1, len(monomer_identifiers[0]))
             backbone1 = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id  >= start1, foldingresult.atoms_df["atoms"].iloc[i].res_id < end1)]).coord

             start2, end2 = (len(monomer_identifiers[0]) +1, len(monomer_identifiers[0])*2)
             backbone2 = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id  >= start2, foldingresult.atoms_df["atoms"].iloc[i].res_id < end2)]).coord

             # Calculate the mean distance of backbone coordinates of each residue from the center of mass, providing a single value that represents the compactness of the protein structure.
             distance.append(np.mean([float(np.std(distance_to_centroid(backbone1))), float(np.std(distance_to_centroid(backbone2)))]))


          else:

            start,end = (1, len(foldingresult.atoms_df["sequence"].iloc[i]))
            backbone = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id >= start, foldingresult.atoms_df["atoms"].iloc[i].res_id < end)]).coord
            
            distance.append(float(np.std(distance_to_centroid(backbone))))
        
        Globularity_df = pd.DataFrame({"sequence" : foldingresult.ptm_df["sequence"], "distance_to_centroid" : distance})
        
        return Globularity_df

    
 
def filament_score(foldingresult : FoldResult) -> dict:
    """The multimers should have translational symmetry for each chain. The distance between consecutive chains' center of mass must be equal to their residue backbone distances.
    Arguments: 
        - foldingresult: Atomic coordinates predicted by ESMFold
    Returns: A dataframe consisting of proten sequences and their translational symmetry value represented as the standard deviation of the difference between center-of-mass distance and backbone distance between monomers, providing a scalar measure of structural alignment.
    """
    translational_sym = []
    for i in range(len(foldingresult.atoms_df)):
    #Calculating translational symmetry as the standard deviatiton of the difference between center-of-mass distance and backbone distance between monomers.
      monomer_identifiers = foldingresult.atoms_df["sequence"].iloc[i].split(":")
      start1, end1 = (1, len(monomer_identifiers[0]))
      backbone1 = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id  >= start1, foldingresult.atoms_df["atoms"].iloc[i].res_id < end1)]).coord

      start2, end2 = (len(monomer_identifiers[0]) + 1, len(monomer_identifiers[1])*2)
      backbone2 = get_backbone_atoms(foldingresult.atoms_df["atoms"].iloc[i][np.logical_and(foldingresult.atoms_df["atoms"].iloc[i].res_id  >= start2, foldingresult.atoms_df["atoms"].iloc[i].res_id < end2)]).coord

      backbone_diff = (np.abs(backbone2 - backbone1))

      #center of mass distance:
      center_of_mass1 = get_center_of_mass(backbone1)
      center_of_mass2 = get_center_of_mass(backbone2)
      com_diff = np.abs(center_of_mass2 - center_of_mass1)

      translational = float(np.std(com_diff - backbone_diff))
      translational_sym.append(translational)

    translational_sym_df = pd.DataFrame({"Sequence": foldingresult.atoms_df["sequence"], "TranslationalSummetry" : translational_sym })

    return translational_sym_df



def downsample_data(row):
    """Adjusts the lengths of the plddt and pae fields in a row of data based on the length of the sequence provided in the row
    Arguments: 
        - row: A dictionary-like object containing sequeence, plddt, and pae keys.
    Returns: The updated row with truncated "plddt" and "pae" fields.
    """
    seq_length = len(row["sequence"])
    row['plddt'] = row['plddt'][:seq_length]
    row["pae"] = row["pae"][0][:seq_length, :seq_length]
    return row


def total_fitness_score(fold_score , globularity_score , filament_score ) -> dict:
        """Computes an overall fitness score for a protein sequence based on structural, globularity, and filament alignment properties.
        The final FitnessScore is calculated as a linear sum of weighted scores, where the 
        folding score, globularity score, and filament symmetry score are assigned specific weights 
        and combined as follows:
        - The folding score is weighted by 1
        - The globularity  and filament scores are weighted by 2 and are inversely considered in fitness score funstion, as these scores are std of the distances and their lower values will have bigger inverse values, therefore, more impact on the fitness score.
        Returns: A datatframe of protein sequences with their corrsponding fitness scores.
        """
        OptimizationFitness_df = pd.DataFrame()
        OptimizationFitness_df["sequence"] = fold_score.FoldingScore_df["sequence"]
        OptimizationFitness_df["plddt"] = fold_score.FoldingScore_df["plddt"]
        OptimizationFitness_df["ptm"] = fold_score.FoldingScore_df["ptm"]
        OptimizationFitness_df["pae"] = fold_score.FoldingScore_df["pae"]
        OptimizationFitness_df["mean_plddt"] = fold_score.FoldingScore_df['mean_plddt']

        #In the case of filaments as the chains are similar, for simplicity, all chains will be treated equally and will get equal fitness scores
        if ":" in OptimizationFitness_df["sequence"][0]:

           OptimizationFitness_df["sequence"] = OptimizationFitness_df["sequence"].str.split(':').str.get(0)[0]

           df_downsampled = OptimizationFitness_df.apply(downsample_data, axis=1)
           OptimizationFitness_df["pae"] = df_downsampled["pae"]
           OptimizationFitness_df["plddt"] = df_downsampled["plddt"]
           OptimizationFitness_df["FitnessScore"] = ((fold_score.FoldingScore_df["GeneralScore"])/100) + (2 * 1/((filament_score.translational_sym_df["TranslationalSummetry"]) + ((np.e) ** -6)))+( 2 * 1/((globularity_score.Globularity_df["distance_to_centroid"]) + ((np.e) ** -6)))
        
        else:
           OptimizationFitness_df["FitnessScore"] = ((fold_score.FoldingScore_df["GeneralScore"])/100) + 2 * 1/((globularity_score.Globularity_df["distance_to_centroid"]) + ((np.e) ** -6))


        return OptimizationFitness_df
