
import numpy as np
import networkx as nx
import random
from dataclasses import dataclass
from networkx.algorithms import community
from typing import List, FrozenSet
from abc import abstractmethod, ABC
import pandas as pd
from Selection import MatingPool



"""
Fifth stage of the algorithm.
Crossover operaton will be applied on parent population to generate the new sequeces as the children populaiton.
"""

@dataclass
class Children: #The output object as children population in a dataframe foramt
    offspring_pop : pd.DataFrame

class Crossover(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def crossover(self, population : MatingPool) :
        pass


def domains_from_pae_matrix_networkx(pae_matrix:np.array, pae_power=1, pae_cutoff = 10, graph_resolution=0.5)->List[FrozenSet[int]]:
    
    """Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each 
    pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    Arguments:

        - pae_matrix: a (n_residues x n_residues) numpy array. Diagonal elements should be set to some non-zero
          value to avoid divide-by-zero warnings
        - pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
        - pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
        - graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
          lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.

    Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.
    """
    
    weights = 1/pae_matrix**pae_power
    g = nx.Graph()
    size = weights.shape[0]
    g.add_nodes_from(range(size))
    edges = np.argwhere(pae_matrix<pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    wedges = [(i,j,w) for (i,j), w in zip(edges, sel_weights)]
    g.add_weighted_edges_from(wedges)
    clusters = community.greedy_modularity_communities(g, weight = "weight", resolution=graph_resolution)
    return clusters



def swap_sequences(seq1: List[str], plddt1: List[List[float]], seq2: List[str], plddt2: List[List[float]]):
            
    """Randomely choosing domains and their corresponding plddt values in each sequence to be swapt with other sequence's.
    
    Arguments:
    - seq1: List of domains of the first chosen sequence (e.g. ['SLISA', 'VASLNG'])
    - plddt1: List of lists containing plddt values for seq1 (e.g. [[88.4260816334587, 22.439318656160356, ...], [73.7323068694926, 21.048009795821244, ...]])
    - seq2: List of domains of the second chosen sequence 
    - plddt2: List of lists containing plddt values for seq2
  
    Returns:
    - new_seq1, new_plddt1, new_seq2, new_plddt2: Swapped sequences and plddt values, the same as the input fomrats
    """
    
    index1 = random.randint(0, len(seq1) - 1)
    index2 = random.randint(0, len(seq2) - 1)
    tmp = seq1[index1]
    seq1[index1] = seq2[index2]
    seq2[index2] = tmp

    tmp = plddt1[index1]
    plddt1[index1] = plddt2[index2] #as the sequences' domains are swapt their corresponding PLDDT socers are swapt too.
    plddt2[index2] = tmp

    return seq1, plddt1, seq2, plddt2

class DomainCrossover(Crossover):
    def __init__(self):
        super().__init__()


    def crossover(self, population : MatingPool) :
        """The method implements crossover between sequences's domains
        Arguments:
        - population: population of sequeces with their corresponding structure scores in a dataframe format
        
        Returns:
        - Children population as a dataframe containing sequences and their corresponding plddt scores.
        """
 
        #New sequecnes and their corresponding plddt scores will be stored in the lists
        NewSeq = []
        NewPlddt = []
        # Iterate through dataframes' sequences in pairs
        for i in range(0, len(population.fittest_pop), 2):
            if i < len(population.fittest_pop) - 1:
                cluster1 = [list(frozenset) for frozenset in domains_from_pae_matrix_networkx(population.fittest_pop["pae"].iloc[i])]
                cluster2 = [list(frozenset) for frozenset in domains_from_pae_matrix_networkx(population.fittest_pop["pae"].iloc[i + 1])]
                domain_specified_seq1 = ["".join(list(population.fittest_pop["sequence"].iloc[i])[k] for k in subgroup) for subgroup in cluster1] #the sequence will be a list composed of its seperated domains
                domain_specified_seq2 = ["".join(list(population.fittest_pop["sequence"].iloc[i+1])[k] for k in subgroup) for subgroup in cluster2]
                plddt_seq1 = [[population.fittest_pop["plddt"].iloc[i][k]for k in subgroup] for subgroup in cluster1]#the domains' coresponding plddt scores will be separated in a list form to be swapt by swap_sequences function
                plddt_seq2 = [[population.fittest_pop["plddt"].iloc[i+1][k]for k in subgroup] for subgroup in cluster2]
                new_seq1, new_plddt1, new_seq2, new_plddt2 = swap_sequences(domain_specified_seq1.copy(), plddt_seq1.copy(), domain_specified_seq2.copy(), plddt_seq2.copy())

                #converting the list of domains into a string again
                new_seq1 = ''.join(val for sublist in new_seq1 for val in sublist)
                new_seq2 = ''.join(val for sublist in new_seq2 for val in sublist)
                new_plddt1 = [value for sublist in new_plddt1 for value in sublist]
                new_plddt2 = [value for sublist in new_plddt2 for value in sublist]

            else:
                #for a population with odd number os sequences, the last sequence's domains are swapt with the first sequence
                cluster1 = [list(frozenset) for frozenset in domains_from_pae_matrix_networkx(population.fittest_pop["pae"].iloc[i])]
                cluster2 = [list(frozenset) for frozenset in domains_from_pae_matrix_networkx(population.fittest_pop["pae"].iloc[0])]
                domain_specified_seq1 = ["".join(list(population.fittest_pop["sequence"].iloc[i])[k] for k in subgroup) for subgroup in cluster1]
                domain_specified_seq2 = ["".join(list(population.fittest_pop["sequence"].iloc[0])[k] for k in subgroup) for subgroup in cluster2]
                plddt_seq1 = [[population.fittest_pop["plddt"].iloc[i][k]for k in subgroup] for subgroup in cluster1]
                plddt_seq2 = [[population.fittest_pop["plddt"].iloc[0][k]for k in subgroup] for subgroup in cluster2]
                new_seq1, new_plddt1, new_seq2, new_plddt2 = swap_sequences(domain_specified_seq1.copy(), plddt_seq1.copy(), domain_specified_seq2.copy(), plddt_seq2.copy())


                new_seq1 = ''.join(val for sublist in new_seq1 for val in sublist)
                new_seq2 = ''.join(val for sublist in new_seq2 for val in sublist)
                new_plddt1 = [value for sublist in new_plddt1 for value in sublist]
                new_plddt2 = [value for sublist in new_plddt2 for value in sublist]

            NewSeq.append(new_seq1)
            NewSeq.append(new_seq2)
            NewPlddt.append(new_plddt1)
            NewPlddt.append(new_plddt2)

        #returning children and parent population as a new population of protein sequences!
        offspring = pd.DataFrame({"sequence" : NewSeq, "plddt" : NewPlddt})
        offspring_pop = pd.concat([offspring,  population.fittest_pop[["sequence", "plddt"]]], ignore_index=True)
        offspring_pop = offspring_pop.sample(frac = 1)
        offspring_pop = offspring_pop.reset_index(drop = True)
        return Children(offspring_pop = offspring_pop)
