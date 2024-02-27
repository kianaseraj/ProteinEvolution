import numpy as np
import networkx as nx
from dataclasses import dataclass
from networkx.algorithms import community
from typing import List, Union, Dict
from abc import abstractmethod, ABC
import pandas as pd



@dataclass
class Children:
    offspring_pop : pd.DataFrame
    domain : Dict

class Crossover(ABC):
    def __init__(self):   
        pass

    @abstractmethod
    def crossover(self, population : MatingPool) :
        pass
       
class DomainCrossover(Crossover):
    def __init__(self):
        super().__init__()


    def crossover(self, population : MatingPool) :

        def domains_from_pae_matrix_networkx(pae_matrix, pae_power=1, pae_cutoff = 10, graph_resolution=0.5):
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
             
        def swap_sequences(seq1, plddt1, seq2, plddt2):


            max_seq_length = len(max(population.fittest_pop['sequence'], key=len))
    
            valid_indices1 = [i for i, domain in enumerate(seq1) if len(domain) > 200/10 ]
            valid_indices2 = [i for i, domain in enumerate(seq2) if len(domain) > 200/10 ]

            # Randomly choose a domain to swap for seq1
            if valid_indices1:
              index1 = np.random.choice(valid_indices1, size=1)[0]
            else:
              index1 = np.random.choice(range(len(seq1)))

            # Randomly choose a domain to swap for seq2
            if valid_indices2:
              index2 = np.random.choice(valid_indices2, size=1)[0]
            else:
              index2 = np.random.choice(range(len(seq2)))

            tmp = seq1[index1]
            seq1[index1] = seq2[index2]
            seq2[index2] = tmp

            tmp = plddt1[index1]
            plddt1[index1] = plddt2[index2]
            plddt2[index2] = tmp

            return seq1, plddt1, seq2, plddt2

           

        # Iterate through the DataFrame in pairs
        NewSeq = []
        NewPlddt = []
        Domain = []
        DomainPlddt = []
        for i in range(0, len(population.fittest_pop), 2):
            if i < len(population.fittest_pop) - 1: 
                cluster1 = [list(frozenset) for frozenset in domains_from_pae_matrix_networkx(population.fittest_pop["pae"].iloc[i][0])]
                cluster2 = [list(frozenset) for frozenset in domains_from_pae_matrix_networkx(population.fittest_pop["pae"].iloc[i + 1][0])]
                domain_specified_seq1 = ["".join(population.fittest_pop["sequence"].iloc[i][k] for k in subgroup) for subgroup in cluster1]
                domain_specified_seq2 = ["".join(population.fittest_pop["sequence"].iloc[i+1][k] for k in subgroup) for subgroup in cluster2]
                plddt_seq1 = [[population.fittest_pop["plddt"].iloc[i][k]for k in subgroup] for subgroup in cluster1]
                plddt_seq2 = [[population.fittest_pop["plddt"].iloc[i+1][k]for k in subgroup] for subgroup in cluster2]
                new_seq1, new_plddt1, new_seq2, new_plddt2 = swap_sequences(domain_specified_seq1.copy(), plddt_seq1.copy(), domain_specified_seq2.copy(), plddt_seq2.copy())
                
                index1, max_mean_sublist1 = max(enumerate(plddt_seq1), key=lambda x: sum(x[1]) / len(x[1]))
                high_plddt_domain1 = domain_specified_seq1[index1]
                high_plddt_domain1_value = max_mean_sublist1
                index2 , max_mean_sublist2 = max(enumerate(plddt_seq2), key=lambda x: sum(x[1]) / len(x[1]))
                high_plddt_domain2 = domain_specified_seq2[index2]
                high_plddt_domain2_value = max_mean_sublist2

                new_seq1 = ''.join(val for sublist in new_seq1 for val in sublist)
                new_seq2 = ''.join(val for sublist in new_seq2 for val in sublist)
                new_plddt1 = [value for sublist in new_plddt1 for value in sublist]
                new_plddt2 = [value for sublist in new_plddt2 for value in sublist]
            else: 
                cluster1 = [list(frozenset) for frozenset in domains_from_pae_matrix_networkx(population.fittest_pop["pae"].iloc[i][0])]
                cluster2 = [list(frozenset) for frozenset in domains_from_pae_matrix_networkx(population.fittest_pop["pae"].iloc[0][0])]
                domain_specified_seq1 = ["".join(population.fittest_pop["sequence"].iloc[i][k] for k in subgroup) for subgroup in cluster1]
                domain_specified_seq2 = ["".join(population.fittest_pop["sequence"].iloc[0][k] for k in subgroup) for subgroup in cluster2]
                plddt_seq1 = [[population.fittest_pop["plddt"].iloc[i][k]for k in subgroup] for subgroup in cluster1]
                plddt_seq2 = [[population.fittest_pop["plddt"].iloc[0][k]for k in subgroup] for subgroup in cluster2]
                new_seq1, new_plddt1, new_seq2, new_plddt2 = swap_sequences(domain_specified_seq1.copy(), plddt_seq1.copy(), domain_specified_seq2.copy(), plddt_seq2.copy())
                
                index1, max_mean_sublist1 = max(enumerate(plddt_seq1), key=lambda x: sum(x[1]) / len(x[1]))
                high_plddt_domain1 = domain_specified_seq1[index1]
                high_plddt_domain1_value = max_mean_sublist1
                index2, max_mean_sublist2 = max(enumerate(plddt_seq2), key=lambda x: sum(x[1]) / len(x[1]))
                high_plddt_domain2 = domain_specified_seq2[index2]
                high_plddt_domain2_value = max_mean_sublist2

                new_seq1 = ''.join(val for sublist in new_seq1 for val in sublist)
                new_seq2 = ''.join(val for sublist in new_seq2 for val in sublist)
                new_plddt1 = [value for sublist in new_plddt1 for value in sublist]
                new_plddt2 = [value for sublist in new_plddt2 for value in sublist] 

            NewSeq.append(new_seq1)
            NewSeq.append(new_seq2)
            NewPlddt.append(new_plddt1)
            NewPlddt.append(new_plddt2)
            Domain.append(high_plddt_domain1)
            Domain.append(high_plddt_domain2)
            DomainPlddt.append(high_plddt_domain1_value)
            DomainPlddt.append(high_plddt_domain2_value)

        #returning children and parents with their highest-plddt-value domain!
        offspring = pd.DataFrame({"sequence" : NewSeq, "plddt" : NewPlddt})
        domain = {"Domain" : Domain, "DomainPlddt" : DomainPlddt}
        offspring_pop = pd.concat([offspring,  population.fittest_pop[["sequence", "plddt"]]], ignore_index=True)
        offspring_pop = offspring_pop.sample(frac = 1)
        offspring_pop = offspring_pop.reset_index(drop = True)
        return Children(offspring_pop = offspring_pop, domain = domain)