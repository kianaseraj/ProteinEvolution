import numpy as np

from typing import List
from abc import ABC, abstractmethod



"""
First stage of the algorithm.
Initial population of sequences will be generated.
"""


class ProteinSequenes(ABC):
    def __init__(self, num_sequence, seq_length) -> None:
        self.num_sequence = num_sequence
        self.length = seq_length

    @abstractmethod
    def GeneratingPopulation(self, amino_acid) -> List[str]:
        pass



#mutation function used in GeneratingPopulation attribute of the class to mutate 20% of each generated sequence
def mutation(seq, amino_acid = ["A", "C", "G", "T", "S", "D", "L", "N", "Q", "P", "F", "V", "Y", "W", "I", "H", "E", "R", "K", "M"]) -> List[str]:
            length = len(seq)
            index = np.random.choice(range(length-1), int(0.2*length), replace=False)
            aa = np.random.choice(amino_acid, int(0.2*length), replace = True )
            seq = list(seq)
            for i, j in enumerate(index):
                seq[j] = aa[i]
            seq = "".join(seq)
            return seq

class RandomPopulation(ProteinSequenes):

    def __init__(self, num_sequence, seq_length):
        super().__init__(num_sequence, seq_length)


    def GeneratingPopulation(self,amino_acid = ["A", "C", "G", "T", "S", "D", "L", "N", "Q", "P", "F", "V", "Y", "W", "I", "H", "E", "R", "K", "M"]) -> List[str]:

        sequence_produced = ""
        for k in range(self.length):
            # no more than three consecutive amino acid repeats
            if k < 3 or sequence_produced[k-3:k] != sequence_produced[k-2:k+1]:
                amino = np.random.choice(amino_acid)
                sequence_produced += amino

        proteins = []
        for l in range(self.num_sequence):
            mutated_seq = mutation(sequence_produced)
            proteins.append(mutated_seq)

        return proteins
