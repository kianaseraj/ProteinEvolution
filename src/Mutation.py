import numpy as np
from typing import List
import torch
from abc import ABC, abstractmethod
import random
from dataclasses import dataclass
from Folding import ESMmodel


"""
The last stage of the algorithm.
Mutation operation will be applied on children popuaiton to change 3% of the sequences with amino acids gained the highest probability from masked language modeling
"""

class Mutation(ABC):
    def __init__(self, model, model_alphabet, seq_length):
        self.model = model
        self.alphabet = model_alphabet
        self.length = seq_length
        
        
    @abstractmethod
    def mutate(self, Individuals) -> List[str]:
        pass


def GeneratingDomain(length:int, amino_acid = ["A", "C", "G", "T", "S", "D", "L", "N", "Q", "P", "F", "V", "Y", "W", "I", "H", "E", "R", "K", "M"]) -> List[str]:
    """
    generating the random domain to be inserted
    """
    sequence_produced = ""
    for k in range(length):
        # no more than three consecutive amino acid repeats
        if k < 3 or sequence_produced[k-3:k] != sequence_produced[k-2:k+1]:
            amino = np.random.choice(amino_acid)
            sequence_produced += amino
        return sequence_produced
    
    
def insertion(seq, length):
    """
    Inserting randomely a domain in a seq
    """
    domain_length = length - len(seq)
    insertion_position = random.randint(0, len(seq))


    mutated_seq = seq[:insertion_position] + GeneratingDomain(domain_length) + seq[insertion_position:]
    return mutated_seq


def masking(seq, plddt, mut_model, mut_alphabet):
    """
    Substituing masked amino acid wrt 5 most probable amino acids predicted with ESM2
    """
    batch_converter = mut_alphabet.get_batch_converter()
    data = [("seq", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    index = np.argsort(plddt)
    masked_position = index[0:int(0.03 * len(index))]
    batch_tokens[0][masked_position] = mut_alphabet.mask_idx
    with torch.no_grad():
        results = mut_model(batch_tokens, repr_layers=[33], return_contacts=False)
    logits = results["logits"]
    probabilities = torch.nn.functional.softmax(logits[0][masked_position], dim=1)
    log_probabilities = torch.log(probabilities)
    aa_index = mut_alphabet.all_toks[4:28]
    pose = []
    for a in log_probabilities:
        high = np.argsort(a[4:28])[-5:]
        ind = np.random.choice(high)
        pose.append(aa_index[ind])
        
    # Create a dictionary to map the masked_position to the new letters
    substitution_map = {index: new_letter for index, new_letter in zip(masked_position, pose)}

    # Substitute the letters in the sequence based on the mapping
    substituted_sequence = ''.join(substitution_map.get(old_index, seq[old_index]) for old_index in range(len(seq)))
    return substituted_sequence




class MaskedSubstitution(Mutation):

    def __init__(self, model, model_alphabet, length):
        super().__init__(model, model_alphabet, length)

    def mutate(self, Individuals) -> List[str]:

       MutatedChildren = []
       for i in range(len(Individuals.offspring_pop)):
          seq = Individuals.offspring_pop["sequence"][i]
          plddt = Individuals.offspring_pop["plddt"][i]
          seq = masking(seq, plddt, self.model, self.alphabet)

          if len(seq)<self.length:
              mutated_seq = insertion(seq)
          else:
              mutated_seq = seq

          mutated_seqs = ":".join([mutated_seq]*2)
          MutatedChildren.append(mutated_seqs)

       return MutatedChildren

       
       
