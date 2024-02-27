import numpy as np

from typing import List
import torch
from abc import ABC, abstractmethod
import esm
from dataclasses import dataclass


class Substitution(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def mutate(self, Individuals : Children) -> List[str]:
        pass

class MaskedMutation(Substitution):

    def __init__(self):
        super().__init__()
        
    def mutate(self, Individuals : Children) -> List[str]:

       def masking(seq, plddt):
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

       MutatedChildren = []
       for i in range(len(Individuals.offspring_pop)):
          seq = Individuals.offspring_pop["sequence"][i]
          plddt = Individuals.offspring_pop["plddt"][i]
          MutatedChildren.append(masking(seq, plddt))
          complex = []
          for seq in MutatedChildren:
            seqs = ":".join([seq]*chain_num)
            complex.append(seqs)
          return(complex)
       return MutatedChildren 