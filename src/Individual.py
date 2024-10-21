import numpy as np
from typing import List
from utils import amino_acid


def mutation(seq:str, length:int) -> str:
  num_mutations = int(0.2 * length) # Calculate the number of mutations needed for 80% similarity
  indices = np.random.choice(range(length), num_mutations, replace=False) # Randomly select indices to mutate
  seq_list = list(seq) # Convert sequence to a list for mutation
  for index in indices:
    original_aa = seq[index]
    possible_mutations = [aa for aa in amino_acid if aa != original_aa]
    seq_list[index] = np.random.choice(possible_mutations)
  seq = "".join(seq_list) # Convert list back to string
  return seq


#First stage: Generating protein sequences randomly with 80% residue similarity.
def GeneratingPopulation(chain_num : int, sequence_num : int, length:int) -> List[str]:
  sequence_produced = ""
  for k in range(length):
      # no more than three consecutive amino acid repeats
      if k < 3 or sequence_produced[k-3:k] != sequence_produced[k-2:k+1]:
          amino = np.random.choice(amino_acid)
          sequence_produced += amino

  proteins = []
  for _ in range(sequence_num):
      mutated_seq = mutation(sequence_produced, length)
      proteins.append(mutated_seq)

  if chain_num == 1:
    return proteins

  else:
    complex = []
    for seq in proteins:
      seqs = ":".join([seq]*chain_num)
      complex.append(seqs)
    return(complex)
        
