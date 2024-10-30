import numpy as np
from typing import List
from utils import amino_acid


def mutation(seq:str, length:int) -> str:
  """Mutating 20% of a sequence randomly
  Arguments:
    - seq: A protein sequence
    - length: the length of the input sequence
  Returns: returns a string of protein sequence
  """
  num_mutations = int(0.2 * length) # Calculate the number of mutations needed for 80% similarity
  indices = np.random.choice(range(length), num_mutations, replace=False) # Randomly select indices for mutation
  seq_list = list(seq) # Convert the sequence to a list for mutation
  for index in indices:
    original_aa = seq[index]
    possible_mutations = [aa for aa in amino_acid if aa != original_aa]
    seq_list[index] = np.random.choice(possible_mutations)
  seq = "".join(seq_list) # Convert the list back to a string
  return seq


#First stage: Generating protein sequences randomly with 80% residue similarity.
def GeneratingPopulation(chain_num : int, sequence_num : int, length:int) -> List[str]:
  """Generating randomly a protein population
  Arguments:
    - chain_num: number of chains in a sequence
    - sequence_num : number of sequences in the population
    - length : length of each protein sequence
  Returns : A list of protein sequences
  """
  
  sequence_produced = ""
  for k in range(length):
      # no more than three consecutive amino acid repeats
      if k < 3 or sequence_produced[k-3:k] != sequence_produced[k-2:k+1]:
          amino = np.random.choice(amino_acid)
          sequence_produced += amino

  proteins = []
  for _ in range(sequence_num):
      mutated_seq = mutation(sequence_produced, length) #mutate each sequence in the population 
      proteins.append(mutated_seq)

  if chain_num == 1:
    return proteins
  #if there are more than one chain in the sequence; the chains will be separated by ":"
  else:
    complex = []
    for seq in proteins:
      seqs = ":".join([seq]*chain_num)
      complex.append(seqs)
    return(complex)
        
