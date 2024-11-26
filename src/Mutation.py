import numpy as np
from typing import List
import torch
from abc import ABC, abstractmethod
import random
from utils import amino_acid, aa_index
from dataclasses import dataclass
from Breeding



"""
The last stage of the algorithm.
Mutation operation will be applied on children popuaiton to change 3% of the sequences with amino acids gained the highest probability from masked language modeling
"""

def GeneratingDomain(length:int, amino_acid: List[str]) -> str:
    """Generatin domains randomly, the domains will be used later to be inserted in sequenes shoter than a thershold
    Arguments:
       -length: Length of a desired domain
       -amino_acid: List of residues used for generarting the domains
    returns: Generates a string of amino acids corresponding to a domain
    
    """
    domain_produced = ""
    for k in range(length):
        # no more than three consecutive amino acid repeats
        if k < 3 or domain_produced[k-3:k] != domain_produced[k-2:k+1]:
            amino = np.random.choice(amino_acid)
            domain_produced += amino
    return domain_produced
    
    
def insertion(seq: str, length: int):
    """Mutating a sequecne by insertiton opeation and randomely inseting domain in a sequence
    Arguments:
        - seq: a chosen sequence for insertioin operation
        -length: a desired length to have after insertion
    Returns: A mutated sequence with the desired length
    """
    
    #length differnce between the chosen sequence and the desired length after mutation, the difference value will be used later to generate a domain
    domain_length = length - len(seq) 
    #randomly choosing a positoin to insert the domain
    insertion_position = random.randint(0, len(seq)) 
    # Create the mutated sequence with the inserted domain
    mutated_seq = seq[:insertion_position] + GeneratingDomain(domain_length, amino_acid) + seq[insertion_position:]
    return mutated_seq



def esm_log_probability(seq, masked_position, mut_model, mut_alphabet):
    """Extract log probabilities for each amino acid in a sequnce with ESM
    Arguments: 
        - seq: A potein sequence 
        - mut_model: Loaded model of ESM to generate logits values for sequence's residues
        - mut_alphabet: ESM's tokenization module to tokenize the seuqnces as inputs for the subsequent inferece task by a langugae model
    Returns: A log probability values for each token of the sequence will be generated.
    """
    # Prepare the batch using the ESM alphabet batch converter
    batch_converter = mut_alphabet.get_batch_converter()
    data = [("seq", seq)] #The required input format of a sequence to be tokenized by the model
    _ , _ , batch_tokens = batch_converter(data) #Tokenizing the sequence
    # Mask the specified position
    batch_tokens[0][masked_position] = mut_alphabet.mask_idx
    with torch.no_grad():
        results = mut_model(batch_tokens, repr_layers=[33], return_contacts=False)
    logits = results["logits"]#Extract the logit values of the neural network
    # Apply softmax and log transformation to get probabilities
    probabilities = torch.nn.functional.softmax(logits[0][masked_position], dim=1) #softmax transformation of the logits to get probability values
    log_probabilities = torch.log(probabilities)
    return log_probabilities
    



def substitution(seq, plddt, mut_model, mut_alphabet):
    """Substituing masked amino acids wrt 5 most probable amino acids predicted with ESM2
    Arguments: 
        - seq: A sequence of amino acids
        - plddt: Numpy array of plddt scores having a shape of seq_lenght*seq_length
        - mut_model: Loaded model of ESM to generate logits values for sequences and their masked tokens
        - mut_alphabet: ESM's tokenization module to tokenize the seuqnces as inputs for the subsequent inferece task by a langugae model
    Returns: A sequence mutated with substitution and insertion operations.
    """
    
    index = np.argsort(plddt) #plddt indices in an ascending form
    masked_position = index[0:int(0.03 * len(index))] #masking 3% of amino acids of a sequence having the lowest plddt values
    log_probabilities = esm_log_probability(seq, masked_position, mut_model, mut_alphabet) #extracting log porbabilities of substitute residues for masekd tokens
    
    #create a list of amino acid substitutes with high porbabilities.
    aa_substitute = []
    for probability in log_probabilities:
        high = np.argsort(probability[4:28])[-5:] #sorting the substitues's indices in an ascnding format and choosing five of them with the highest log prorbability values.
        ind = np.random.choice(high) #out of 5 chosen substitutes with high probabilities, one is randomely chosen
        aa_substitute.append(aa_index[ind])    
    # Create a dictionary to map the masked_position to the new letters
    substitution_map = {index: new_letter for index, new_letter in zip(masked_position, aa_substitute)}

    # Substitute the letters in the sequence based on the mapping
    substituted_sequence = ''.join(substitution_map.get(old_index, seq[old_index]) for old_index in range(len(seq)))
    return masked_position, substituted_sequence


class Mutation(ABC):
    """
    Abstract base class defining the interface for mutation operations.

    This class serves as a blueprint for implementing specific mutation strategies
    to introduce variability in the population.
    """
    def __init__(self, model, model_alphabet, seq_length):
        """Initializes the Mutation class.

        Arguments:
        - model: The ESM model used for mutation operations.
        - model_alphabet: The ESM alphabet used for tokenizing sequences.
        - seq_length (int): The length threshold for sequences, used in insertion operations.
        """
        self.model = model
        self.alphabet = model_alphabet
        self.length = seq_length
        
        
    @abstractmethod
    def mutate(self, Individuals) -> List[str]:
        """Abstract method to define the mutation operation.

        Arguments:
        - Individuals: A population of protein sequences.

        Returns:
        - List[str]: A list of mutated sequences.
        """
        pass

class MaskedSubstitution(Mutation):
    """
    Implements the Mutation class with masked substitution and insertion operations.

    The mutation strategy involves substituting low-confidence residues (low PLDDT scores) 
    and inserting randomly generated domains for sequences shorter than a specified threshold.
    """

    def __init__(self, model, model_alphabet, length):
        """Initializes the MaskedSubstitution class.

        Arguments:
        - model: The ESM model used for mutation operations.
        - model_alphabet: The ESM alphabet used for tokenizing sequences.
        - length (int): The length threshold for sequences, used in insertion operations.
        """
        super().__init__(model, model_alphabet, length)

    def mutate(self, Individuals) -> List[str]:
        
       """Mutating sequences with insertion and substituttion operations
       Arguments:
            - Individuals:  Population of proten sequecnes geneated by Breeding module
            
       Returns: List of mutated sequences 
        
       """

       #Generate a list of population mutated by insertion and substitution operations
       MutatedChildren = []
       for i in range(len(Individuals.offspring_pop)):
          seq = Individuals.offspring_pop["sequence"][i]
          plddt = Individuals.offspring_pop["plddt"][i]
          _, seq = substitution(seq, plddt, self.model, self.alphabet) #masking the sequences and substitute them
          
          #For sequences having lenghts less than the threshold, a randomly generated domain will be inserted
          if len(seq)<self.length:
              mutated_seq = insertion(seq)
          else:
              mutated_seq = seq
          # Duplicate the mutated sequence for both chains and join with ":"
          mutated_seqs = ":".join([mutated_seq]*2)
          MutatedChildren.append(mutated_seqs)

       return MutatedChildren

       
       
