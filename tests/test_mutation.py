import torch
import random
from utils import amino_acid
from Mutation import GeneratingDomain, insertion, esm_log_probability, substitution
from Folding import ESMmodel

# Set the random seed for reproducibility in tests
random.seed(42)


#load esm model and tokenizing module

dir_path = "/PATH/to/load/ESM/model"
esm_model = ESMmodel()
mut_model, mut_alphabet = esm_model.load(dir_path)

#Testing GeneratingDomain funciton; the length of the generated domain must be the same as the input length
def test_generating_domain_length():
    domain = GeneratingDomain(10, amino_acid)
    assert len(domain) == 10
       
#Testing GeneratingDomain funciton; the domain must be composed of only 20 standard amino acids    
def test_generating_domain_amino_acids():  
    domain = GeneratingDomain(20, amino_acid) 
    assert all(aa in "ACGTSLDNQPFVYWIHERK" for aa in domain)

#Testing insertion function; the output sequence must have the same length as the desired length
def test_insertion_output_length():
    seq = "TLIFACENDSIAPVN"
    result = insertion(seq, 30)
    assert len(result) == 30
 
#Testing esm_log_probability function; the log probabilities must have the shape of (seq_length+2)*33, adding 2 is because of the first and end tokens.
def test_esm_log_probability_shape():
    sequence = "TLIFACENDSIAPVN"
    log_probabilities = esm_log_probability(sequence, mut_model, mut_alphabet, masked_position = [2])
    log_probabilities.shape == torch.Size([len(sequence)+2, 33])
    
#Testing substitution function; the masked amino acids should not be substituted with the original residues
def test_substitution_amino_acids():
    original_sequence = "TLIFACENDSIAPVN"
    plddt = [0.21, 0.77, 0.15, 0.65, 0.44, 0.81, 0.34, 0.21, 0.52, 0.46, 0.32, 0.28, 0.62, 0.3, 0.17 ]
    masked_position, mutated_sequence = substitution(original_sequence, plddt, mut_model, mut_alphabet)
    for index in  masked_position:
        assert mutated_sequence[index] != original_sequence[index]
    

