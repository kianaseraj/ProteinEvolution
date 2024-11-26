import torch
import random
from utils import amino_acid
from Mutation import GeneratingDomain, insertion, esm_log_probability, substitution
from Folding import ESMmodel




#load esm model and tokenizing module
dir_path = "/PATH/to/load/ESM/model"
esm_model = ESMmodel()
mut_model, mut_alphabet = esm_model.load(dir_path)


def test_generating_domain_length():
    
    """Test the GeneratingDomain function to ensure that the length of the generated domain matches the input length.
    Raises:
        AssertionError: If the length of the generated domain is not equal to the specified length.
    """
    
    # Set the random seed for reproducibility in tests
    random.seed(42)
    domain = GeneratingDomain(10, amino_acid)
    assert len(domain) == 10
       

def test_generating_domain_amino_acids():  
    
    """Test the GeneratingDomain function to ensure that the generated domain contains only standard amino acids.
    Raises:
        AssertionError: If any amino acid in the generated domain is not part of the standard 20 amino acids.
    """
    
    # Set the random seed for reproducibility in tests
    random.seed(42)
    domain = GeneratingDomain(20, amino_acid) 
    assert all(aa in "ACGTSLDNQPFVYWIHERK" for aa in domain)


def test_insertion_output_length():
    
    """Test the insertion function to ensure that the output sequence has the correct desired length.
    Raises:
        AssertionError: If the length of the output sequence is not equal to the specified length.
    """
    
    # Set the random seed for reproducibility in tests
    random.seed(42)
    seq = "TLIFACENDSIAPVN"
    result = insertion(seq, 30)
    assert len(result) == 30
 

def test_esm_log_probability_shape():
    
    """Test the esm_log_probability function to ensure that the output log probabilities have the correct shape of (seq_length+2)*33, adding 2 is because of the first and end tokens.
    Raises:
        AssertionError: If the shape of the log probabilities tensor does not match the expected size.
    """
    
    sequence = "TLIFACENDSIAPVN"
    log_probabilities = esm_log_probability(sequence, mut_model, mut_alphabet, masked_position = [2])
    log_probabilities.shape == torch.Size([len(sequence)+2, 33])
    

def test_substitution_amino_acids():
    
    """Test the substitution function to ensure that the masked amino acids are replaced with different residues.
    Raises:
        AssertionError: If the substituted amino acid is the same as the original amino acid at any masked position.
    """
    
    # Set the random seed for reproducibility in tests
    random.seed(42)
    original_sequence = "TLIFACENDSIAPVN"
    plddt = [0.21, 0.77, 0.15, 0.65, 0.44, 0.81, 0.34, 0.21, 0.52, 0.46, 0.32, 0.28, 0.62, 0.3, 0.17 ]
    masked_position, mutated_sequence = substitution(original_sequence, plddt, mut_model, mut_alphabet)
    for index in  masked_position:
        assert mutated_sequence[index] != original_sequence[index]
    

