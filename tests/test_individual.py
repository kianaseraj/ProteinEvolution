import sys
import numpy as np
import Individual






def test_mutation_function():
    """Test the mutation function to ensure that it preserves approximately 80% of the input sequence unchanged.
    
    This test takes an input sequence and uses the mutation function from the Individual module to generate a mutated
    sequence. It then calculates the similarity ratio between the original and mutated sequence to verify that the mutation
    function leaves around 80% of the sequence unchanged.
    The allowed range for similarity is between 79% and 81% to account for minor deviations in mutation.
    """ 
    
    # Set a random seed for reproducibility
    np.random.seed(42)
    
    sequence = "IIPYKLEMTSLLLEMMMTTSLYILMTSLYIF"
    length = len(sequence)
    result = Individual.mutation(sequence, length)
    similarity = sum(a == b for a, b in zip(sequence, result)) / length
    assert 0.79 <= similarity <= 0.81  # allowing some flexibility
    

def test_no_triple_repeats():
    """Test the GeneratingPopulation function to ensure that no sequence contains more than 3 consecutive repeated amino acids.
    
    This test generates a population of sequences using the GeneratingPopulation function from the Individual module.
    It then iterates through each generated sequence to verify that there are no instances of more than three identical
    amino acids occurring consecutively in a row.
    Raises an assertion error if more than three consecutive amino acids are found.
    """
    
    # Set a random seed for reproducibility
    np.random.seed(42)
    
    chain_num = 1
    sequence_num = 10
    length = 50
    result = Individual.GeneratingPopulation(chain_num, sequence_num, length)
    print(result)
    for seq in result:
        for i in range(len(seq) - 3):
            assert not (seq[i] == seq[i+1] == seq[i+2] == seq[i+3])
                

def test_generated_chains():
    
    """Test the GeneratingPopulation function to ensure that generated chains are correctly formed.
    
    This test generates a population of chains using the GeneratingPopulation function from the Individual module.
    Each chain is composed of sequences separated by the ":" character. This function verifies that the sequences 
    in each chain are identical to each other.
    Raises an assertion error if the sequences separated by ":" are not equal.
    """
    
    # Set a random seed for reproducibility
    np.random.seed(42)
    
    chain_num = 3
    sequence_num = 10
    length = 50
    result = Individual.GeneratingPopulation(chain_num, sequence_num, length)
    for seq in result:
        assert seq.split(":")[0] == seq.split(":")[1] == seq.split(":")[2]
    

