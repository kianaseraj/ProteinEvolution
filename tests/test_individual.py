import sys
import numpy as np


import Individual


# Set a random seed for reproducibility
np.random.seed(42)


#test mutation function; the function must preserve 80% of the sequence unchanged
def test_mutation_function():
    sequence = "IIPYKLEMTSLLLEMMMTTSLYILMTSLYIF"
    length = len(sequence)
    result = Individual.mutation(sequence, length)
    similarity = sum(a == b for a, b in zip(sequence, result)) / length
    assert 0.79 <= similarity <= 0.81  # allowing some flexibility
    
#test GenerattingPopulation function; the sequences must not have more than 3 consecuive repeated amino acids in a row.
def test_no_triple_repeats():
        chain_num = 1
        sequence_num = 10
        length = 50
        result = Individual.GeneratingPopulation(chain_num, sequence_num, length)
        print(result)
        for seq in result:
            for i in range(len(seq) - 3):
                assert not (seq[i] == seq[i+1] == seq[i+2] == seq[i+3])
                
# test GenerattingPopulation function; the chains must composed of ":" and sequences separated by ":" must be equal to each other.
def test_generated_chains():
    chain_num = 3
    sequence_num = 10
    length = 50
    result = Individual.GeneratingPopulation(chain_num, sequence_num, length)
    for seq in result:
        assert seq.split(":")[0] == seq.split(":")[1] == seq.split(":")[2]
    

