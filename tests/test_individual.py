import sys

sys.path.insert(0, "/Users/kianaseraj/desktop/github-kianaseraj/ProteinEvolution/src")

import Individual
#testing the individual module

def test_sequence_similarity(self, random_population):
        chain_num = 1
        sequence_num = 10
        length = 100
        result = random_population.GeneratingPopulation(chain_num, sequence_num, length)
        
        # Check if sequences are not identical (80% similarity rule)
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                similarity = sum(a == b for a, b in zip(result[i], result[j])) / length
                assert 0.7 <= similarity <= 0.9  # allowing some flexibility

def test_no_triple_repeats(self, random_population):
        chain_num = 1
        sequence_num = 10
        length = 50
        result = random_population.GeneratingPopulation(chain_num, sequence_num, length)
        
        for seq in result:
            for i in range(len(seq) - 2):
                assert not (seq[i] == seq[i+1] == seq[i+2])

