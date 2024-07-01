import pytest
import numpy as np
import sys

sys.path.insert(0,"/Users/kianaseraj/desktop/github-kianaseraj/ProteinEvolution/src")

from Breeding import domains_from_pae_matrix_networkx, swap_sequences, DomainCrossover, MatingPool, Children

@pytest.fixture
def sample_pae_matrix():
    return np.random.rand(10, 10) * 20  # Random 10x10 matrix with values between 0 and 20

@pytest.fixture
def sample_mating_pool():
    data = {
        'sequence': ['ACGT' * 25, 'TGCA' * 25],
        'plddt': [list(np.random.rand(100)) for _ in range(2)],
        'pae': [np.random.rand(100, 100) * 20 for _ in range(2)]
    }
    return MatingPool(fittest_pop=pd.DataFrame(data))

def test_domains_from_pae_matrix_networkx(sample_pae_matrix):
    clusters = domains_from_pae_matrix_networkx(sample_pae_matrix)
    assert isinstance(clusters, list)
    assert all(isinstance(cluster, frozenset) for cluster in clusters)

def test_swap_sequences():
    seq1 = ['ACGT', 'TGCA']
    plddt1 = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    seq2 = ['AAAA', 'TTTT']
    plddt2 = [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]

    new_seq1, new_plddt1, new_seq2, new_plddt2 = swap_sequences(seq1, plddt1, seq2, plddt2)

    assert len(new_seq1) == len(seq1)
    assert len(new_seq2) == len(seq2)
    assert len(new_plddt1) == len(plddt1)
    assert len(new_plddt2) == len(plddt2)
    assert new_seq1 != seq1 or new_seq2 != seq2  # At least one sequence should have changed

def test_domain_crossover_init():
    crossover = DomainCrossover()
    assert isinstance(crossover, DomainCrossover)

def test_domain_crossover_odd_population(sample_mating_pool):
    # Add an odd entry to the mating pool
    odd_entry = pd.DataFrame({
        'sequence': ['ATCG' * 25],
        'plddt': [list(np.random.rand(100))],
        'pae': [np.random.rand(100, 100) * 20]
    })
    sample_mating_pool.fittest_pop = pd.concat([sample_mating_pool.fittest_pop, odd_entry], ignore_index=True)

    crossover = DomainCrossover()
    result = crossover.crossover(sample_mating_pool)

    assert isinstance(result, Children)
    assert isinstance(result.offspring_pop, pd.DataFrame)
    assert len(result.offspring_pop) == len(sample_mating_pool.fittest_pop) * 2
    assert 'sequence' in result.offspring_pop.columns
    assert 'plddt' in result.offspring_pop.columns

@pytest.mark.parametrize("pae_power,pae_cutoff,graph_resolution", [
    (1, 10, 0.5),
    (2, 15, 0.7),
    (0.5, 5, 0.3)
])
def test_domains_from_pae_matrix_networkx_parameters(sample_pae_matrix, pae_power, pae_cutoff, graph_resolution):
    clusters = domains_from_pae_matrix_networkx(sample_pae_matrix, pae_power, pae_cutoff, graph_resolution)
    assert isinstance(clusters, list)
    assert all(isinstance(cluster, frozenset) for cluster in clusters)
