import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys

sys.path.insert(0,"/Users/kianaseraj/desktop/github-kianaseraj/ProteinEvolution/src")


from Mutation import GeneratingDomain, insertion, masking, MaskedSubstitution

@pytest.fixture
def mock_model():
    return Mock()

@pytest.fixture
def mock_alphabet():
    alphabet = Mock()
    alphabet.get_batch_converter.return_value = lambda x: (None, None, torch.zeros((1, 10, 1)))
    alphabet.mask_idx = 0
    alphabet.all_toks = list(range(30))
    return alphabet


def test_generating_domain():
    domain = GeneratingDomain(10)
    assert len(domain) == 10
    assert all(aa in "ACGTSLDNQPFVYWIHERK" for aa in domain)

def test_insertion():
    seq = "ACGT"
    result = insertion(seq, 10)
    assert len(result) == 10
    assert seq in result

def test_mutation_initialization(mock_model, mock_alphabet):
    mutation = MaskedSubstitution(mock_model, mock_alphabet, 100)
    assert mutation.model == mock_model
    assert mutation.alphabet == mock_alphabet
    assert mutation.length == 100


def test_insertion_with_short_sequence():
    seq = "AC"
    result = insertion(seq, 10)
    assert len(result) == 10
    assert seq in result

def test_masking_with_all_same_plddt(mock_model, mock_alphabet):
    seq = "ACGT"
    plddt = np.array([0.5, 0.5, 0.5, 0.5])
    
    mock_model.return_value = {"logits": torch.zeros((1, 4, 30))}
    
    result = masking(seq, plddt, mock_model, mock_alphabet)
    assert len(result) == 4

