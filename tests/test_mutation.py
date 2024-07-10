import pytest
import numpy as np
import torch
from unittest.mock import Mock
import sys

sys.path.insert(0,"/Users/kianaseraj/desktop/github-kianaseraj/ProteinEvolution/src")


from Mutation import GeneratingDomain, insertion, masking, MaskedSubstitution


def test_generating_domain():
    domain = GeneratingDomain(10)
    assert len(domain) == 10
    assert all(aa in "ACGTSLDNQPFVYWIHERK" for aa in domain)

def test_insertion():
    seq = "ACGT"
    result = insertion(seq, 10)
    assert len(result) == 10
    assert seq in result


