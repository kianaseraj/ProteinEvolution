import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0,"/Users/kianaseraj/desktop/github-kianaseraj/ProteinEvolution/src")

import Selection
from Selection import MatingPool
from Fitness import OptimizationFitness


#A sample dataset for testing
data= {
        'sequence': ['ATCG', 'GCTA', 'TGCA', 'ACGT', 'CGTA'],
        'FitnessScore': [0.8, 0.6, 0.9, 0.7, 0.5]
    }
sample_data = pd.DataFrame(data)


#Testing keepfitest attribute of the selection class

def test_keep_fittest_selects_half_population(sample_data):
    select = Selection.Select()
    result = select.KeepFittest(sample_data)
    assert len(result.fittest_pop) == len(sample_data) / 2
    
def test_keep_fittest_selects_highest_scores(sample_data):
    select = Selection.Select()
    result = select.KeepFittest(sample_data)
    assert result.fittest_pop['FitnessScore'].min() >= sample_data['FitnessScore'].median()
    
def test_keep_fittest_shuffles_population(sample_data):
    select = Selection.Select()
    result = select.KeepFittest(sample_data)
    assert not result.fittest_pop['sequence'].equals(sample_data['sequence'].sort_values(ascending=False))
    





