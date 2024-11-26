import pandas as pd
import Selection


#A sample fitness score dataframe for testing
data= {
        'sequence': ['AT', 'GA', 'TGA', 'ACGT', 'CGTA'],
        'FitnessScore': [0.8, 0.6, 0.9, 0.7, 0.5]
    }
sample_data = pd.DataFrame(data)
length_threshold = 3


def test_keep_fittest_selects_half_population(sample_data, length_threshold):
    """Test the KeepFittest method of the Selection module to ensure that it selects half of the population.
        
    Arguments:
        -sample_data (pd.DataFrame): The dataframe containing sequences and their corresponding fitness scores.
        -length_threshold (int): The threshold length for selecting sequences.
    """
    select = Selection.Select()
    result = select.KeepFittest(sample_data, length_threshold)
    assert len(result.fittest_pop) == int(len(sample_data) / 2)
    

def test_keep_fittest_selects_highest_scores(sample_data, length_threshold):
    """Test the KeepFittest method of the Selection module to ensure it selects sequences with the highest fitness scores.
   
    Arguments:
        -sample_data (pd.DataFrame): The dataframe containing sequences and their corresponding fitness scores.
        -length_threshold (int): The threshold length for selecting sequences.
    """
    select = Selection.Select()
    result = select.KeepFittest(sample_data, length_threshold)
    assert result.fittest_pop['FitnessScore'].min() >= sample_data['FitnessScore'].median()
    

def test_keep_fittest_shuffles_population(sample_data, length_threshold):
    
    """Test the KeepFittest method of the Selection module to ensure that it shuffles the population.
  
    Argumentss:
        -sample_data (pd.DataFrame): The dataframe containing sequences and their corresponding fitness scores.
        -length_threshold (int): The threshold length for selecting sequences.

    """
    select = Selection.Select()
    result = select.KeepFittest(sample_data, length_threshold)
    assert not result.fittest_pop['sequence'].equals(sample_data['sequence'].sort_values(ascending=False))
 
  
def test_keep_fittest_select_threshold_sequences(sample_data, length_threshold):
    
    """Test the KeepFittest method of the Selection module to ensure that it selects sequences longer than a specified threshold.
  
    Arguments:
        -sample_data (pd.DataFrame): The dataframe containing sequences and their corresponding fitness scores.
        -length_threshold (int): The threshold length for selecting sequences.
    """
    select = Selection.Select()
    result = select.KeepFittest(sample_data, length_threshold)
    for seq in result["sequence"]:
        assert len(seq) >=  length_threshold




