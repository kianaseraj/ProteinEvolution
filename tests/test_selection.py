import pandas as pd
import Selection


#A sample fitness score dataframe for testing
data= {
        'sequence': ['AT', 'GA', 'TGA', 'ACGT', 'CGTA'],
        'FitnessScore': [0.8, 0.6, 0.9, 0.7, 0.5]
    }
sample_data = pd.DataFrame(data)
length_threshold = 3

#Testing keepfittest method of Selection module; testing if the method keeps the half of the population
def test_keep_fittest_selects_half_population(sample_data, length_threshold):
    select = Selection.Select()
    result = select.KeepFittest(sample_data, length_threshold)
    assert len(result.fittest_pop) == int(len(sample_data) / 2)
    
#Testing keepfittest method of Selection module; testing if the module keeps the highestt scores  
def test_keep_fittest_selects_highest_scores(sample_data, length_threshold):
    select = Selection.Select()
    result = select.KeepFittest(sample_data)
    assert result.fittest_pop['FitnessScore'].min() >= sample_data['FitnessScore'].median()
    
#Testing keepfittest method of Selection module; testing if the module shuffle the data entries
def test_keep_fittest_shuffles_population(sample_data, length_threshold):
    select = Selection.Select()
    result = select.KeepFittest(sample_data)
    assert not result.fittest_pop['sequence'].equals(sample_data['sequence'].sort_values(ascending=False))
 
  
#Testing keepfittest method of Selection module; testing if the module selects sequences longer than a threshold
def test_keep_fittest_select_threshold_sequences(sample_data, length_threshold):
    select = Selection.Select()
    result = select.KeepFittest(sample_data)
    for seq in result["sequence"]:
        assert len(seq) >=  length_threshold




