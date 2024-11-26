import pytest
import numpy as np
import random
from Breeding import domains_from_pae_matrix_networkx, swap_sequences



@pytest.fixture
def pae_matrix():
  
    """Fixture that provides a sample PAE matrix for testing purposes.
    Returns:
        -np.ndarray: A 10x10 numpy array representing the PAE matrix.
    """
    return np.array([[ 0.2553455 ,  2.1096013 ,  4.3020062 ,  6.0023804 ,  5.70045   ,
         5.0911565 ,  7.6306534 ,  8.57113   ,  8.941341  , 10.038839  ],
       [ 1.0851111 ,  0.25086007,  1.3389152 ,  2.1682181 ,  2.9111462 ,
         3.308     ,  3.1444001 ,  3.699607  ,  4.4982023 ,  5.1512394 ],
       [ 2.4342415 ,  0.8953974 ,  0.25022718,  1.1943722 ,  2.0848699 ,
         2.9402454 ,  2.527351  ,  2.5925205 ,  4.011569  ,  4.75624   ],
       [ 3.7924576 ,  1.9523008 ,  0.8081494 ,  0.2503295 ,  0.86523575,
         1.1504298 ,  1.2602508 ,  1.2063042 ,  1.4133687 ,  1.803237  ],
       [ 4.238002  ,  2.8546662 ,  1.1384538 ,  0.7814436 ,  0.25022808,
         0.7707377 ,  0.89012283,  1.0171578 ,  1.0734541 ,  1.105521  ],
       [ 4.2496943 ,  2.7558198 ,  1.2276105 ,  0.9345625 ,  0.7673484 ,
         0.25007573,  0.76243436,  0.8578852 ,  0.9398722 ,  0.90837145],
       [ 4.548851  ,  2.724541  ,  1.3551605 ,  0.9887471 ,  0.8582412 ,
         0.75723445,  0.25006768,  0.7628012 ,  0.8257399 ,  0.8613892 ],
       [ 4.0813    ,  2.9213707 ,  1.4214323 ,  1.0407493 ,  0.9500258 ,
         0.8255142 ,  0.7588707 ,  0.25017142,  0.7743008 ,  0.8490029 ],
       [ 4.4147654 ,  3.1428874 ,  1.4685389 ,  1.1016421 ,  0.94352436,
         0.83242   ,  0.8204022 ,  0.76834327,  0.25068104,  0.75889844],
       [ 4.122622  ,  3.0157313 ,  1.6225138 ,  1.2238846 ,  1.0305163 ,
         0.8540813 ,  0.8364479 ,  0.81597555,  0.758469  ,  0.25004017]])


def test_all_indices_in_clusters(pae_matrix):
  
     """Test domains_from_pae_matrix_networkx function to ensure all indices from the PAE matrix are present in the clusters.
    
     Arguments:
        -pae_matrix (np.ndarray): A numpy array representing the PAE matrix.
    
     Raises:
        AssertionError: If not all indices are present in the output clusters.
     """
  
     indices = [index for sublist in domains_from_pae_matrix_networkx(pae_matrix) for index in sublist]

      # Check that all indices from 0 to fittest_pop.shape[0] - 1 are in l
     assert all(i in indices for i in range(pae_matrix.shape[0])), "Not all indices are present in the list"


def test_swapt_sequences_equality():
  
   """Test swap_sequences function to ensure that the new sequences and their corresponding PLDDTs have the same length.

   Raises:
        AssertionError: If the length of the new sequences does not match the length of their respective PLDDT lists.
   """
  
   #Set the random seed for reproducibility in tests
   random.seed(42)
   seq1 = ['SMGGGGSLISAANNPSLKAAAPQ', 'AALRQVASLNG']
   plddt1 = [[88.4260816334587, 4.650702452297506, 22.439318656160356, 98.14087555247572, 95.54523051691886, 11.610893507247166, 73.7323068694926, 21.048009795821244, 27.517503749892967, 8.495394742415485, 19.086089467691934, 75.78823382952324, 82.39451869813249, 87.31708825818109, 20.00950041059464, 87.14546712226186, 86.48439035638037, 66.82728902459367, 46.389505774015596, 56.04638199141105, 77.10669772677534, 29.799339667694056, 55.358220221662336], [99.66853727614678, 33.94351840354507, 55.4277466101775, 52.71269198078736, 12.382873576996822, 84.79681883524619, 87.00542357974008, 71.58190777477301, 13.907540844441668, 50.1070753890187, 14.882769613351133]]
   seq2 = ['TLIFACENDSIAPVN', 'SRNAKQFLEINGGSHSC']
   plddt2 = [[97.50457636819638, 84.35880739042035, 6.930471437432084, 73.26718439102521, 46.71240163933591, 41.67043004433372, 6.43339614816536, 78.03594339619704, 85.01671988212762, 74.03338223685823, 45.41634257720367, 83.56576215423932, 83.80475283242154, 21.338009256195257, 48.71888031216176], [46.589120353117174, 21.130554879085352, 88.11509547400411, 38.394404435739474, 53.09997553123112, 42.14787320589808, 72.64628704484198, 95.07887286938445, 4.438352628642694, 37.42102063660093, 23.76871668861441, 15.557755696716825, 79.34056687857523, 37.65144962950996, 84.26386079346052, 44.006447172092656, 10.48553780832181]]

   new_seq1, new_plddt1, new_seq2, new_plddt2 = swap_sequences(seq1, plddt1, seq2, plddt2)
   assert len([aa for domain in new_seq1  for aa in domain]) == len([value for values in new_plddt1 for value in values])
   assert len([aa for domain in new_seq2  for aa in domain]) == len([value for values in new_plddt2 for value in values])
   
