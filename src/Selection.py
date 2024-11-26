
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass



"""
Fourth stage of the algorithm.
The fitness score of the population from Fitness module will be evaluated and the population having highest scores will be chosen as the parent population.
"""
@dataclass
class MatingPool:
    """
    Represents the final selected population for mating based on fitness scores.
    """
    fittest_pop : pd.DataFrame #The final population selected based on highest scores for mating.

class selection(ABC):
    """
    Abstract base class defining the interface for the selection process.

    This class serves as a blueprint for implementing specific selection strategies
    to choose the fittest individuals from a population.
    """
    def __init__(self):
        """Initializes the Selection class.
    
        This constructor is empty as this class serves as an interface
        for implementing concrete selection strategies.
        """
        pass

    @abstractmethod
    def KeepFittest(self, score, length_threshold) -> pd.DataFrame:
        """
        Selects the fittest individuals from the population based on their fitness scores.
        """
        pass

class Select(selection):
    """Implements the Selection abstract base class using a specific strategy.

    This strategy involves filtering sequences based on a length threshold, 
    sorting by fitness scores, and selecting the top-performing individuals 
    to form the mating pool.
    """
    def __init__(self):
                      
        """Initializes the Select class.

        This constructor calls the base class constructor to ensure proper
        initialization of the abstract base class.
        """
        super().__init__()
      
    def KeepFittest(self, score, length_threshold:int) -> MatingPool:
        """Creating the mating pool composed of 50% of the population having high general fit scores!
        Arguments: 
            - score: The overall fitness score of each potein sequence obtained by Fitness module
            -length_threshold: A threshold value for choosing sequences as a parent population, if they are longer than a thershold value.
        Returns: A dataframe of sequence population as the parent population 
        """
      
        df = score.sort_values(by = "FitnessScore", ascending = False)
        fittest_pop = pd.DataFrame()
        #Condititoning the method to only select the sequences longer than a threshold
        for i in range(len(df)):
            if len(df["sequence"][i]) >= length_threshold:
                fittest_pop = pd.concat([fittest_pop, df.iloc[i:i+1]], ignore_index=True)
        fittest_pop = fittest_pop.head(int(0.5*len(df)))
    
        #shuffling the population
        fittest_pop = fittest_pop.sample(frac = 1)
        fittest_pop = fittest_pop.reset_index(drop = True)
        
        return MatingPool(fittest_pop = fittest_pop)

        