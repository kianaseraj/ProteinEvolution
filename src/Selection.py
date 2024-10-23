from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from Fitness import OptimizationFitness


"""
Fourth stage of the algorithm.
The fitness score of the population from Fitness module will be evaluated and the population having highest scores will be chosen as the parent population.
"""
@dataclass
class MatingPool: #The final population selected based on highest scores for mating.
    fittest_pop : pd.DataFrame

class selection(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def KeepFittest(self, score, length_threshold) -> pd.DataFrame:
        pass

class Select(selection):
    def __init__(self):
        super().__init__()
      
    def KeepFittest(self, score:OptimizationFitness, length_threshold:int) -> MatingPool:
        #Creating the mating pool composed of 50% of the population having high general fit scores!
      
        parent = score.OptimizationFitness_df
        df = parent.sort_values(by = "FitnessScore", ascending = False)
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
