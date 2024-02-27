from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from Fitness import OptimizationFitness

@dataclass
class MatingPool:
    fittest_pop : pd.DataFrame

class selection(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def KeepFittest(self, score) -> pd.DataFrame:
        pass

class Select(selection):
    def __init__(self):
        super().__init__()
     #choosing the population having longer seq than the threshold    
    def KeepFittest(self, score : OptimizationFitness) -> MatingPool:
        #Creating the mating pool composed of 50% of the population having high general fit scores!
      
        parent = score.OptimizationFitness_df
        df = parent.sort_values(by = "FitnessScore", ascending = False)
        fittest_pop = pd.DataFrame()
        for i in range(len(df)):
            
            if len(df["sequence"][i]) >= seq_length:
                fittest_pop = pd.concat([fittest_pop, df.iloc[i:i+1]], ignore_index=True)
        fittest_pop = fittest_pop.head(int(0.5*len(df)))
        #test if the matingpool has the required size:


        #shuffling the population
        fittest_pop = fittest_pop.sample(frac = 1)
        fittest_pop = fittest_pop.reset_index(drop = True)
        
        return MatingPool(fittest_pop = fittest_pop)