from Individual import GeneratingPopulation
from Folding import Predictor
from Fitness import filament_score, maximizedglobularity_score, folding_score, total_fitness_score
from Selection import Select
from Breeding import DomainCrossover
from Mutation import MaskedMutation
from tqdm import tqdm
import argparse
import logging


def create_parser():
    """
    Creates an argument parser for configuring the evolutionary scenario.

    Returns:
    - argparse.ArgumentParser: A parser to handle command-line arguments.
    """

    parser = argparse.ArgumentParser(description = "Generating protein population")
    
    parser.add_argument("num_steps", 
                        type = int, 
                        help = "Choosing the number of iteration"
                        )
    parser.add_argument("length", 
                        type = int, 
                        help = "setting the length of seed population"
                        )
    parser.add_argument("num_sequence", 
                        type = int, 
                        help = "setting the size of the seed population"
                        )
    parser.add_argument("chain_num", 
                        type = int, 
                        help = "setting the number of sequence chains"
                        )
    parser.add_argument("dir_path", 
                        type = str, 
                        help = "choosing a path for loading the model"
                        )
    parser.add_argument("population_path", 
                        type = str, 
                        help = "choosing a path for saving the generated population"
                        )
    return parser
    
    
class EvolutionScenario:
    
    """
    Simulates the evolutionary process for generating protein sequences with improved properties.
    """

    def __init__(self, chain_num: int, num_sequence: int, length: int , dir_path: str, population_path: str ):
        
        """simulates the protein evolution.
        Arguments:
          - chain_num : number of chains in a protein.
          - num_sequence : number of protein sequences in a population.
          - length : length of a protein sequence.
          - dir_path : path to load the model.
          - population_path : path to save the output population.
        """
        
        logging.basicConfig(filename = f"{population_path}", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        #stage1: initializing the input population with randomely generated sequences.
        self.proteins = GeneratingPopulation(chain_num, num_sequence, length)
        # Load models for folding and mutation
        self.folding_model = Predictor.ESMFold.load(dir_path)
        self.mut_model, self.mut_alphabet = Predictor.ESMmodel.load(dir_path)
        self.length = length
        self.chain_num = chain_num

    def evolve(self, num_steps : int):
      
        """
        Arguments:
          - num_steps : number of iterations.
        Returns: Hallucinated protein sequences with their structural confidence scores
        """      
        assert self.proteins is not None
        
        
        population = GeneratingPopulation(2, 20, 150)
        for j in tqdm(range(1, num_steps + 1), desc="Generating population", unit="population"):
            
            #stage2 : Measuring folding scores of the sequences with Esmfold
            FoldResult = self.folding_model.fold(population)

            #stage3 : Calculating the structural confidence score of sequences.
            StructureScore = folding_score(FoldResult)
            GlobularityScore = maximizedglobularity_score(FoldResult)
            FilamentDimer = filament_score(FoldResult)
            
            if self.chain_num >= 2:  #for more than one chian of sequence, filament score is also considered.
              OptimizationFitness = total_fitness_score(StructureScore, GlobularityScore, FilamentDimer)
            else:
              OptimizationFitness = total_fitness_score(StructureScore, GlobularityScore)

              
            #stage4 : Selection operation; sequences having high general structural scores will be selected.
            selecting = Select()
            MatingPool = selecting.KeepFittest(OptimizationFitness)

            #stage5 : domain-based breeding; The crossover happens between domains of each sequence pair.
            breed = DomainCrossover()
            Children = breed.crossover(MatingPool)

            #stage6 : Mutation with the masked prediction task; 3% of the amino acids of a sequence having the lowest plddt values will be covered and substituted with residues pedicted by ESM model to have high logit values
            mutating = MaskedMutation(self.mut_model, self.mut_alphabet, self.length)
            MutatedChildren = mutating.mutate(Children)
            population = MutatedChildren #The final population, and the input for the subsequent iteration
            #Logging 10 sequences of the fittest population having the highest mean plddt scores
            sorted_df = (MatingPool.fittest_pop.sort_values(by='mean_plddt', ascending=False)).head(10)
            iter_data = { "step" : j, "sequence" : sorted_df["sequences"].values, "mean_plddt" : sorted_df["mean_plddt"].values, "iteration_mean" :  df["mean_plddt"].mean(), "var" : df["mean_plddt"].var()}
            logging.info(iter_data)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    evolution_scenario = EvolutionScenario(args.chain_num, args.num_sequence, args.length, args.dir_path, args.population_path)
    evolution_scenario.evolve(args.num_steps)





        






    
