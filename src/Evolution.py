from Individual import RandomPopulation
from Folding import Predictor
from Fitness import Fitness
from Selection import Select
from Breeding import DomainCrossover
from Mutation import MaskedMutation
from tqdm import tqdm
import argparse
import logging





def create_parser():
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

  
    def __init__(self, chain_num: int, num_sequence: int, length: int , dir_path: str, population_path: str ):
        
        logging.basicConfig(filename = f"{population_path}", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        """
        initilizing the population with randomely generated sequence space
        """
        #stage1: initializing the input population which has beed randomely generated.
        self.proteins = RandomPopulation.GeneratingPopulation(chain_num, num_sequence, length )
        self.folding_model = Predictor.ESMFold.load(dir_path)
        self.mut_model, self.mut_alphabet = Predictor.ESMmodel.load(dir_path)
        self.length = length
        self.chain_num = chain_num

    def evolve(self, num_steps : int):
        """
        using the defined modules to optimize the protein population
        """
        assert self.proteins is not None
        
        
        population = self.proteins  
        for j in tqdm(range(1, num_steps + 1), desc="Generating population", unit="population"):
            
            #stage2 : Measuring folding scores for the sequences
            FoldResult = self.folding_model.fold(population)

            #stage3 : Calculating structure fitness score 
            score1 = Fitness.FoldingFitness()
            StructureScore = score1.Score(FoldResult)

            score2 = Fitness.MaximizedGlobularity()
            GlobularityScore = score2.Score(FoldResult)

            fitness = Fitness.TotalFitness()
          
            if self.chain_num >= 2:
              score3 = Fitness.Filament()
              FilamentDimer = score3.Score(FoldResult)
              OptimizationFitness = fitness.FitnessScore(StructureScore, GlobularityScore, FilamentDimer)

            else:
              OptimizationFitness = fitness.FitnessScore(StructureScore, GlobularityScore)

              
            #stage4 : selection based on the highest general structural scores
            selecting = Select()
            MatingPool = selecting.KeepFittest(OptimizationFitness)

            #stage5 : domain-based breeding
            breed = DomainCrossover()
            Children = breed.crossover(MatingPool)

            #stage5 : Mutation with masked prediction task
            mutating = MaskedMutation(self.mut_model, self.mut_alphabet, self.length)
            MutatedChildren = mutating.mutate(Children)
            population = MutatedChildren
            # 
            df = MatingPool.fittest_pop
            sorted_df = (df.sort_values(by='mean_plddt', ascending=False)).head(10)
            iter_data = { "step" : j, "seq" : sorted_df["sequences"].values, "mean_plddt" : sorted_df["mean_plddt"].values, "iteration_mean" :  df["mean_plddt"].mean(), "var" : df["mean_plddt"].var()}
            logging.info(iter_data)



if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    evolution_scenario = EvolutionScenario(args.chain_num, args.num_sequence, args.length, args.dir_path, args.population_path)
    evolution_scenario.evolve(args.num_steps)





        






    
