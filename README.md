#**In Silico Protein Evolution with Hallucination**


There has been enumuros progress in protein structure prediction using language models. Here one of
the SOTA models, ESM, has been used to examine if the features captured by it can be used in generating new sequences. Starting
with the population of randomly generated protein sequences, are then given to the ESM model for structure
prediction and as expected they will be featureless. with one of the evolution algorithms, Genetic Algorithm it has been
tried to optimize the population iteratively to the desired structural constraints. 
Deep learning methods by borrowing structural predictors they can generate new protein sequecnes with
their backbones without considering their sequence or generating new sequences withou considering their backbones.
However, none of them generatted both sequences and backbones simultaneously.
for each sequence a genetitc algorith has been carries out and each step consists of applying crossover and mutation op-
eratioons on the sequences and calculating their disances and evaluaing their strucural prediction
confidence as one of the fitness criteria.


