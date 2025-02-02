
[![DOI](https://zenodo.org/badge/758700892.svg)](https://doi.org/10.5281/zenodo.14359834)

# Textual complexity
This repository contains the code base accompanying the paper '__Model interpretability enhances domain
generalization in the case of textual complexity
modeling__'. 
It consists of two main parts: Python code for FNN (Feedforward Neural Network) models and r code for GLM (General Linear Model) models.

## R code
R-code for modelling is included in the directory __r-code__. 
* __r-code/models_glmnet_expanded_cv.R__ contains the main model training loop. Loads feature sets (text-ids, features, labels) from _data/feature_sets_ after which it spans different model configurations. Each configuration is evaluated using cross-validation to measure the resultant model's effectiveness. After cross-validation, each configuration is re-trained to the full training data set and evaluated on the target data. Logs are kept in _logs/glmnet_cv/_.    

## Python code
The Python code contains three main loops:
* __python/scripts/prepare_combis.py__. Calculates and ranks 2nd and 3rd-order feature interactions on F-scores. These calculations are ran on a GPU given the large number of possible combinations. The rankings are subsequently used to add feature interaction depth to GLM models.
* __python/notebooks/Bert-Features-Text.ipynb__ Applies a sliding window to texts to extract BERT's embedding features. Stores these features into the BERT feature set (_bert_base_uncased-features_embedding-layer-10_11.parquet_). 
* __python/notebooks/pipeline_loop3.ipynb__. The main FNN model training loop. Loads feature sets (text-ids, features, labels) from _data/feature_sets_ after which it spans a nested resampling. The outer loop (train-test split) varies models on depth and width (number of parameters). The inner loop (cross validation) varies models on their hyperparameters. Each configuration is tested using cross-validation to measure the resultant model's effectiveness. After tuning and testing, each model is retrained on the full training data set and evaluated on the target data. Logs are kept in _logs/torch_cv/_. 

## Scala code
The Scala code contains all the code used to extract the engineered feature set from the corpora. It has a dedicated readme.md file outlining details.

## Models and results
Data files are released through Figshare (doi: [10.6084/m9.figshare.25676394](https://doi.org/10.6084/m9.figshare.25676394)).

## Reference
Van der Sluis, F., & van den Broek, E. L. (2025). Model interpretability enhances domain generalization in the case of textual complexity modeling. *Patterns, 6*, 101177. https://doi.org/10.1016/j.patter.2025.101177
