# Textual complexity
This repository contains the code base accompanying the paper 'Interpretability benefits Generalisability but not
Accuracy for Models of Textual Complexity'. 
It consists of two main parts: Python code for FNN (Feedforward Neural Network) models and r code for GLM (General Linear Model) models.

## R code
R-code for modelling is included in the directory __r-code__. 
* __r-code/models_glmnet_expanded_cv.R__ contains the main model training loop. Loads feature sets (text-ids, features, labels) from _data/feature_sets_ after which it spans different model configurations. Each configuration is evaluated using cross-validation to measure the resultant model's effectiveness. After cross-validation, each configuration is re-trained to the full training data set and evaluated on the target data. Logs are kept in _logs/ _logs/glmnet_cv/_.    

## Python code
The Python code contains three main loops:
* __python/scripts/prepare_combis.py__. Calculates and ranks 2nd and 3rd-order feature interactions on F-scores. These calculations are ran on a GPU given the large number of possible combinations. The rankings are subsequently used to add feature interaction depth to GLM models.
* __python/notebooks/Bert-Features-Text.ipynb__ Applies a sliding window to texts to extract BERT's embedding features. Stores these features into the BERT feature set (_bert_base_uncased-features_embedding-layer-10_11.parquet_). 
* __python/notebooks/pipeline_loop3.ipynb__. The main FNN model training loop. Loads feature sets (text-ids, features, labels) from _data/feature_sets_ after which it spans a nested resampling. The outer loop (train-test split) varies models on depth and width (number of parameters). The inner loop (cross validation) varies models on their hyperparameters. Each configuration is tested using cross-validation to measure the resultant model's effectiveness. After tuning and testing, each model is retrained on the full training data set and evaluated on the target data. Logs are kept in _logs/ _logs/torch_cv/_. 

## Data
Data are stored in the _data_ subfolder. Currently, only the main results table is included in __data/results.csv__. This file contains the results for each model configuration ran. It details the following columns: 
* _data_file_ the feature set used as input for the model
* _*epoch_ specifies the number of training epochs ran in resp. the outer and inner loop.
* _net*_ specifies parameters (ranges) for each of the models. 
* _tuned*_ specifies the parameters set after tuning parameter ranges.
* _classif*_ lists classification metrics on task 1 (text classification).
* _guardian*_ gives correlation metrics for task 2 (predicting appraisals of processing difficulty). Rho gives rank order correlation, rprob Pearson's correlation with the predicted class probability for the 'complex' class, and rlogit gives Pearson's correlation with the logit-value of the prediction.
* _parameters_ and _parameters.rounded_ give a calculation of the number of model coefficients or parameters.
* Remaining parameters specify names and categorizations for the models.
