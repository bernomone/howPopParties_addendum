"""
This script contains the parameters for the grid search used in the paper.
"""

param_space = dict()
param_space["RandomForest"] = {}
param_space["RandomForest"]['n_estimators'] = [400]

param_space["RandomForest"]['criterion'] = ["entropy"]
param_space["RandomForest"]['max_features'] = ["log2"]
param_space["RandomForest"]['min_samples_split'] = [10]
param_space["RandomForest"]['min_samples_leaf'] = [1]

param_space["GradientBoosting"] = {}

param_space["GradientBoosting"]['learning_rate'] = [0.1,0.2, 0.5]
param_space["GradientBoosting"]['n_estimators'] = [400]
param_space["GradientBoosting"]['criterion'] = ["friedman_mse"]
param_space["GradientBoosting"]['max_features'] = ["log2"]
param_space["GradientBoosting"]['min_samples_split'] = [2,5]
param_space["GradientBoosting"]['min_samples_leaf'] = [1,2]

param_space["NeuralNetwork"] = {}
param_space["NeuralNetwork"]['activation'] = ["relu", "logit"]
param_space["NeuralNetwork"]['hidden_layer_sizes'] = [(50,1),(50,2),(100,1),(200,1)]

param_space["Logistic"] = {}
param_space["Logistic"]['penalty'] = ["l2"]
# param_space["Logistic"]['C'] = [1,1.5,2]


nation_stats = {
    "AT":{
        "N_sentences":14156,
        "frac_sentences":0.24
    },
    "AT_new":{
        "N_sentences":14156,
        "frac_sentences":0.24
    },
    "FR":{
        "N_sentences":12599,
        "frac_sentences":0.27
    },
    "DE":{
        "N_sentences":30399,
        "frac_sentences":0.17
    },
    "IT":{
        "N_sentences":13004,
        "frac_sentences":0.20
    },
    "IT_resh":{
        "N_sentences":13004,
        "frac_sentences":0.20
    },
    "NL":{
        "N_sentences":77504,
        "frac_sentences":0.16
    },
    "ES":{
        "N_sentences":95997,
        "frac_sentences":0.20
    },
    "IT_speeches":{
        "N_sentences":13004,
        "frac_sentences":0.20
    },
    "IT_manual":{
        "N_sentences":13004,
        "frac_sentences":0.20
    },

    "AT_new_resh":{
        "N_sentences":14156,
        "frac_sentences":0.24
    },
    "FR_resh":{
        "N_sentences":12599,
        "frac_sentences":0.27
    },
    "DE_resh":{
        "N_sentences":30399,
        "frac_sentences":0.17
    },
    "IT_resh":{
        "N_sentences":13004,
        "frac_sentences":0.20
    },
    "NL_resh":{
        "N_sentences":77504,
        "frac_sentences":0.16
    },
    "ES_resh":{
        "N_sentences":95997,
        "frac_sentences":0.20
    },
}