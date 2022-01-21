import sys,os
import csv
import pickle
import scipy
import numpy as np
from IPython.display import clear_output
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
import sklearn.model_selection
import sklearn.feature_extraction
import sklearn.dummy
import itertools
import json
from train_configurations import *
import time

#####################################
parties_to_exclude = {
    "IT":['Forward Italy', 'PdL', 'Italy of Values', 'Casapound', 'Houses of Freedom'],

    "FR":['The Greens','French Communist Party', "Nouveau Parti Anticapitaliste", "Resistons",'Debout la France'],
    "AT":['Peter Pilz List'],

    "NL":['DENK','Party for the Animals','Reformed Political Party','50Plus','Green Left'],
    "ES":['Amaiur','Andalusian Party','Aragonist Council','Basque Country Unite'\
          ,'Basque Nationalist Party','Basque Solidarity','Canarian Coalition','Catalan Republican Left'\
          ,'Citizens','Commitment-Q','Commitment-We can-It is time','Democratic Convergence of Catalonia'\
          ,'Forum Asturias','Future Yes','Galician Nationalist Bloc','In Tide',"Navarrese People's Union",'Valencian style'],
    "DE":['Pirates']
}

populist_parties = {
    "IT":['Northern League', 'PaP', 'M5S', 'Brothers of Italy'],

    "FR":['National Front','Indomitable France'],
    "AT":['Austrian Freedom Party','Alliance for the Future of Austria','Team Stronach for Austria'],

    "NL":['Party of Freedom','List Pim Fortuyn','Socialist Party','Forum for Democracy'],
    "ES":['We can','In Common We Can',"Vox"],
    "DE":['The Left','Alternative for Germany']
    
}

numbers = [str(n) for n in range(1000)]

#####################################

"""

This script trains all the classifier models per each country. 

For each country it will perform a grid search over a set of parameters, depending on the model selected in the nations_params dictionary.

Launch this script using simply "python 01_train_all_models.py" in the main directory of this repository
to train iteratively all the models needed to reproduce the findings of this work. It is highly time consuming.

All the models' best meta-parameters, threshold and parameters will be saved into the models folder.
A recap of the trainings will be saved into the "training_results.json" file. 



"""

####################

"""
These are general parameters that are used in the main text.

- n_splits = the number of folds for the K-fold cross validation
- p_train = the fracion of data used for training and validation, must be in [0,1]
- n_jobs = number of cores to use during the grid-search

"""

n_splits = 5
p_train = 0.7
n_jobs = 40

#####################


'''
Modify the nations_params list if you want to change the target and model to train for each country.
The format of each record in the list should be

[
    {   "nation":string,
        "model":string,
        "target": string,
        "random_state":int
    }
]

Possible values:

- nation = the desired nation, possible values are {IT,FR,SP,GE,NE,AU, IT_speeches, IT_manual}
- model = the type of classifier to use {RandomForest, GradientBoosting, NeuralNetwork}
- target = select which score will be used to pick the best model in the grid search. Possible values are:
    - AUC = Area Under ROC 
    - Accuracy = classification accuracy
    - F1 = f1 score

'''

nations_params = [
    
     {
         "nation": "IT_speeches",
         "model":"GradientBoosting",
         "target": "AUC",
         "random_state":1
     },
     {
         "nation": "IT_manual",
         "model":"GradientBoosting",
         "target": "F1",
         "random_state":1
     },
     {
         "nation": "NL",
         "model":"GradientBoosting",
         "target": "AUC",
         "random_state":1
     },
     {   "nation": "IT",
         "model":"GradientBoosting",
         "target": "AUC",
         "random_state":1
     },
     {   "nation": "FR",
         "model":"GradientBoosting",
         "target": "AUC",
         "random_state":1
     },
     {   "nation": "ES",
         "model":"GradientBoosting",
         "target": "AUC",
         "random_state":1
     },
     {
         "nation": "DE",
         "model":"GradientBoosting",
         "target": "AUC",
         "random_state":1
     },
     {
         "nation": "AT",
         "model":"GradientBoosting",
         "target": "AUC",
         "random_state":1
     }
]



print("Starting training for all countries as indicated in the nations_params dictionary...")

for params in nations_params:
    
    nation = params["nation"]    
    random_state = params["random_state"]
    model_type = params["model"]
    target_score = params["target"]  
    
    ########################
    
    print("\nreading data for {0}..".format(nation))

    if nation in ["IT_speeches","IT_manual"]:
        data = json.load(open("./datasets/{}_sentences.json".format(nation),"r"))        
    else:
        data = json.load(open("./datasets/{}_manifesto_sentences.json".format(nation),"r"))

    texts = np.array([[w for w in record["clean_text"] if w not in numbers] for record in data])
    texts = np.array([" ".join(sent) for sent in texts])
    
    parties = np.array([record["party"] for record in data])
    years = np.array([record["party"] for record in data])
    orientations = np.array([record["orientation"] for record in data])
    indices = np.arange(0,len(texts)).astype(int)

    
    if nation =="IT_speeches":
        to_exclude = np.array([party in parties_to_exclude["IT"] for party in parties])
        Y = np.array([party in populist_parties["IT"] for party in parties])
    elif nation == "IT_manual":
        Y = np.array([record["is_populist"] for record in data])
        to_exclude = np.array([False for y in Y])
    else:
        to_exclude = np.array([party in parties_to_exclude[nation] for party in parties])
        Y = np.array([party in populist_parties[nation] for party in parties])        

    texts = texts[~to_exclude]
    parties = parties[~to_exclude]
    orientations = orientations[~to_exclude]
    indices = indices[~to_exclude]
    Y = Y[~to_exclude]


    texts_train,texts_test, Y_train, Y_test, indices_train, indices_test = sklearn.model_selection.train_test_split(texts,Y,indices,random_state=random_state, test_size=1-p_train)

    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    X_train = (vectorizer.fit_transform(texts_train)>0).astype(int)
    X_test = (vectorizer.transform(texts_test)>0).astype(int)  

    ########################
    
    print("training {0} for {1} with {2} as target score".format(nation,model_type,target_score))
    training_results = {
        "nation": nation,
        "model_type":model_type,
        "target_score":target_score,
        "random_state": random_state,
        "N_sentences": nation_stats[nation]["N_sentences"],
        "frac_sentences": nation_stats[nation]["frac_sentences"],
    }


    
    t_start = time.time()

    scoring = {'AUC': 'roc_auc', 'Accuracy': "accuracy", "F1":"f1"}


    cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if model_type == "RandomForest":
        cw = None
        if nation in ["IT_manual"]: cw = "balanced_subsample"
        model = RandomForestClassifier(random_state=random_state,class_weight = cw)
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=random_state)
    elif model_type == "NeuralNetwork":
        model = MLPClassifier(random_state=1)
    elif model_type == "Logistic":
        model = LogisticRegression(random_state=1,fit_intercept=False)    
    else:
        raise RuntimeError("Unspecified model. Select between RandomForest - GradientBoosting - NeuralNetwork - Logistic")


    # define search
    search = sklearn.model_selection.GridSearchCV(model, param_space[model_type], scoring=scoring, cv=cv, refit=target_score,n_jobs=n_jobs, verbose=3)
    result = search.fit(X_train, Y_train)
    best_model = result.best_estimator_



    # report progress
    best_index = search.cv_results_["params"].index(search.best_params_)
    n_splits = search.cv.n_splits
    for k in scoring:
        avg_score = [search.cv_results_['split{0}_test_{1}'.format(split,k)][best_index] for split in range(n_splits)]
        print("{0} Valid = {1} +/- {2}".format(k, np.mean(avg_score), np.sqrt(np.var(avg_score)/len(avg_score))))
        training_results[k] = np.mean(avg_score)
        training_results[k+"_err"] = np.sqrt(np.var(avg_score)/len(avg_score))


    print("best parameters:")
    print(search.best_params_)

    training_results["best_params"] = search.best_params_
    
    
    ########################
    
    print("computing threshold and performances on train set {0} for {1} with {2} as target score".format(nation,model_type,target_score))

    all_thresholds = []
    all_aurocs_train = []
    all_accuracies_train = []
    all_F1_train = []
    for train_index_batch, valid_index_batch in cv.split(X_train, Y_train):
        X_batch = X_train[valid_index_batch]
        Y_batch = Y_train[valid_index_batch]

        X_batch_train = X_train[train_index_batch]
        Y_batch_train = Y_train[train_index_batch]
        Y_batch_pred = best_model.predict_proba(X_batch_train)[:,1]

        ###################################

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_batch_train, Y_batch_pred,drop_intermediate=False)
        tnr = 1 - fpr
        fnr = 1 - tpr
        youdens = tpr/(tpr+fnr) + tnr/(tnr+fpr) - 1 
        max_threshold = thresholds[youdens.argmax()]
        all_thresholds.append(max_threshold)
        ###################################
        Y_batch_class= Y_batch_pred>max_threshold

        auroc_train = sklearn.metrics.roc_auc_score(Y_batch_train, Y_batch_pred)
        accuracy_train = sklearn.metrics.accuracy_score(Y_batch_train, Y_batch_class)
        F1_train = sklearn.metrics.f1_score(Y_batch_train, Y_batch_class)

        all_aurocs_train.append(auroc_train)
        all_accuracies_train.append(accuracy_train)
        all_F1_train.append(F1_train)




    print("Avg. AUC on train set = {0} +/- {1}".format(np.mean(all_aurocs_train), np.sqrt(np.var(all_aurocs_train)/len(all_aurocs_train))))
    print("Avg. Accuracy on train set = {0} +/- {1}".format(np.mean(all_accuracies_train), np.sqrt(np.var(all_accuracies_train)/len(all_accuracies_train))))
    print("Avg. F1 on train set = {0} +/- {1}".format(np.mean(all_F1_train), np.sqrt(np.var(all_F1_train)/len(all_F1_train))))    

    max_threshold = np.mean(all_thresholds)
    training_results["threshold"] = max_threshold


    training_results["AUC_train"] = np.mean(all_aurocs_train)
    training_results["F1_train"] = np.mean(all_accuracies_train)
    training_results["Accuracy_train"] = np.mean(all_F1_train)


    training_results["AUC_train_err"] = np.sqrt(np.var(all_aurocs_train)/len(all_aurocs_train))
    training_results["F1_train_err"] = np.sqrt(np.var(all_accuracies_train)/len(all_accuracies_train))
    training_results["Accuracy_train_err"] = np.sqrt(np.var(all_F1_train)/len(all_F1_train))
    

    ########################
    
    print("computing performances on validation set {0} for {1} with {2} as target score".format(nation,model_type,target_score))    
    # report progress
    best_index = search.cv_results_["params"].index(search.best_params_)
    n_splits = search.cv.n_splits
    for k in scoring:
        avg_score = [search.cv_results_['split{0}_test_{1}'.format(split,k)][best_index] for split in range(n_splits)]
        print("{0} on validation set = {1} +/- {2}".format(k, np.mean(avg_score), np.sqrt(np.var(avg_score)/len(avg_score))))

        training_results["{}_valid".format(k)] = np.mean(avg_score)
        training_results["{}_valid_err".format(k)] = np.sqrt(np.var(avg_score)/len(avg_score))

    
    ########################
    
    print("computing performances on test set {0} for {1} with {2} as target score".format(nation,model_type,target_score))

    Y_test_pred = best_model.predict_proba(X_test)[:,1]
    Y_test_classpred = best_model.predict(X_test)
    Y_test_class= Y_test_pred>max_threshold

    auroc_test = sklearn.metrics.roc_auc_score(Y_test, Y_test_pred)
    accuracy_test = sklearn.metrics.accuracy_score(Y_test, Y_test_class)
    F1_test = sklearn.metrics.f1_score(Y_test, Y_test_class)

    print("AUC on test set= ", auroc_test)
    print("Accuracy on test set = ", accuracy_test)
    print("F1 on test set = ", F1_test)
    training_results["AUC_test"] = auroc_test
    training_results["F1_test"] = F1_test
    training_results["Accuracy_test"] = accuracy_test

    ########################

    dummy_class = sklearn.dummy.DummyClassifier(strategy='uniform')
    dummy_class.fit(X_train, Y_train)
    Y_test_class = dummy_class.predict(X_test)

    accuracy_test = sklearn.metrics.accuracy_score(Y_test, Y_test_class)
    F1_test = sklearn.metrics.f1_score(Y_test, Y_test_class)

    print("Accuracy on test set (dummy classifier) = ", accuracy_test)
    print("F1 on test set (dummy classifier) = ", F1_test)

    training_results["F1_test_dummy"] = F1_test
    training_results["Accuracy_test_dummy"] = accuracy_test

    #########################
   
    print("saving on test set {0} for {1} with {2} as target score".format(nation,model_type,target_score))
    print("see 01_train_model.ipynb for a descrition of the outputs\n")
    
    params = result.best_params_.copy()
    params["threshold"] = max_threshold

    pickle.dump(params, open("./models/{0}_{1}_{2}_{3}_best_model_params.pkl".format(nation, model_type,target_score,random_state), "wb"))
    pickle.dump(best_model, open("./models/{0}_{1}_{2}_{3}_best_model.pkl".format(nation, model_type,target_score,random_state), "wb"))
    pickle.dump(indices_test, open("./models/{0}_{1}_{2}_{3}_test_indices.pkl".format(nation, model_type,target_score,random_state), "wb"))
    pickle.dump(search, open("./models/{0}_{1}_{2}_{3}_search.pkl".format(nation, model_type,target_score,random_state), "wb"))


    if not os.path.isfile("./training_results.json"):
        json.dump([training_results],open("./training_results.json", "w"))
    else:
        training_results_old = json.load(open("./training_results.json", "r"))

        found_flag = False
        for index, res in enumerate(training_results_old):
            if res["nation"] == training_results["nation"] and \
                res["model_type"] == training_results["model_type"] and \
                    res["target_score"] == training_results["target_score"] and \
                        res["random_state"] == training_results["random_state"]:
                            found_flag = True
                            break

        if found_flag:
            print("same configuration found!")
            print("deleting old results and overwriting..")
            del training_results_old[index]

        print("saving..")
        training_results_old.append(training_results)
        json.dump(training_results_old,open("./training_results.json", "w"))