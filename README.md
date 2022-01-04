# howPopParties_addendum

This repository contains scripts and data related to the Corrigendum and Addendum of the *"How populist are parties? measuring degrees of populism in party manifestos using supervised machine learning"* published on *Political Analysis* (DOI: https://doi.org/10.1017/pan.2021.29).

The original version of this repository is:

- Di Cocco, Jessica; Monechi, Bernardo, 2021, "Replication Material for "How Populist Are Parties? Measuring Degrees of Populism in Party Manifestos Using Supervised Machine Learning"", https://doi.org/10.7910/DVN/BMJYAN, Harvard Dataverse, V1

This repository has been tested on Manjaro (21.0.2) and Ubuntu (20.0.4). Please contact the authors if you would like it to be tested, or to report that you tested it,  on other Operating Systems.

The PC configuration used for the repository testing is:

- CPU: Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz (8 cores)
- RAM: 32 GB DDR4
- Graphics: Intel UHD Graphics 630 (first graphic card) and GeForce GTX 1650 Mobile (second graphic card)
- Hard Disk: Intel SSDPEMKF010T8 NVMe 1024GB

The requirements for the repo to work are in env.yml. If you use Anaconda then *conda env create -f env.yml*, otherwise just install the listed dependencies via pip.

# Raw Data

The raw data of political manifestos and leaders' speeches can be found in the **/datasets** directory. All the files are in .json format and each record of the json represents a sentence and contains the following fields:

- year: the year of the manifesto or the speech the sentence comes from
- party: the party the manifesto or the speaker belongs to
- leader: the leader of the party in that specific year (can be "null" if missing)
- text: the raw text of the sentence
- cleaned_text: a list of stemmed words obtained from the raw text

In the original version, party and year labels for Austria (AT) were randomly assigned to manifestos, while the German data (DE) had a repeated manifesto. Here you can find the correct version.

# Pre-computed Populist Scores

Pre-computed Populist scores are stored in the **/scores** folder:

- global_scores_{nation}.csv files contain the scores computed using each manifesto for each party
- scores_in_time_{nation}.csv files contain the scores computed by dividing manifestos by year

# How to use it?

To compute scores from scratch:

1. Run the *00_generate_bag_of_words.ipynb * notebook to preprocess the data in **/dataset** creating bag-of-words and labels.
2. Run *python 01_train_all_models.py* and *python 01_train_all_models_resh.py* to train the classifiers on the dataset and the reshuffled dataset

Since point *2* is quite time consuming, there are pre-computed classifier models in the **/models** and **/models_resh** directories. Skip point *2* if you are willing to use them.

3. Run the *02_compute_scores.ipynb* and *02_compute_scores_reshuffled.ipynb* to compute the scores in the normal and reshuffled cases. This will overwrite the **/scores** and **/scores_resh** folders.

4. Run *03_shap_values.ipynb* and *04_reshuffling_effect.ipynb* to reproduce the results.

# Acknowledgements

We would like to acknowledge *Michael Jankowski* (Institute for Social Sciences, University of Bremen) and *Robert A. Huber* (Department of Political Science, University of Salzburg) that found the issues with the Austrian and German datasets.

Cheers!