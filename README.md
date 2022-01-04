# howPopParties_addendum

This repository contains scripts and data related to the Corrigendum and Addendum of the "How populist are parties? measuring degrees of populism in party manifestos using supervised machine learning" published on Political Analysis (DOI: https://doi.org/10.1017/pan.2021.29).

The original version of this repository is:

- Di Cocco, Jessica; Monechi, Bernardo, 2021, "Replication Material for "How Populist Are Parties? Measuring Degrees of Populism in Party Manifestos Using Supervised Machine Learning"", https://doi.org/10.7910/DVN/BMJYAN, Harvard Dataverse, V1

This repository has been tested on Manjaro (21.0.2) and Ubuntu (20.0.4). Please contact the authors if you would like it to be tested, or to report that you tested it,  on other Operating Systems.

The PC configuration used for the repository testing is:

- CPU: Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz (8 cores)
- RAM: 32 GB DDR4
- Graphics: Intel UHD Graphics 630 (first graphic card) and GeForce GTX 1650 Mobile (second graphic card)
- Hard Disk: Intel SSDPEMKF010T8 NVMe 1024GB

The requirements for the repo to work are in env.yml (if you use conda: conda env create -f env.yml).

# Raw Data

The raw data of political manifestos and leaders' speeches can be found in the "datasets" directory. All the files are in .json format and each record of the json represents a sentence and contains the following fields:

- year: the year of the manifesto or the speech the sentence comes from
- party: the party the manifesto or the speaker belongs to
- leader: the leader of the party in that specific year (can be "null" if missing)
- text: the raw text of the sentence
- cleaned_text: a list of stemmed words obtained from the raw text

In the original version, party and year labels for Austria (AT) were randomly assigned to manifestos, while the German data (DE) had a repeated manifesto. Here you can find the correct version.

