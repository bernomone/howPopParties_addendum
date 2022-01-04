# howPopParties_addendum
This repository contains scripts and data related to the Corrigendum and Addendum of the "How populist are parties? measuring degrees of populism in party manifestos using supervised machine learning" published on Political Analysis (DOI: https://doi.org/10.1017/pan.2021.29).

This repository has been tested on Manjaro (21.0.2) and Ubuntu (20.0.4). Please contact the authors if you would like it to be tested, or to report that you tested it,  on other Operating Systems.

The PC configuration used for the repository testing is:

- CPU: Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz (8 cores)
- RAM: 32 GB DDR4
- Graphics: Intel UHD Graphics 630 (first graphic card) and GeForce GTX 1650 Mobile (second graphic card)
- Hard Disk: Intel SSDPEMKF010T8 NVMe 1024GB

Note that the most resource consuming part of the repository to run is the notebook (01_train_model.ipynb). We provided a sample of already trained models for users to be able to skip this part if they have lower resources.

The requirements for the repo to work are in env.yml (if you use conda: conda env create -f env.yml).


