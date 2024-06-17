# pyKinML: Package for training Neural Net Potential Energy Surfaces

## Description

This repository contains the code to train NNPESs and use those models with an ASE calculator.

### How to install

This package can be installed with pip or by cloning this repo and installing it locally.
First, make sure to have pytorch installed:

    https://pytorch.org/

This package relies on pytorch_scatter to sum the atomic contributions to energy. Ensure you have the proper version. Instructions for installing torch_scatter can be found at:

    https://github.com/rusty1s/pytorch_scatter


## Install with pip:
    pip install pykinml

### Clone from repo:
    git clone git@github.com:sandialabs/pykinml.git

We also highly recomend (required for force training) using the aevmod package for calculation of the aevs and their jacobians:
https://github.com/sandialabs/aevmod.git

For transition state optimization, we recomend Sella:
https://github.com/zadorlab/sella



This code is designed for the training and running of atomistic neural network potentials.
It pulls training data from sql files which should include energies, forces, and atomic corrdinates.
The sql files should be named in with the chemical formula of the molecules within them (e.g. C5H5.db, C2H4.db, etc.)
and the atomic coordinates and forces should be in the same sequence as thier cooresponding atom type
in the database name. The models use ANI style atomic environment vectors (AEVs). 
For details on AEV descriptors see DOI:https://doi.org/10.1039/C6SC05720A.
During training, the AEVs for each structure are computed prior to entering the training loop and saved.
This ensures that the AEVs only need to be computed a single time.
