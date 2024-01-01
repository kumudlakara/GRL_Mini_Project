# GRL_Mini_Project
This repository contains the code base for the Graph Representation Learning (GRL) code base for a mini project as a part of the GRL course at the University of Oxford.

## Experiments
Multiple experiments in line with the report can be run using this codebase. In order to run the experiments, the virtual environment must first be activated using the `pipenv shell` command. The following experiments are available:
1. DropGNN vs RNI comparison: This is the first experiment presented in the report and is performed by first running probability vs accuracy experiments for the DropGNN model and by running the RNI model with `prob_rni` augmentation:

`python experiment.py --augmentation dropout --prob_experiment --dataset <dataset_name>`
`python experiment.py --augmentation prob_rni --rni_experiment --dataset <dataset_name>`

2. Restricted RNI (rRNI): The rRNI experiment where expressive power or model accuracy is compared for different random values can be run by using the following command:

`python experiments.py --augmentation rrni --rrni_experiment --dataset <dataset_name>`

3. RNI vs rRNI comparison: This experiment can be performed by first running the RNI model on multiple datasets using the following command:

`python experiment.py --augmentation prob_rni --dataset <dataset_name>`

Then run the following command for obtaining results for rRNI:

`python experiment.py --augmentation rrni --dataset <dataset_name>`

## Models
There are two models:

1. DropGNN: This is the GIN model which has dropout incorporated into its forward pass.
2. GNN Model: This is the vanilla GNN model which is a GIN model with ε=0.

## Datasets
The datasets used are:

1. TRIANGLES: Nodes have to predict whether they are part of a triangle 
2. LCC: Nodes have to predict their local clustering coefficient
3. LIMITS 1 and LIMITS 2: Compare two smaller structures versus one larger structure. We 4. 
4. 4-CYCLES:classify graphs on containing a cycle of length 4. The regularity in this dataset is increased by ensuring that each node has a degree of 2. Finally we experiment on circular 5. SKIP-CIRCLES: Circular graphs with skip links where the model needs to classify if a given circular graph has skip links of length {2,3,4,5,6,9,11,12,13,16}.

Note: all the datasets and their authors are referenced in the report.

