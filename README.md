# Results and code

This repository contains the results and code from the PhD thesis **Extraction of Quantitative Grammatical Rules from Syntactic Treebanks**.


## Results

All results are in the ```results``` directory. The JSON files contain the scope, the conclusion and the extracted rules for each experience of the thesis.

## Code

The main code for rule extraction is in ```src```. Within this directory:

- ```grex2```: the rule extraction scripts, including those for the decision tree, sparse logistic regression, and the RuleFit implementation. 
- ```univariate```: script for compute univariate measures over features.
- ```evaluation```: evaluations and global measure scripts.

## Experiments

This directory includes bash scripts for executing the tasks of each experiment. It also contains other scripts or notebooks specific to each experiment.

## Setup

Clone the repository and then install the project with ```pip install -e .```

I recommend installing the project in a virtual environment, e.g., ```python -m venv .venv```

Some paths need to be adjusted in order to run the experiments.

