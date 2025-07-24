# Results and code

This repository contains the results and code from the PhD thesis **Extraction of Quantitative Grammatical Rules from Syntactic Treebanks**.


## Results

All results are in the ```results``` directory. The JSON files contain the scope, the conclusion and the extracted rules for each experience of the thesis.

## Code

The main code for rule extraction is in ```src```. Within this directory:

- ```grex2```: the rule extraction scripts, including those for the decision tree, sparse logistic regression, and the RuleFit implementation.

The work is built upon a modified version of Grex2. The original Grex2 project is maintained at https://github.com/FilippoC/grex2. The code was primarily developed by Caio Corro.

The official documentation for Grex2 (currently under development) can be found at https://grex.grew.fr.

If you use this software, please cite our paper:

    @inproceedings{herrera2024grex,
        title = "Sparse Logistic Regression with High-order Features for Automatic Grammar Rule Extraction from Treebanks",
        author = "Herrera, Santiago and Corro, Caio and Kahane, Sylvain",
        booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
        month = may,
        year = "2024",
        address = "Torino, Italia",
        url = "https://arxiv.org/abs/2403.17534",
    }



- ```univariate```: script for compute univariate measures over features.
- ```evaluation```: evaluations and global measure scripts.

## Experiments

This directory includes bash scripts for executing the tasks of each experiment. It also contains other scripts or notebooks specific to each experiment.

## Setup

Clone the repository and then install the project with ```pip install -e .```

I recommend installing the project in a virtual environment, e.g., ```python -m venv .venv```

Some paths need to be adjusted in order to run the experiments.

