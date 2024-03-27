# Atom representations based on attention (The README and project are currently being developed and may undergo further refinements)

The official paper can be found below:
* *[Atom representations based on attention]()* () [Will be added after publication]

## Overview

[TODO Decription]

## Requirements

1. Python >= 3.7
2. Numpy
3. Pandas
4. [pyTorch](https://pytorch.org/)
5. [PyTDC](https://gpytorch.ai/)
6. [PyG](https://pytorch-geometric.readthedocs.io/en/latest/)
7. [Neptune (Optional)](https://neptune.ai/)

### Installation

```
pip install numpy pandas torch PyTDC torch_geometric
```

## Running the code [TODO]

You can run 
```
python atom_representations_attention.py
```
Add --h to list all the possible arguments.

[TODO train.py]

The [train.py](./train.py) script performs the whole training, evaluation and final test procedure.

### Methods

This repository provides implementations of several attention pooling versions:
* `[version 3]` - [TODO]
* `[version 4]` - [TODO]

You must use those exact strings at training and test time when you call the script (see below). 

### Datasets

In our project, we used the QM9, ESOL, Solubility_AqSolDB, Human, and Rat datasets for training and testing. We trained the model on QM9 specifically to predict g298, and later applied transfer learning to other provided datasets, as QM9 contained the largest amount of data. When using ESOL and Solubility datasets, our focus was on solubility, while Human and Rat datasets were used for half-life predictions. We recommend utilizing the GPU mode for improved performance, especially when operating on the QM9 dataset.

This is an example of how to download and prepare a dataset for training/testing.

[TODO Script to download datasets and GPU version]

These are the instructions to train and test the methods reported in the paper in the various conditions.

[TODO]

### Neptune [Optional TODO]

We provide logging the training / validation metrics and details to [Neptune](https://neptune.ai/). To do so, one must export the following env variables before running `train.py`.

```bash
export NEPTUNE_PROJECT=...
export NEPTUNE_API_TOKEN=...
```


Acknowledgements
---------------
In our work, we compare the results of models trained using our attention pooling layers to models that employ feature representation "1" and "10", which yielded the best performance in the study "Comparison of Atom Representations in Graph Neural Networks for Molecular Property Prediction" (https://arxiv.org/abs/2012.04444).

## Bibtex citations

If you find our work useful, please consider citing it:

[TODO Our Citations]