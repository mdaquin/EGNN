# EGNN (Graph Neural Network)

EGNN is designed for the task of predicting molecular interaction energies based on graph-structured data. The model uses graph representations of chemical systems, where atoms are represented as nodes, and their interactions are represented as edges. This version of EGNN computes interaction colors and incorporates various atomic and distance-based features for better predictions. The data to train GNN model were taken from [1]. 


## Graph representations 

<div align="center">
<img src="https://github.com/mdaquin/EGNN/blob/main/EGNN_logo.png?raw=true">
</div>

## Model Overview

The model is designed to predict the interaction energy of a molecule using a graph-based representation, with the following key features:

**Node Features:**
- Atomic Properties (e.g., *atom*, *metal*, *fluoride*, *potassium*)
- Color Features (e.g., *colR*, *colG*, *colB* for red, green, and blue), which represent the electron occupation in t$_2$g *d* orbitals for this dataset
- Position Features (e.g., *dx*, *dy*, *dz* for  displacements in x/y/z directions)

**Edge Features:**
- Interaction Features (e.g., *distance*, *dx*, *dy*, *dz*)
- Interaction Color Features (e.g., *colIR*, *colIGreen*, *colIB*, *colIG*)

**Target:**
- Interaction Energy (*dE_scaled*): The target energy value is scaled for improved performance and accuracy during training.

## Installation 
Clone the project to your machine and install all requirements:
```bash
  git clone https://github.com/mdaquin/EGNN.git
  cd EGNN
  pip install -r requirements.txt
```
## How to run?  
```bash
  python3.10 main_gpu.py input_config.json
```
Here is *input_config.json* file where the setting are initialized. Please, change them with respect to your task. 

## Data Preparation 

- Normalisation: The node and edge features are normalized using min-max scaling to ensure that all features are on a similar scale for better training convergence. 

- De-normalisation: To compute loss we de-normalise data back. 

# Results: 

The model aims to predict the interaction energy of a molecular system by learning the complex relationships between atoms and their interactions. The performance can be visualized using plots of predicted vs. actual interaction energies for cases: 

- **Yes Interaction colour**, **No F atom**, **No K atom** ; 
- **No Interaction colours**, **No F atom**, **No K atom** ; 
- **Yes Interaction colour**, **Yes F atom**, **Yes K atom** ; 
- **No Interaction colours**, **Yes F atom**, **Yes K atom** ;


<div align="center">
<img src="https://github.com/mdaquin/EGNN/blob/main/results_mae.png?raw=true">
</div>

## Dataset description

The dataset is derived from the study by the Pascale et al. (2024)  [[1]](#1) and Pascale et al. (2023)[[3]](#3), which explores the orbital ordering (OO) patterns 
in KBF$_3$ perovskites, where B represents transition metals such as Sc, Ti, Fe, and Co. 
The research employs quantum mechanical methods, specifically a Gaussian-type basis set with the B3LYP hybrid functional with the CRYSTAL17 code, 
to analyze the partial occupancy of the t$_2$g *d* orbitals in these materials. 
By modeling a 40-atom supercell, the study identifies 162 distinct classes of equivalent OO configurations for each fluoroperovskite. 
The findings indicate that the energy differences among these configurations are minimal, ranging from 1 to 2 millielectronvolts per formula unit, suggesting that multiple configurations may coexist at room and even low temperatures. 
Additionally, a linear model of 10 parameters considering the relative orbital order in adjacent sites effectively reproduces the energy hierarchy (MAE of 92 $\mu$E$_h$ for this dataset) within the full set of configurations, implying its potential applicability for studying OO in larger supercells.

The dataset contains the following sheets:

    "geometry" Sheet outlines the relationship between the crystallographic structure of the perovskites and a set of reduced parameters used to describe it.

    "data_by_metal" Sheet provides:
        Optimized geometries,
        Energy values, and
        Orbital occupations (represented as color codes) for the 162 irreducible configurations of the t₂g electrons for each of the four metals (Sc, Ti, Fe, and Cr).

    "alldata" Sheet compiles the data from the other sheets and includes an additional quadrupolar component for each metal, resulting in a comprehensive and correlated dataset of 648 entries.

Detailed explanations of the dataset, its generation, and its implications can be found in the associated research articles and supplementary materials [[1]](#1) and [[3]](#3). 

The raw data from which the dataset is derived is accessible in the nomad repository [[2]](#2).

## References

<a id="1">[1]</a>
  t$_2$g *d* orbital ordering patterns in KBF3 (B = Sc,  Ti,  Fe,  Co) perovskites},
  Pascale Fabien, D'Arco Philippe, Mustapha Sami and Dovesi Roberto, 
  Journal of Computational Chemistry,
  2024, 45(24), 2048-2058,
  [doi: 10.1002/jcc.27391](https://dx.doi.org/10.1002/jcc.27391).

<a id="2">[2]</a>
  Solid solution K[Sc,Ti,Co,Fe]F3 dataset, with [DOI 10.17172/NOMAD/2024.04.16-1](https://dx.doi.org/10.17172/NOMAD/2024.04.16-1).

<a id="3">[3]</a>
  The $d$ Orbital Multi Pattern Occupancy in a Partially Filled $d$ Shell: The KFeF$_3$ Perovskite as a Test Case,
  Pascale Fabien, Mustapha Sami, D'Arco Philippe and Dovesio Roberto, 
  Materias, 2023, 16 (4), 1532,
  [doi: 10.3390/ma16041532](https://dx.doi.org/10.3390/ma16041532).
