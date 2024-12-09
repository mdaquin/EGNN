# EGNN (Graph Neural Network)

EGNN is designed for the task of predicting molecular interaction energies based on graph-structured data. The model uses graph representations of chemical systems, where atoms are represented as nodes, and their interactions are represented as edges. This version of EGNN computes interaction colors and incorporates various atomic and distance-based features for better predictions.


## Graph representations 

<div align="center">
<img src="https://github.com/mdaquin/EGNN/blob/main/EGNN_logo.png?raw=true">
</div>

## Model Overview

The model is designed to predict the interaction energy of a molecule using a graph-based representation, with the following key features:

**Node Features:**
- Atomic Properties (e.g., *atom*, *metal*, *fluoride*, *potassium*)
- Color Features (e.g., *colR*, *colG*, *colB* for red, green, and blue)
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
Here is *input_config.json* file where the setting are initialized. Please, chnage them with respect to your task. 

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


