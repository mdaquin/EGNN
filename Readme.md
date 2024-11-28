# EGNN (Edge Graph Neural Network)

EGNN is designed for the task of predicting molecular interaction energies based on graph-structured data. The model uses graph representations of chemical systems, where atoms are represented as nodes, and their interactions are represented as edges. This version of EGNN computes interaction colors and incorporates various atomic and distance-based features for better predictions.


## Features

- **Graph Representation** : 

## Model Overview

The model is designed to predict the interaction energy of a molecule using a graph-based representation, with the following key features:

**Node Features:**
- Atomic Properties (e.g., atom, metal, fluoride, potassium)
- Color Features (e.g., colR, colG, colB for red, green, and blue)
- Position Features (e.g., dx, dy, dz for position offsets)

**Edge Features:**
- Interaction Features (e.g., distance, dx, dy, dz)
- Interaction Color Features (e.g., colIR, colIGreen, colIB, colIG)

**Target:**
- Interaction Energy (dE_scaled): The target energy value is scaled for improved performance and accuracy during training.
