# BiomechPriorVAE
A biomechanics prior based VAE project

## Overview
The project focuses on generate more reasonable human lower body posture with pretrained VAE model based on open-source musculoskeletal motion dataset, as a biomechanics prior, provides guidance for optimal control for gait generation

## Structure
### 1.scripts
Contains the main python scripts for data convertion and model training, as well as the interface for Matlab use

- **`b3dconverter.py`** - Convert the .b3d files into python-friendly numpy array.
- **`datasetvisualize.py`** - Visualize the musculoskeletal model with specific posture using nimblephysics.
- **`vaetrainer.py`** - VAE model trainer.
- **`vaemodel.py`** - Interface for using VAE model in Matlab

### 2.result
Contains the VAE model trained using PyTorch and saved StandardScaler