# Physics-Informed Deep Learning for Nonlinear Friction Model of Bow-String Interaction (DAFx25)

This repository contains the implementation of the methods described in our paper on physics-informed deep learning for nonlinear bow-string interaction modeling.

## Quick Start

- **Main Scripts**  
  - Training and testing PINNs: `main/bowmass_pinn.py`  
  - Training and testing PI-DeepONets: `main/bowmass_deepOnet.py`  

- **Pre-trained Models**  
  - Available in the `trained_model` directory.

- **FDM Simulation Data**  
  - Located in the `data` directory.

- **Figure Reproduction Scripts**  
  - Scripts to reproduce the figures in the paper can be found in the `export_result` directory.

- **Intermediate Saved Results**  
  - Saved intermediate results used to generate the figures are in the `saved_data` directory.

- **Dependencies**  
  - Please install [PyHessian](https://github.com/amirgholami/PyHessian) for computing the Hessian matrix.

- **SOAP Optimizer**  
  - The SOAP optimizer implementation in `utils/soap.py` is sourced from [this repository](https://github.com/nikhilvyas/SOAP/tree/main).

## Sound Samples

- `.wav` files are located in the `audio` directory.



