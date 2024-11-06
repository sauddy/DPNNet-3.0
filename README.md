## DPNNet-3.0

This project is build a branched CNN model that can classify the number of hidden planets and predict the corresponding planet mass and position range for each of the planet from the protoplanetary disk images directly.

# Directory Strucure

DPNNet-RT.ipynb - Setup input parameters such as image resolution (choice between 64^2, 128^, 256^2, 512^2), transfer block (base model) or network, wether or not to tune hyperparameters, choice of non-tunable hyperparameters, and image sampling specifics.

plots.ipynb - Notebook containing all paper figure codes for reproducibilty and templates for checking on sources other than the 2 mentioned in the paper.

MODULES_DPNNeT - Support functions for DPNNet-RT.ipynb.

main.py - Script for a GUI based web tool for the models produced by DPNNET-RT.ipynb. Currently set to 256^2 image resolution model.

# Instructions to use the GUI web tool

```
streamlit run main.py
```


Please contact Subhrat Praharaj (s.praharaj@uva.nl) for the models used in the paper or the simulated image dataset of 105,000 images used for the training.
