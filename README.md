# Recreation of the paper "Learning Recommendations from User Actions in the Item-poor Insurance Domain"

This repository aims to recreate the findings from "Learning Recommendations from User Actions in the Item-poor Insurance Domain" by Simone Borg Bruun, Maria Maistro, and Christina Lioma. It contains adapted code from the [author's repository](https://github.com/simonebbruun/cross-sessions_RS). This repository is supplementary to our report (DOI: 10.5281/zenodo.14790070)

## Module versions

A requirements.txt file is included which can be used to setup the environment used to run the code.

## File Structure

### /cross_examine.py

contains scripts to to test all the RNN,random and popular models with different learning rate ([01,0.5]), seeds and learning rates are customisable. This code also generates the comparison visualisation for the RNN models in /visualizations folder.

### /datasets

contains all the dataset which were originally provided by the author

### /ablation

contains the files needed for the abaltion study. This includes ablation.py which creates the dataset subsets for the ablation categories. The other files are adaptations of the original learning and evaluation scripts from the paper that take in the adapted datasets and train and evaluate on them

### /models_code

directory containing the training and evaluation code of all models

SKNN.py: script that contains our custom version of the session based k nearest neighbour algorithm.

### /statistical_test

contains the custom script for McNemar and one-way ANOVA testing of the evaluation metrics as well as the output data of the evaluation functions needed for it.

### /visualizations

contains the scripts and data to recreate the visualizations that can be found in our report

### /weights

weights of the trained models
