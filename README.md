# MechInterpCode
##### MechInterpCode is a mechanistic interpretability project that explores how internal model representations change during fine-tuning.

##### Specifically, this repository contains code to train a Sparse Autoencoder (SAE) on a specific layer of the Pythia-160M model. By doing this, we can extract human-interpretable features and evaluate whether (and how) these features change after the base model is fine-tuned on a narrow task.
---
## Repository Structure
##### The workflow is divided into three main stages, each handled by a dedicated script/notebook:
##### notebook1-finetune-pythia (1).ipynb
##### Handles the fine-tuning of the base Pythia-160M model on a specific, narrow dataset/task.
##### notebook2_train_saes_.py
##### Trains Sparse Autoencoders (SAEs) on the activations of a chosen layer in the Pythia model. This step is crucial for disentangling the activations into understandable features.
##### notebook3_feature_comparison_.py
##### Compares the SAE-extracted features from the base model against the features from the fine-tuned model to measure feature shift, ##### emergence, or collapse.
---
