# MetSpace

AI-Driven Reconstruction of the Bile Acid Metabolome Unveils Thousands of Novel Structures.

This project provides tools for training a GPT-2 based model to generate and evaluate SMILES strings, which are representations of chemical molecules. It includes scripts for data preprocessing, model training, fine-tuning, and scoring SMILES strings based on their likelihood under the trained model.

## 1. Project Overview
-------------------
The goal of this project is to develop a generative model for chemical structures represented as SMILES strings. The model can be trained on a dataset of molecules, fine-tuned for specific datasets, and used to evaluate the quality of SMILES strings by calculating a score indicating their likelihood.

## 2. Data Preparation
-------------------
- Data Files:
  - `HMDB_Database.txt`: Used for pretraining the model.
  - `BAs_set.txt`: Used for fine-tuning.
  - `Smiles_Input.txt`: Contains SMILES strings to evaluate.
- Format:
  - Each line should contain a single SMILES string.
- Location:
  - Place data files in the `../data/` directory relative to the scripts.

## 3. Pre-Training and Fine-tuning the MetLLM Model
---------------------
Run the training script: `python MetSpace_Training.py`
The script:
- Builds a character-level Byte Pair Encoding (BPE) tokenizer.
- Pretrains the GPT-2 model on the HMDB dataset.
- Fine-tunes the model on the BAs dataset.
- Saves models at each epoch in the `../saved_models/` directory.

## 4. MetLLM Evaluation
----------------------------
After training, you can evaluate SMILES strings using the scoring script: `python MetSpace_Predict.py`
- Inputs:
  - `../data/Smiles_Input.txt` (SMILES strings to score)
- Outputs:
  - `../result/Smiles_Input_scores_test.txt` (SMILES with their scores)

The scoring method:
- Computes the probability of each SMILES under the model.
- Normalizes the scores to a 0-1 range, where higher scores suggest better likelihood.

## 5. Environment Setup
---------------------
- Python 3.x
- Install dependencies: `pip install transformers torch tokenizers numpy`
- Use a virtual environment for dependency management.

## 6. Contact
----------
For questions or feedback, please contact: tianbiao@hku.hk.


