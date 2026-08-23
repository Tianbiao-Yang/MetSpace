# MetSpace

AI-Driven Reconstruction of the Human Bile Acid Sub-Metabolome.


## 1. Project Overview
-------------------
Here, we present MetLLM, a transformer-based language model that decoded the bile acid chemical space to generate 3,900 candidate BA analogues, many with previously undetected or uncharacterized structures. Our AI-driven platform MetSpace reveals extensive structural diversity, including potential modification patterns absent from current databases. Experimental validation in human feces confirmed multiple new BAs, establishing MetSpace as a resource for BA discovery.

## 2. Data Preparation
-------------------
- Data Files:
  - `HMDB_Database.txt`: Used for pretraining the model.
  - `BAs_set.txt`: Used for fine-tuning.
  - `Smiles_Input.txt`: Contains SMILES strings to evaluate.
- Format:
  - Each line should contain a single SMILES string.
- Location:
  - Place data files in the `./data/` directory relative to the scripts.

## 3. Pre-Training and Fine-tuning the MetLLM Model
---------------------
Run the training script: `python ./code/MetSpace_Training.py`
The script:
- Builds a character-level Byte Pair Encoding (BPE) tokenizer.
- Pretrains the GPT-2 model on the HMDB dataset.
- Fine-tunes the model on the BAs dataset.
- Saves models at each epoch in the `./saved_models/` directory.
- Expected run time for demo on a "normal" desktop computer about 7-15 days.

## 4. MetLLM Evaluation
----------------------------
After training, you can evaluate SMILES strings using the scoring script: `python ./code/MetSpace_Predict.py`
- Inputs:
  - `./data/Smiles_Input.txt` (SMILES strings to score)
- Outputs:
  - `./result/Smiles_Input_scores_test.txt` (SMILES with their scores)
- Expected run time for demo on a "normal" desktop computer about 5-10 min.

The scoring method:
- Computes the probability of each SMILES under the model.
- The scores to a 0-1 range, where higher scores suggest better likelihood.

## 5. Environment Setup
---------------------
- Python 3.11.5
- Install dependencies: To set up the environment, you can utilize the `./code/metspace.yaml` file, representing the conda environment for this project (`conda env create -f ./code/metspace.yaml`). Alternatively, you have the option to deploy the environment using the `./code/metspace.txt` file.
- Typical installation on a standard desktop computer takes about one to two hours.

## 6. AttenRT and BA Receptor
---------------------
- Two-Stage AttenRT Model for Retention Time Prediction was in the `./code/AttenRT` directory.
- AI-based pharmacological screening of MS2141 against BA-related receptors was in the `./code/BAReceptor` directory.

## 7. Contact
----------
The code repository is available at https://doi.org/10.5281/zenodo.20320639 on Zenodo. For questions or feedback, please contact: tianbiao@hku.hk.

