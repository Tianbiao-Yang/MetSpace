# MetSpace

AI-guided discovery and expansion of the human bile acid landscape


## 1. Project Overview
-------------------
Here, MetLLM is a transformer-based language model for discovering novel bile acid analogues by learning bile acid modification patterns. This repository provides the code and resources for MetLLM, together with MetSpace, an in silico bile acid chemical landscape containing 3,900 prioritized candidates. The framework supports systematic expansion and exploration of the searchable bile acid metabolome.


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

## 3. Environment Setup
---------------------
- Python 3.11.5
- Install dependencies: To set up the environment, you can utilize the `./code/metspace.yaml` file, representing the conda environment for this project (`conda env create -f ./code/metspace.yaml`). Alternatively, you have the option to deploy the environment using the `./code/metspace.txt` file.
- Typical installation on a standard desktop computer takes about one to two hours.

## 4. Running MetLLM Model
----------------------------
About running the MetLLM model, you can evaluate SMILES strings using the scoring script: `python ./code/Running_MetLLM.py`
- Inputs:
  - `./data/Smiles_Input.txt` (SMILES strings to score)
- Outputs:
  - `./result/Smiles_Input_scores_test.txt` (SMILES with their scores)
- Expected run time for demo on a "normal" desktop computer about 5-10 min.

The scoring method:
- Computes the probability of each SMILES under the model.
- The scores to a 0-1 range, where higher scores suggest better likelihood.

## 5. AttenRT and BA Receptor
---------------------
- Two-Stage AttenRT Model for Retention Time Prediction was in the `./code/AttenRT` directory.
- AI-based pharmacological screening of MS2141 against BA-related receptors was in the `./code/BAReceptor` directory.

## 6. Contact
----------
For questions or feedback, please contact: tianbiao_yang at 126 dot com.

