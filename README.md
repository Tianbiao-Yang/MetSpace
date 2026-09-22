# MetSpace

AI-guided discovery and expansion of the human bile acid landscape


## 1. Project Overview
-------------------
Here, we present MetLLM, a transformer-based language model that learns bile acid modification patterns to prioritize candidate bile acid analogues, and MetSpace, an in silico bile acid chemical landscape containing 3,900 prioritized candidates. Liquid chromatography–high resolution mass spectrometry analysis of human feces confirmed multiple newly detected and structurally novel bile acids, including MS2141 and MS1281. In vitro functional validation showed that MS2141 modulates M2 muscarinic receptor-associated inhibitory G protein signaling and alters electrophysiological parameters in human induced pluripotent stem cell-derived cardiomyocytes. This work expands the searchable bile acid metabolome and provides a computational framework for systematic metabolite discovery.


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

