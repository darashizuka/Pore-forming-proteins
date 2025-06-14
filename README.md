# Prediction of Pore-forming-proteins 
This repository contains code for building and evaluating machine learning models to classify pore-forming proteins using various protein features.

---
## Dataset
Protein sequence data used in our study is sourced from https://github.com/basf/pore_forming_protein_DLM. We use these sequences to extract several features and embeddings using protein language model

## Setup Instructions
### 1. Clone this GitHub repository
```bash
git clone https://github.com/darashizuka/Pore-forming-proteins.git
cd Pore-forming-proteins
```
### 2. Install required python packages
```bash
pip install -r requirements.txt
```
### 3. Download and extract the data
Run the setup script to automatically download and extract the Data folder from Google Drive:
```bash
python setup_data.py
```
This script will:
- Download a zipped folder from Google Drive
- Extract it to the project root
After this step, you should see a Data/ folder in your project directory
  
### 4. Run the main pipeline script
```bash
python pore_forming_pipeline.py
```
You will be prompted to:
- Choose which models to run
- Select feature types to use

### 5. (Optional) Save output to a file
To save console output which includes results to a file:
```bash
python pore_forming_pipeline.py > results.txt
```
---
Note: If you want to use different paths, edit the config.yaml file in the project root.




