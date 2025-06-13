# protbert ensemble

#pore forming proteins - dl dataset, equal samples from all classes', only protbert (no feature selection)

import pandas as pd 
import numpy as np
from Bio import SeqIO
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')
from config_loader import load_config
import os

config = load_config()

# Define the set of standard amino acids and dipeptides
standard_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
dipeptides = [a1 + a2 for a1 in standard_amino_acids for a2 in standard_amino_acids]

def split_fasta_sequences(fasta_files):
    train_sequences = []
    test_sequences = []
    for fasta_file in fasta_files:
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        train, test = train_test_split(sequences, test_size=0.2, random_state=42)
        train_sequences.extend([record.id for record in train])
        test_sequences.extend([record.id for record in test])
    return train_sequences, test_sequences

def calculate_aa_composition(fasta_file):
    composition_data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        composition = {aa: 0 for aa in standard_amino_acids}
        total_length = len(sequence)
        for aa in sequence:
            if aa in composition:
                composition[aa] += 1
        composition = {aa: count / total_length for aa, count in composition.items()}
        composition['sequence_id'] = record.id
        composition_data.append(composition)
    return pd.DataFrame(composition_data)

def calculate_dipeptide_composition(fasta_file):
    composition_data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        composition = {dp: 0 for dp in dipeptides}
        total_length = len(sequence)
        for i in range(total_length - 1):
            dipeptide = sequence[i:i + 2]
            if dipeptide in composition:
                composition[dipeptide] += 1
        composition = {dp: count / (total_length - 1) for dp, count in composition.items()}
        composition['sequence_id'] = record.id
        composition_data.append(composition)
    return pd.DataFrame(composition_data)

def load_and_merge_data(fasta_files, pseaac_paths, physiochem_paths, esm_paths, prot_paths, labels, train_ids, test_ids):
    combined_df_list = []

    for fasta_file, pseaac_path, physiochem_path, esm_path, prot_path, label in zip(fasta_files, pseaac_paths, physiochem_paths, esm_paths, prot_paths, labels):
        prot_df = pd.read_csv(prot_path)
        prot_df['label'] = label
        combined_df_list.append(prot_df)

    full_df = pd.concat(combined_df_list, axis=0, ignore_index=True)

    # Split into train and test data based on sequence IDs
    train_df = full_df[full_df['ID'].isin(train_ids)]
    test_df = full_df[full_df['ID'].isin(test_ids)]
    print(train_df.shape, test_df.shape)
    return train_df, test_df

# PyTorch ANN Model
class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, max(10, input_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(10, input_dim // 2), max(10, input_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(10, input_dim // 4), 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=100, learning_rate=0.0005, batch_size=128, device=None):
        self.input_dim = input_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        # Convert to tensors
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, device=self.device).float()
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, device=self.device).float().view(-1, 1)
        
        # Initialize model
        self.model = ANNModel(self.input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        dataset_size = X_tensor.shape[0]
        num_batches = (dataset_size + self.batch_size - 1) // self.batch_size
        
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i in range(num_batches):
                start = i * self.batch_size
                end = min(start + self.batch_size, dataset_size)
                inputs = X_tensor[start:end]
                labels = y_tensor[start:end]

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"ANN Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / num_batches:.4f}")
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, device=self.device).float()
            predictions = self.model(X_tensor).cpu().numpy()
            return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, device=self.device).float()
            predictions = self.model(X_tensor).cpu().numpy().flatten()
            # Return probabilities for both classes
            return np.column_stack([1 - predictions, predictions])

# File paths (using your original paths)
positive_fasta_files = [
    os.path.join(config['data_dir'], "alpha_pfts", "Uniprot_Actinoporin_148_cleanv1.fasta"),
    os.path.join(config['data_dir'], "alpha_pfts", "Uniprot_Colicin_1000_cleanv1.fasta"),
    os.path.join(config['data_dir'], "alpha_pfts", "Uniprot_Hemolysin_2000_cleanv1.fasta"),
    os.path.join(config['data_dir'], "alpha_pfts", "Uniprot_PesticidalCrystal_334_cleanv1.fasta"),
    os.path.join(config['data_dir'], "beta_pfts", "Uniprot_Aerolysin_419_cleanv1.fasta"),
    os.path.join(config['data_dir'], "beta_pfts", "Uniprot_Cytolysin_2023_cleanv1.fasta"),
    os.path.join(config['data_dir'], "beta_pfts", "Uniprot_Haemolysin_387_cleanv1.fasta"),
    os.path.join(config['data_dir'], "beta_pfts", "Uniprot_Perfringolysin_AlphaHemolysin_Leucocidin_422_cleanv1.fasta")
]

negative_fasta_files = [
    config['cullpdb_fasta']  
]

positive_pseaac_paths = [
    os.path.join(config['pseaac_dir'], "pseaac_alpha_actinoporin.csv"),
    os.path.join(config['pseaac_dir'], "pseaac_alpha_colicin.csv"),
    os.path.join(config['pseaac_dir'], "pseaac_alpha_hemolysin.csv"),
    os.path.join(config['pseaac_dir'], "pseaac_alpha_pesticidal.csv"),
    os.path.join(config['pseaac_dir'], "pseaac_beta_aerolysin.csv"),
    os.path.join(config['pseaac_dir'], "pseaac_beta_cytolysin.csv"),
    os.path.join(config['pseaac_dir'], "pseaac_beta_haemolysin.csv"),
    os.path.join(config['pseaac_dir'], "pseaac_beta_perfringolysin.csv")
]

negative_pseaac_path = [
    os.path.join(config['pseaac_dir'], "pseaac_cullpdb_negative.csv")
]

positive_physiochem = [
    os.path.join(config['physico_dir'], "physiochem_actinoporin.csv"),
    os.path.join(config['physico_dir'], "physiochem_colicin.csv"),
    os.path.join(config['physico_dir'], "physiochem_hemolysin.csv"),
    os.path.join(config['physico_dir'], "physiochem_pesticidal.csv"),
    os.path.join(config['physico_dir'], "physiochem_aerolycin.csv"),
    os.path.join(config['physico_dir'], "physiochem_cytolysin.csv"),
    os.path.join(config['physico_dir'], "physiochem_haemolysin.csv"),
    os.path.join(config['physico_dir'], "physiochem_perfringolysin.csv")
]

negative_physiochem = [
    os.path.join(config['physico_dir'], "physiochem_cullpdb_negative.csv")

]

positive_esm_embeddings_paths = [
    os.path.join(config['embedding_dir'], "Uniprot_Actinoporin_148_cleanv1_embeddings.csv"),
    os.path.join(config['embedding_dir'], "Uniprot_Colicin_1000_cleanv1_embeddings.csv"),
    os.path.join(config['embedding_dir'], "Uniprot_Hemolysin_2000_cleanv1_embeddings.csv"),
    os.path.join(config['embedding_dir'], "Uniprot_PesticidalCrystal_334_cleanv1.csv"),
    os.path.join(config['embedding_dir'], "Uniprot_Aerolysin_419_cleanv1_embeddings.csv"),
    os.path.join(config['embedding_dir'], "Uniprot_Cytolysin_2023_cleanv1_embeddings.csv"),
    os.path.join(config['embedding_dir'], "Uniprot_Haemolysin_387_cleanv1_embeddings.csv"),
    os.path.join(config['embedding_dir'], "Uniprot_Perfringolysin_AlphaHemolysin_Leucocidin_422_cleanv1_embeddings.csv")
]

negative_esm_embeddings_path = [
    os.path.join(config['embedding_dir'], "Cullpdb_5616HitsRemoved_embeddings.csv")
]

positive_protbert = [
    os.path.join(config['protbert_dir'], 'Uniprot_Actinoporin_148_cleanv1_protembeddings.csv'), 
    os.path.join(config['protbert_dir'], 'Uniprot_Aerolysin_419_cleanv1_protembeddings.csv'),
    os.path.join(config['protbert_dir'], 'Uniprot_Colicin_1000_cleanv1_protembeddings.csv'),
    os.path.join(config['protbert_dir'], 'Uniprot_Cytolysin_2023_cleanv1_protembeddings.csv'),
    os.path.join(config['protbert_dir'], 'Uniprot_Haemolysin_387_cleanv1_protembeddings.csv'),
    os.path.join(config['protbert_dir'], 'Uniprot_Hemolysin_2000_cleanv1_protembeddings.csv'),
    os.path.join(config['protbert_dir'], 'Uniprot_Perfringolysin_AlphaHemolysin_Leucocidin_422_cleanv1_protembeddings.csv'),
    os.path.join(config['protbert_dir'], 'Uniprot_PesticidalCrystal_334_cleanv1_protembeddings.csv')
]

negative_protbert = [
    os.path.join(config['protbert_dir'], 'Cullpdb_5616HitsRemoved_protembeddings.csv')
]

all_fasta_files = positive_fasta_files + negative_fasta_files
all_pseaac_paths = positive_pseaac_paths + negative_pseaac_path
all_esm_embeddings_paths = positive_esm_embeddings_paths + negative_esm_embeddings_path
all_physiochem_paths = positive_physiochem + negative_physiochem
all_protbert_paths = positive_protbert + negative_protbert

# Create labels (1 for positive, 0 for negative)
all_labels = [1] * len(positive_fasta_files) + [0] * len(negative_fasta_files)

train_ids, test_ids = split_fasta_sequences(all_fasta_files)

# Load and merge all data
train_df, test_df = load_and_merge_data(all_fasta_files, all_pseaac_paths, all_physiochem_paths, 
                                       all_esm_embeddings_paths, all_protbert_paths, all_labels, 
                                       train_ids=train_ids, test_ids=test_ids)

print("Train DataFrame shape:", train_df.shape)
print("Test DataFrame shape:", test_df.shape)

# Prepare features and labels
X_train = train_df.drop(columns=['ID', 'label'], errors='ignore')
y_train = train_df['label']
X_test = test_df.drop(columns=['ID', 'label'], errors='ignore')
y_test = test_df['label']

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Check for NaN values
nan_count = X_train.isna().sum()
total_nan = X_train.isna().sum().sum()
print(f"Total number of NaN values in X_train: {total_nan}")

# Define models and parameter grids
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "ANN": PyTorchClassifier(input_dim=X_train.shape[1], epochs=100, learning_rate=0.0005, batch_size=128)
}

param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'subsample': [0.8, 1.0]
    },
    "ANN": {
        'epochs': [50, 100],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [64, 128]
    }
}

print("\n" + "="*50)
print("INDIVIDUAL MODEL TRAINING AND EVALUATION")
print("="*50)

best_models = {}

# Train individual models
for model_name, model in models.items():
    print(f"\nPerforming GridSearchCV for {model_name}...")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        scoring='f1', 
        cv=5,  # Reduced CV for faster training
        n_jobs=-1 if model_name != "ANN" else 1,  # ANN doesn't support parallel jobs
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score for {model_name}: {grid_search.best_score_:.4f}")

# Evaluate individual models
print("\n" + "="*50)
print("INDIVIDUAL MODEL PERFORMANCE ON TEST SET")
print("="*50)

individual_results = {}

for model_name, best_model in best_models.items():
    print(f"\nEvaluating {model_name}...")
    
    # Fit the model if it's not already fitted
    if hasattr(best_model, 'model') and best_model.model is None:  # For ANN
        best_model.fit(X_train, y_train)
    elif not hasattr(best_model, 'classes_'):  # For sklearn models
        best_model.fit(X_train, y_train)
    
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    
    # Store results
    individual_results[model_name] = {
        'predictions': predictions,
        'probabilities': probabilities,
        'accuracy': accuracy_score(y_test, predictions),
        'f1': f1_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'auc': roc_auc_score(y_test, probabilities),
        'mcc': matthews_corrcoef(y_test, predictions)
    }
    
    print(f"{model_name} Results:")
    print(f"  Accuracy: {individual_results[model_name]['accuracy']:.4f}")
    print(f"  F1 Score: {individual_results[model_name]['f1']:.4f}")
    print(f"  Precision: {individual_results[model_name]['precision']:.4f}")
    print(f"  Recall: {individual_results[model_name]['recall']:.4f}")
    print(f"  AUC Score: {individual_results[model_name]['auc']:.4f}")
    print(f"  MCC: {individual_results[model_name]['mcc']:.4f}")

print("\n" + "="*50)
print("ENSEMBLE METHODS")
print("="*50)

# Prepare models for ensemble (exclude ANN from sklearn VotingClassifier)
sklearn_models = [(name, model) for name, model in best_models.items() if name != "ANN"]

# 1. Hard Voting Ensemble (sklearn models only)
print("\n1. Hard Voting Ensemble (RF, SVM, XGBoost):")
hard_voting = VotingClassifier(estimators=sklearn_models, voting='hard')
hard_voting.fit(X_train, y_train)
hard_voting_pred = hard_voting.predict(X_test)

print("Hard Voting Results:")
print(f"  Accuracy: {accuracy_score(y_test, hard_voting_pred):.4f}")
print(f"  F1 Score: {f1_score(y_test, hard_voting_pred):.4f}")
print(f"  Precision: {precision_score(y_test, hard_voting_pred):.4f}")
print(f"  Recall: {recall_score(y_test, hard_voting_pred):.4f}")
print(f"  MCC: {matthews_corrcoef(y_test, hard_voting_pred):.4f}")

# 2. Soft Voting Ensemble (sklearn models only)
print("\n2. Soft Voting Ensemble (RF, SVM, XGBoost):")
soft_voting = VotingClassifier(estimators=sklearn_models, voting='soft')
soft_voting.fit(X_train, y_train)
soft_voting_pred = soft_voting.predict(X_test)
soft_voting_proba = soft_voting.predict_proba(X_test)[:, 1]

print("Soft Voting Results:")
print(f"  Accuracy: {accuracy_score(y_test, soft_voting_pred):.4f}")
print(f"  F1 Score: {f1_score(y_test, soft_voting_pred):.4f}")
print(f"  Precision: {precision_score(y_test, soft_voting_pred):.4f}")
print(f"  Recall: {recall_score(y_test, soft_voting_pred):.4f}")
print(f"  AUC Score: {roc_auc_score(y_test, soft_voting_proba):.4f}")
print(f"  MCC: {matthews_corrcoef(y_test, soft_voting_pred):.4f}")

# 3. Custom Ensemble (All 4 models including ANN)
print("\n3. Custom Ensemble (RF, SVM, XGBoost, ANN):")

# Get all predictions and probabilities
all_predictions = np.array([individual_results[name]['predictions'] for name in models.keys()])
all_probabilities = np.array([individual_results[name]['probabilities'] for name in models.keys()])

# Hard voting for all models
custom_hard_pred = np.round(np.mean(all_predictions, axis=0)).astype(int)

# Soft voting for all models
custom_soft_proba = np.mean(all_probabilities, axis=0)
custom_soft_pred = (custom_soft_proba > 0.5).astype(int)

print("Custom Hard Voting Results (All 4 models):")
print(f"  Accuracy: {accuracy_score(y_test, custom_hard_pred):.4f}")
print(f"  F1 Score: {f1_score(y_test, custom_hard_pred):.4f}")
print(f"  Precision: {precision_score(y_test, custom_hard_pred):.4f}")
print(f"  Recall: {recall_score(y_test, custom_hard_pred):.4f}")
print(f"  MCC: {matthews_corrcoef(y_test, custom_hard_pred):.4f}")

print("\nCustom Soft Voting Results (All 4 models):")
print(f"  Accuracy: {accuracy_score(y_test, custom_soft_pred):.4f}")
print(f"  F1 Score: {f1_score(y_test, custom_soft_pred):.4f}")
print(f"  Precision: {precision_score(y_test, custom_soft_pred):.4f}")
print(f"  Recall: {recall_score(y_test, custom_soft_pred):.4f}")
print(f"  AUC Score: {roc_auc_score(y_test, custom_soft_proba):.4f}")
print(f"  MCC: {matthews_corrcoef(y_test, custom_soft_pred):.4f}")

# 4. Weighted Ensemble based on individual F1 scores
print("\n4. Weighted Ensemble (All 4 models):")
f1_scores = [individual_results[name]['f1'] for name in models.keys()]
weights = np.array(f1_scores) / np.sum(f1_scores)

print("Model weights based on F1 scores:")
for name, weight in zip(models.keys(), weights):
    print(f"  {name}: {weight:.4f}")

weighted_proba = np.average(all_probabilities, axis=0, weights=weights)
weighted_pred = (weighted_proba > 0.5).astype(int)

print("\nWeighted Ensemble Results:")
print(f"  Accuracy: {accuracy_score(y_test, weighted_pred):.4f}")
print(f"  F1 Score: {f1_score(y_test, weighted_pred):.4f}")
print(f"  Precision: {precision_score(y_test, weighted_pred):.4f}")
print(f"  Recall: {recall_score(y_test, weighted_pred):.4f}")
print(f"  AUC Score: {roc_auc_score(y_test, weighted_proba):.4f}")
print(f"  MCC: {matthews_corrcoef(y_test, weighted_pred):.4f}")

print("\n" + "="*50)
print("SUMMARY OF ALL RESULTS")
print("="*50)

# Create summary table
summary_data = []

# Individual models
for name in models.keys():
    summary_data.append([
        name,
        f"{individual_results[name]['accuracy']:.4f}",
        f"{individual_results[name]['f1']:.4f}",
        f"{individual_results[name]['precision']:.4f}",
        f"{individual_results[name]['recall']:.4f}",
        f"{individual_results[name]['auc']:.4f}",
        f"{individual_results[name]['mcc']:.4f}"
    ])

# Ensemble methods
summary_data.append([
    "Hard Voting (3 models)",
    f"{accuracy_score(y_test, hard_voting_pred):.4f}",
    f"{f1_score(y_test, hard_voting_pred):.4f}",
    f"{precision_score(y_test, hard_voting_pred):.4f}",
    f"{recall_score(y_test, hard_voting_pred):.4f}",
    "N/A",
    f"{matthews_corrcoef(y_test, hard_voting_pred):.4f}"
])

summary_data.append([
    "Soft Voting (3 models)",
    f"{accuracy_score(y_test, soft_voting_pred):.4f}",
    f"{f1_score(y_test, soft_voting_pred):.4f}",
    f"{precision_score(y_test, soft_voting_pred):.4f}",
    f"{recall_score(y_test, soft_voting_pred):.4f}",
    f"{roc_auc_score(y_test, soft_voting_proba):.4f}",
    f"{matthews_corrcoef(y_test, soft_voting_pred):.4f}"
])

summary_data.append([
    "Custom Soft (4 models)",
    f"{accuracy_score(y_test, custom_soft_pred):.4f}",
    f"{f1_score(y_test, custom_soft_pred):.4f}",
    f"{precision_score(y_test, custom_soft_pred):.4f}",
    f"{recall_score(y_test, custom_soft_pred):.4f}",
    f"{roc_auc_score(y_test, custom_soft_proba):.4f}",
    f"{matthews_corrcoef(y_test, custom_soft_pred):.4f}"
])

summary_data.append([
    "Weighted Ensemble (4 models)",
    f"{accuracy_score(y_test, weighted_pred):.4f}",
    f"{f1_score(y_test, weighted_pred):.4f}",
    f"{precision_score(y_test, weighted_pred):.4f}",
    f"{recall_score(y_test, weighted_pred):.4f}",
    f"{roc_auc_score(y_test, weighted_proba):.4f}",
    f"{matthews_corrcoef(y_test, weighted_pred):.4f}"
])

# Print summary table
print(f"{'Model':<25} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'AUC':<10} {'MCC':<10}")
print("-" * 85)
for row in summary_data:
    print(f"{row[0]:<25} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10} {row[6]:<10}")

# Save the best models
print(f"\nSaving best models...")