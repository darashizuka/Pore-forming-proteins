#esm ensemble

#pore forming proteins - dl dataset, equal samples from all classes', only esm - no feature selection

import pandas as pd 
import numpy as np
from Bio import SeqIO
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
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

def load_and_merge_data(fasta_files, pseaac_paths, physiochem_paths, esm_paths, labels, train_ids, test_ids):
    combined_df_list = []

    for fasta_file, pseaac_path, physiochem_path, esm_path, label in zip(fasta_files, pseaac_paths, physiochem_paths, esm_paths, labels):
        esm_df = pd.read_csv(esm_path)
        esm_df['sequence_id'] = esm_df['sequence_id'].str.lstrip('>')
        esm_df['label'] = label
        combined_df_list.append(esm_df)

    full_df = pd.concat(combined_df_list, axis=0, ignore_index=True)

    # Split into train and test data based on sequence IDs
    train_df = full_df[full_df['sequence_id'].isin(train_ids)]
    test_df = full_df[full_df['sequence_id'].isin(test_ids)]

    return train_df, test_df

# ANN Model Definition
def build_model(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.Linear(input_dim, max(10, input_dim // 2)),
        nn.ReLU(),
        nn.Linear(max(10, input_dim // 2), max(10, input_dim // 4)),
        nn.ReLU(),
        nn.Linear(max(10, input_dim // 4), 1),
        nn.Sigmoid()
    )
    return model

# Training function for ANN
def fit_ann(model, X_train, Y_train, epochs=100, learning_rate=0.0005, batch_size=128, plot_path="training_loss.png"):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset_size = X_train.shape[0]
    num_batches = (dataset_size + batch_size - 1) // batch_size
    loss_values = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            inputs = X_train[start:end]
            labels = Y_train[start:end]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / num_batches
        loss_values.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close() 
    print(f"Training loss plot saved to: {plot_path}")

# Prediction functions for ANN
def predict_ann(model, X_batch):
    with torch.no_grad():
        return model(X_batch).round()

def predict_proba_ann(model, X_batch):
    with torch.no_grad():
        return model(X_batch)


class ANNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=100, learning_rate=0.0005, batch_size=128):
        self.input_dim = input_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def fit(self, X, y):
        self.model = build_model(self.input_dim).to(self.device)
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, device=self.device).float()
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, device=self.device).float().view(-1, 1)
        fit_ann(self.model, X_tensor, y_tensor, self.epochs, self.learning_rate, self.batch_size)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, device=self.device).float()
        predictions = predict_ann(self.model, X_tensor).cpu().numpy().flatten()
        return predictions.astype(int)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, device=self.device).float()
        proba = predict_proba_ann(self.model, X_tensor).cpu().numpy().flatten()
        # Return probabilities for both classes
        return np.column_stack([1 - proba, proba])

# File paths (replace these with your actual file paths)
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

# Create labels (1 for positive, 0 for negative)
all_labels = [1] * len(positive_fasta_files) + [0] * len(negative_fasta_files)

train_ids, test_ids = split_fasta_sequences(all_fasta_files)

# Load and merge all data
train_df, test_df = load_and_merge_data(all_fasta_files, all_pseaac_paths, all_physiochem_paths, all_esm_embeddings_paths, all_labels, train_ids=train_ids, test_ids=test_ids)

print("Train DataFrame Columns:")
print(train_df.columns.tolist())

print("\nTest DataFrame Columns:")
print(test_df.columns.tolist())

X_train = train_df.drop(columns=['sequence_id', 'label','sequence_id.1'], errors='ignore')
y_train = train_df['label']
X_test = test_df.drop(columns=['sequence_id', 'label', 'sequence_id.1'], errors='ignore')
y_test = test_df['label']

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

nan_count = X_train.isna().sum()
print("Number of NaN values in each column of X_train:")
print(nan_count)

total_nan = X_train.isna().sum().sum()
print(f"Total number of NaN values in X_train: {total_nan}")

# Define models including ANN
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "ANN": ANNClassifier(input_dim=X_train.shape[1], epochs=100, learning_rate=0.0005, batch_size=128)
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
        'epochs': [50, 100, 150],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [64, 128, 256]
    }
}

best_models = {}

# Train and optimize individual models
for model_name, model in models.items():
    print(f"\nPerforming GridSearchCV for {model_name}...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        scoring='f1', 
        cv=5,
        n_jobs=-1 if model_name != "ANN" else 1,  # ANN should use single job due to CUDA
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score for {model_name}: {grid_search.best_score_}")

# Evaluate individual models on test set
individual_results = {}
for model_name, model in best_models.items():
    print(f"\nEvaluating {model_name} with best parameters on test set...")
    predictions = model.predict(X_test)
    prob_predictions = model.predict_proba(X_test)[:, 1]
    
    individual_results[model_name] = {
        'predictions': predictions,
        'probabilities': prob_predictions,
        'accuracy': accuracy_score(y_test, predictions),
        'f1': f1_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'auc': roc_auc_score(y_test, prob_predictions),
        'mcc': matthews_corrcoef(y_test, predictions)
    }
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"{model_name} Accuracy: {individual_results[model_name]['accuracy']:.4f}")
    print(f"{model_name} F1 Score: {individual_results[model_name]['f1']:.4f}")
    print(f"{model_name} Precision: {individual_results[model_name]['precision']:.4f}")
    print(f"{model_name} Recall: {individual_results[model_name]['recall']:.4f}")
    print(f"{model_name} AUC Score: {individual_results[model_name]['auc']:.4f}")
    print(f"{model_name} Matthews Correlation Coefficient: {individual_results[model_name]['mcc']:.4f}")

# Create ensemble model using VotingClassifier (soft voting)
print("\n" + "="*50)
print("CREATING ENSEMBLE MODEL")
print("="*50)


ensemble_models = [(name, model) for name, model in best_models.items()]

# Create soft voting ensemble
ensemble_model = VotingClassifier(
    estimators=ensemble_models,
    voting='soft'  # Use probabilities for voting
)

# Train ensemble
print("Training ensemble model...")
ensemble_model.fit(X_train, y_train)

# Evaluate ensemble
print("Evaluating ensemble model...")
ensemble_predictions = ensemble_model.predict(X_test)
ensemble_probabilities = ensemble_model.predict_proba(X_test)[:, 1]

print("\nENSEMBLE MODEL RESULTS:")
print("="*50)
print("Ensemble Classification Report:")
print(classification_report(y_test, ensemble_predictions))
print(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_predictions):.4f}")
print(f"Ensemble F1 Score: {f1_score(y_test, ensemble_predictions):.4f}")
print(f"Ensemble Precision: {precision_score(y_test, ensemble_predictions):.4f}")
print(f"Ensemble Recall: {recall_score(y_test, ensemble_predictions):.4f}")
print(f"Ensemble AUC Score: {roc_auc_score(y_test, ensemble_probabilities):.4f}")
print(f"Ensemble Matthews Correlation Coefficient: {matthews_corrcoef(y_test, ensemble_predictions):.4f}")

# Compare individual models vs ensemble
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"{'Model':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'AUC':<10} {'MCC':<10}")
print("-"*70)

for model_name, results in individual_results.items():
    print(f"{model_name:<15} {results['accuracy']:<10.4f} {results['f1']:<10.4f} {results['precision']:<10.4f} {results['recall']:<10.4f} {results['auc']:<10.4f} {results['mcc']:<10.4f}")

print(f"{'Ensemble':<15} {accuracy_score(y_test, ensemble_predictions):<10.4f} {f1_score(y_test, ensemble_predictions):<10.4f} {precision_score(y_test, ensemble_predictions):<10.4f} {recall_score(y_test, ensemble_predictions):<10.4f} {roc_auc_score(y_test, ensemble_probabilities):<10.4f} {matthews_corrcoef(y_test, ensemble_predictions):<10.4f}")

# Save models
print("\nSaving models...")
for model_name, model in best_models.items():
    if model_name != "ANN":
        joblib.dump(model, f'best_{model_name.lower().replace(" ", "_")}_model.pkl')
    else:
        # Save ANN model using torch
        torch.save(model.model.state_dict(), 'best_ann_model.pth')

joblib.dump(ensemble_model, 'ensemble_model.pkl')
print("All models saved successfully!")