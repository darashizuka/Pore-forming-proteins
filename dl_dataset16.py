#  basic features - ann
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, classification_report
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score

import os
import matplotlib.pyplot as plt
from config_loader import load_config
import os

config = load_config()

plot_path = config['plot_path']

standard_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
dipeptides = [a1 + a2 for a1 in standard_amino_acids for a2 in standard_amino_acids]

def calculate_aa_composition(df):
    composition_data = []
    for idx, row in df.iterrows():
        sequence = row['seq']
        composition = {aa: 0 for aa in standard_amino_acids}
        total_length = len(sequence)
        for aa in sequence:
            if aa in composition:
                composition[aa] += 1
        composition = {aa: count / total_length for aa, count in composition.items()}
        composition['sequence_id'] = row['sequence_id']
        composition_data.append(composition)
    return pd.DataFrame(composition_data)

def calculate_dipeptide_composition(df):
    composition_data = []
    for idx, row in df.iterrows():
        sequence = row['seq']
        composition = {dp: 0 for dp in dipeptides}
        total_length = len(sequence)
        for i in range(total_length - 1):
            dipeptide = sequence[i:i + 2]
            if dipeptide in composition:
                composition[dipeptide] += 1
        composition = {dp: count / (total_length - 1) for dp, count in composition.items()}
        composition['sequence_id'] = row['sequence_id']
        composition_data.append(composition)
    return pd.DataFrame(composition_data)


# Load dataset
file_path = config['train_test_csv']
df = pd.read_csv(file_path)
df = df.drop_duplicates(subset=["sequence_id"], keep="first")


# Load physicochemical properties
physico_path = config['physico_csv']  
physico_df = pd.read_csv(physico_path)
physico_df = physico_df.drop_duplicates(subset=["sequence_id"], keep="first")

# Merge datasets on sequence_id
df = df.merge(physico_df, on="sequence_id", how="left")


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

pseaac_df = pd.concat([pd.read_csv(path) for path in positive_pseaac_paths + negative_pseaac_path], ignore_index=True)

pseaac_df['sequence_id'] = pseaac_df['sequence_id'].str.replace('>', '', regex=False)
pseaac_df = pseaac_df.drop_duplicates(subset=["sequence_id"], keep="first")

df = df.merge(pseaac_df, on="sequence_id", how="inner")
print(pseaac_df.shape)
print(df.isna().sum().sum())

aa_composition_df = calculate_aa_composition(df)
dipeptide_composition_df = calculate_dipeptide_composition(df)

df = df.merge(aa_composition_df, on='sequence_id', how='left')
df = df.merge(dipeptide_composition_df, on='sequence_id', how='left')

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

negative_protbert = os.path.join(config['protbert_dir'], 'Cullpdb_5616HitsRemoved_protembeddings.csv')

# Combine all positive embeddings
positive_prot_dfs = [pd.read_csv(path) for path in positive_protbert]
positive_prot_df = pd.concat(positive_prot_dfs)

negative_prot_df = pd.read_csv(negative_protbert)

prot_df = pd.concat([positive_prot_df, negative_prot_df])

prot_df = prot_df.rename(columns={'ID': 'sequence_id'})
# df = df.merge(prot_df, on='sequence_id', how='left')

print("Final DataFrame shape:", df.shape)

feature_columns = [col for col in df.columns if col not in ["sequence_id", "seq", "is_Train", "label"]]
feature_data = df[feature_columns]
feature_data["is_Train"] = df["is_Train"]
feature_data["label"] = df["label"]


train_data = feature_data[feature_data["is_Train"] == 1].drop(columns=["is_Train"])
test_data = feature_data[feature_data["is_Train"] == 0].drop(columns=["is_Train"])

X_train = train_data.drop(columns=["label"]) 
X_test = test_data.drop(columns=["label"])
y_train = train_data["label"]
y_test = test_data["label"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)

#ann
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


# Training function
def fit(model, X_train, Y_train, epochs=100, learning_rate=0.0005, batch_size=128):
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

        print(f"Epoch {epoch + 1}, Loss: {running_loss / num_batches}")
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

# Prediction functions
def predict(model, X_batch):
    with torch.no_grad():
        return model(X_batch).round()

def predict_proba(model, X_batch):
    with torch.no_grad():
        return model(X_batch)

# Example usage
input_dim = X_train.shape[1] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train, device=device).float()
Y_train_tensor = torch.tensor(y_train.values, device=device).float().view(-1, 1)
X_test_tensor = torch.tensor(X_test, device=device).float()

model = build_model(input_dim).to(device)
fit(model, X_train_tensor, Y_train_tensor, epochs=100, learning_rate=0.0005, batch_size=128)

y_pred = predict(model, X_test_tensor).cpu().numpy()
y_proba = predict_proba(model, X_test_tensor).cpu().numpy()

print("ANN Classification Report:\n", classification_report(y_test, y_pred))
print("ANN Accuracy:", accuracy_score(y_test, y_pred))
print("ANN F1 Score:", f1_score(y_test, y_pred))
print("ANN AUC Score:", roc_auc_score(y_test, y_proba))
print("ANN MCC:", matthews_corrcoef(y_test, y_pred))
print("ANN Precision:", precision_score(y_test, y_pred))
print("ANN Recall:", recall_score(y_test, y_pred))