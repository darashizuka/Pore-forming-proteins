# only pseaac - ann
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, classification_report
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score
import re
import os
import matplotlib.pyplot as plt
import random
from config_loader import load_config
import os

config = load_config()

SEED = 40

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
plot_path = config['plot_path']
# Load dataset
file_path = config['train_test_csv']
df = pd.read_csv(file_path)
df = df.drop_duplicates(subset=["sequence_id"], keep="first")


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

all_pseaac_paths = positive_pseaac_paths + negative_pseaac_path
pseaac_df = pd.concat([pd.read_csv(path) for path in all_pseaac_paths], ignore_index=True)
pseaac_df['sequence_id'] = pseaac_df['sequence_id'].str.replace('^>', '', regex=True)
pseaac_df = pseaac_df.drop_duplicates(subset=["sequence_id"], keep="first")


# Merge datasets on sequence_id
df = df.merge(pseaac_df, on="sequence_id", how="left")

df = df.dropna()

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

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

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