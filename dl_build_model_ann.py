#amino and dipeptide composition - ann
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef,classification_report
import pickle
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from config_loader import load_config

config = load_config()

plot_path = config['plot_path']
feature_pickle_path = config['feature_pickle']

print("10 fold grid search")
with open(feature_pickle_path, "rb") as f:
    df = pickle.load(f)

df = df.loc[:, ~df.columns.str.contains('t2feat|tfeat')]
df = df.drop_duplicates(subset='sequence_id', keep='first')
print(f"Shape of the df after removing duplicates: {df.shape}")

# Prepare data for training
def prepare_data(df, representation, feature_type, offset_combinations):
    features = []
    for index, row in df.iterrows():
        row_input = []
        
        if feature_type == 'counts':
            row_input = np.hstack([row_input, row[f"{representation}_counts"].flatten()])
        elif feature_type == 'mat':
            for offset in offset_combinations:
                matrix = row[f"{representation}_mat_{offset}"].flatten()
                row_input = np.hstack([row_input, matrix])
        
        features.append(row_input)
    
    labels = df['label'].values
    return np.array(features), labels


# Main processing
train_df = df[df['is_Train'] == 1]
test_df = df[df['is_Train'] == 0]

representations = ['no_reduction', 'hydro', 'conform']
feature_types = ['counts', 'mat']
offset_combinations = [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]

results = []

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

        # print(f"Epoch {epoch + 1}, Loss: {running_loss / num_batches}")
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

def predict(model, X_batch):
    with torch.no_grad():
        return model(X_batch).round()

def predict_proba(model, X_batch):
    with torch.no_grad():
        return model(X_batch)

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
 
for representation in representations:
    for feature_type in feature_types:
        if feature_type == 'mat':
            for offset_combination in offset_combinations:
                print(f"\nProcessing: {representation}, {feature_type}, offset: {offset_combination}")
                X_train, y_train = prepare_data(train_df, representation, feature_type, offset_combination)
                X_test, y_test = prepare_data(test_df, representation, feature_type, offset_combination)
                print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                input_dim = X_train.shape[1] 
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                X_train_tensor = torch.tensor(X_train, device=device).float()
                if hasattr(y_train, 'values'):
                    Y_train_tensor = torch.tensor(y_train.values, device=device).float().view(-1, 1)
                else:
                    Y_train_tensor = torch.tensor(y_train, device=device).float().view(-1, 1)
            
                X_test_tensor = torch.tensor(X_test, device=device).float()

                model = build_model(input_dim).to(device)
                fit(model, X_train_tensor, Y_train_tensor, epochs=100, learning_rate=0.0005, batch_size=128)
                y_pred = predict(model, X_test_tensor).cpu().numpy()
                y_proba = predict_proba(model, X_test_tensor).cpu().numpy()
                
                metrics = calculate_metrics(y_test, y_pred, y_proba)
                print("Metrics:", metrics)
                results.append([representation, feature_type, str(offset_combination)] + list(metrics.values()))
               
                            
        else:
            X_train, y_train = prepare_data(train_df, representation, feature_type, [])
            X_test, y_test = prepare_data(test_df, representation, feature_type, [])
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            input_dim = X_train.shape[1] 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            X_train_tensor = torch.tensor(X_train, device=device).float()
            if hasattr(y_train, 'values'):
                Y_train_tensor = torch.tensor(y_train.values, device=device).float().view(-1, 1)
            else:
                Y_train_tensor = torch.tensor(y_train, device=device).float().view(-1, 1)
           
            X_test_tensor = torch.tensor(X_test, device=device).float()

            model = build_model(input_dim).to(device)
            fit(model, X_train_tensor, Y_train_tensor, epochs=100, learning_rate=0.0005, batch_size=128)
            y_pred = predict(model, X_test_tensor).cpu().numpy()
            y_proba = predict_proba(model, X_test_tensor).cpu().numpy()

            metrics = calculate_metrics(y_test, y_pred, y_proba)
            print("Metrics:", metrics)
            results.append([representation, feature_type,  "N/A"] + list(metrics.values()))
    
# Save results to CSV
columns = ['Representation', 'Feature Type', 'Offsets', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC', 'MCC']
pd.DataFrame(results, columns=columns).to_csv("all_features_models_ann.csv", index=False)
print("Results saved")
