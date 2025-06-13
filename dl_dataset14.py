#all basic features with feature selection 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
)
from config_loader import load_config
import os

config = load_config()


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
pseaac_df = pseaac_df.drop_duplicates(subset=["sequence_id"], keep="first")


df = df.merge(pseaac_df, on="sequence_id", how="left")


aa_composition_df = calculate_aa_composition(df)
dipeptide_composition_df = calculate_dipeptide_composition(df)

df = df.merge(aa_composition_df, on='sequence_id', how='left')
df = df.merge(dipeptide_composition_df, on='sequence_id', how='left')

print("Final DataFrame shape:", df.shape)

print(df.shape)
# Drop non-feature columns
feature_columns = [col for col in df.columns if col not in ["sequence_id", "seq", "is_Train", "label"]]
feature_data = df[feature_columns]
feature_data["is_Train"] = df["is_Train"]
feature_data["label"] = df["label"]

# Split into training and test sets
train_data = feature_data[feature_data["is_Train"] == 1].drop(columns=["is_Train"])
test_data = feature_data[feature_data["is_Train"] == 0].drop(columns=["is_Train"])

X_train = train_data.drop(columns=["label"]) 
X_test = test_data.drop(columns=["label"])
y_train = train_data["label"]
y_test = test_data["label"]
print("train:", X_train.shape, y_train.shape, "test:", X_test.shape, y_test.shape)

# Feature selection using Gini importance
rf_feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_feature_selector.fit(X_train, y_train)

feature_importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_feature_selector.feature_importances_
}).sort_values(by="importance", ascending=False)

best_subset = []
best_cv_score = 0
step = 5
cv_scores = []

for i in range(step, len(feature_importances) + 1, step):
    selected_features = feature_importances["feature"].head(i).tolist()
    cv_score = cross_val_score(RandomForestClassifier(random_state=42), 
                               X_train[selected_features], y_train, cv=10, scoring="f1").mean()
    
    cv_scores.append((selected_features, cv_score))  
    
    if cv_score > best_cv_score:
        best_cv_score = cv_score
        best_subset = selected_features

for selected_features, score in cv_scores:
    print(f"Selected features: {selected_features} - CV Score: {score}")

print("\nBest subset of features by CV:", best_subset)
print("Best CV score with selected features:", best_cv_score)

# Define models and hyperparameter grids
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

param_grids = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "subsample": [0.8, 1.0]
    }
}

best_models = {}

# Perform hyperparameter tuning
for model_name, model in models.items():
    print(f"\nPerforming GridSearchCV for {model_name}...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        scoring="f1", 
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train[best_subset], y_train)

    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score for {model_name}: {grid_search.best_score_}")

# Evaluate models on the test set
for model_name, model in best_models.items():
    print(f"\nEvaluating {model_name} on test set...")
    predictions = model.predict(X_test[best_subset])
    y_proba = model.predict_proba(X_test[best_subset])[:, 1]
    print(f"{model_name} Classification Report:\n", classification_report(y_test, predictions))
    print(f"{model_name} Accuracy:", accuracy_score(y_test, predictions))
    print(f"{model_name} F1 Score:", f1_score(y_test, predictions))
    print(f"{model_name} AUC Score:", roc_auc_score(y_test, y_proba))
    print(f"{model_name} MCC:", matthews_corrcoef(y_test, predictions))
