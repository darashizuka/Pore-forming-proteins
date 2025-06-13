#only physic- basic models
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
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from config_loader import load_config
import os

config = load_config()

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


# Perform Gini-based feature selection with RandomForest
rf_feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_feature_selector.fit(X_train, y_train)


feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_feature_selector.feature_importances_
}).sort_values(by='importance', ascending=False)

best_subset = []
best_cv_score = 0
step = 5

for i in range(step, len(feature_importances) + 1, step):
    selected_features = feature_importances['feature'].head(i).tolist()
    
    # Performing cross-validation with the selected subset
    cv_score = cross_val_score(RandomForestClassifier(random_state=42), 
                               X_train[selected_features], y_train, cv=5, scoring='f1').mean()
    
    # Tracking the best performing feature subset
    if cv_score > best_cv_score:
        best_cv_score = cv_score
        best_subset = selected_features
    

print("Best subset of features by CV:", best_subset)
print("Best CV score with selected features:", best_cv_score)

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
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
    }
}


best_models = {}

for model_name, model in models.items():
    print(f"\nPerforming GridSearchCV for {model_name}...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        scoring='f1', 
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train[best_subset], y_train)

    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score for {model_name}: {grid_search.best_score_}")

# Evaluate the best models on the test set
for model_name, model in best_models.items():
    print(f"\nEvaluating {model_name} with best parameters on test set...")
    predictions = model.predict(X_test[best_subset])
    y_prob = model.predict_proba(X_test[best_subset])[:, 1]
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"{model_name} Accuracy:", accuracy_score(y_test, predictions))
    print(f"{model_name} F1 Score:", f1_score(y_test, predictions))
    print(f"{model_name} Precision:", precision_score(y_test, predictions))
    print(f"{model_name} Recall:", recall_score(y_test, predictions))
    print(f"{model_name} AUC Score:", roc_auc_score(y_test, y_prob))
    print(f"{model_name} Matthews Correlation Coefficient:", matthews_corrcoef(y_test, predictions))


