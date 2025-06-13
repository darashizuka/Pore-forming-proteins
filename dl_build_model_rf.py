import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import pickle
from config_loader import load_config

config = load_config()

print("10 fold grid search")
feature_pickle_path = config['feature_pickle']
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

# Perform Grid Search and Cross-Validation
def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

# Main processing
train_df = df[df['is_Train'] == 1]
test_df = df[df['is_Train'] == 0]

representations = ['no_reduction', 'hydro', 'conform']
feature_types = ['counts', 'mat']
offset_combinations = [[1], [2], [3]]

results = []

for representation in representations:
    for feature_type in feature_types:
        if feature_type == 'mat':
            for offset_combination in offset_combinations:
                X_train, y_train = prepare_data(train_df, representation, feature_type, offset_combination)
                model = train_model(X_train, y_train)
                X_test, y_test = prepare_data(test_df, representation, feature_type, offset_combination)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                results.append([representation, feature_type, str(offset_combination)] + list(metrics.values()))
        else:
            X_train, y_train = prepare_data(train_df, representation, feature_type, [])
            model = train_model(X_train, y_train)
            X_test, y_test = prepare_data(test_df, representation, feature_type, [])
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            results.append([representation, feature_type, "N/A"] + list(metrics.values()))

# Save results to CSV
columns = ['Representation', 'Feature Type', 'Offsets', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC', 'MCC']
pd.DataFrame(results, columns=columns).to_csv("all_features_models.csv", index=False)
print("Results saved")
