#pore forming proteins - dl dataset, equal samples from all classes', only esm - no feature selection

import pandas as pd 
import numpy as np
from Bio import SeqIO
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,precision_score,recall_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
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
        # aa_df = calculate_aa_composition(fasta_file)
        # dp_df = calculate_dipeptide_composition(fasta_file)
        # combined_df = pd.merge(aa_df, dp_df, on='sequence_id', suffixes=('_aa', '_dp'))

        # pseaac_df = pd.read_csv(pseaac_path)
        # pseaac_df['sequence_id'] = pseaac_df['sequence_id'].str.lstrip('>')
        # combined_df = pd.merge(combined_df, pseaac_df, on='sequence_id', how='inner')

        # physiochem_df = pd.read_csv(physiochem_path)
        # physiochem_df['sequence_id'] = physiochem_df['sequence_id'].str.lstrip('>')
        # combined_df = pd.merge(combined_df, physiochem_df, on='sequence_id', how='inner')

        esm_df = pd.read_csv(esm_path)
        esm_df['sequence_id'] = esm_df['sequence_id'].str.lstrip('>')
        # combined_df = pd.merge(combined_df, esm_df, on='sequence_id', how='inner')

        esm_df['label'] = label
        combined_df_list.append(esm_df)

    full_df = pd.concat(combined_df_list, axis=0, ignore_index=True)

    # Split into train and test data based on sequence IDs
    train_df = full_df[full_df['sequence_id'].isin(train_ids)]
    test_df = full_df[full_df['sequence_id'].isin(test_ids)]

    return train_df, test_df

# fasta files
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




all_fasta_files = positive_fasta_files + negative_fasta_files
all_pseaac_paths = positive_pseaac_paths + negative_pseaac_path
all_esm_embeddings_paths = positive_esm_embeddings_paths + negative_esm_embeddings_path
all_physiochem_paths = positive_physiochem + negative_physiochem

# Create labels (1 for positive, 0 for negative)
all_labels = [1] * len(positive_fasta_files) + [0] * len(negative_fasta_files)

train_ids, test_ids = split_fasta_sequences(all_fasta_files)

# Load and merge all data
train_df, test_df = load_and_merge_data(all_fasta_files, all_pseaac_paths, all_physiochem_paths, all_esm_embeddings_paths, all_labels, train_ids=train_ids, test_ids=test_ids)



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

# # Perform Gini-based feature selection with RandomForest
# rf_feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_feature_selector.fit(X_train, y_train)


# feature_importances = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': rf_feature_selector.feature_importances_
# }).sort_values(by='importance', ascending=False)

# best_subset = []
# best_cv_score = 0
# step = 50

# for i in range(step, len(feature_importances) + 1, step):
#     selected_features = feature_importances['feature'].head(i).tolist()
    
#     # Performing cross-validation with the selected subset
#     cv_score = cross_val_score(RandomForestClassifier(random_state=42), 
#                                X_train[selected_features], y_train, cv=5, scoring='f1').mean()
    
#     # Tracking the best performing feature subset
#     if cv_score > best_cv_score:
#         best_cv_score = cv_score
#         best_subset = selected_features
    

# print("Best subset of features by CV:", best_subset)
# print("Best CV score with selected features:", best_cv_score)

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

    grid_search.fit(X_train, y_train)

    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score for {model_name}: {grid_search.best_score_}")

# Evaluate the best models on the test set
for model_name, model in best_models.items():
    print(f"\nEvaluating {model_name} with best parameters on test set...")
    predictions = model.predict(X_test)
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"{model_name} Accuracy:", accuracy_score(y_test, predictions))
    print(f"{model_name} F1 Score:", f1_score(y_test, predictions))
    print(f"{model_name} Precision:", precision_score(y_test, predictions))
    print(f"{model_name} Recall:", recall_score(y_test, predictions))
    print(f"{model_name} AUC Score:", roc_auc_score(y_test, predictions))
    print(f"{model_name} Matthews Correlation Coefficient:", matthews_corrcoef(y_test, predictions))


