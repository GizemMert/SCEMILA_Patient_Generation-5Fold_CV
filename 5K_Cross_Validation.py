import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# Set paths
data_path = "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/data/data"
metadata_path = "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/data/metadata.csv"
output_path = "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/Folds"

# Define classes
classes = ["CBFB_MYH11", "control", "NPM1", "PML_RARA", "RUNX1_RUNX1T1"]

# Read metadata
metadata = pd.read_csv(metadata_path)

# Collect data
data = []
labels = []
for label in classes:
    class_path = os.path.join(data_path, label)
    patients = os.listdir(class_path)
    data.extend([os.path.join(class_path, patient) for patient in patients])
    labels.extend([label] * len(patients))

# Convert to DataFrame
df = pd.DataFrame({'path': data, 'label': labels})
df['patient_id'] = df['path'].apply(lambda x: os.path.basename(x))

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create output folders
Path(output_path).mkdir(parents=True, exist_ok=True)

for fold, (train_idx, test_idx) in enumerate(skf.split(df['path'], df['label'])):
    print(f"Fold {fold}:")
    print(f"  Train: index={train_idx}")
    print(f"  Test:  index={test_idx}")

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    fold_path = os.path.join(output_path, f'fold_{fold}')
    Path(fold_path).mkdir(parents=True, exist_ok=True)

    train_path = os.path.join(fold_path, 'train')
    test_path = os.path.join(fold_path, 'test')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(test_path).mkdir(parents=True, exist_ok=True)

    # Copy training directories
    for train_file in train_df['path']:
        label = train_df[train_df['path'] == train_file]['label'].values[0]
        dest_folder = os.path.join(train_path, label, os.path.basename(train_file))
        Path(os.path.join(train_path, label)).mkdir(parents=True, exist_ok=True)
        shutil.copytree(train_file, dest_folder)

    # Copy testing directories
    for test_file in test_df['path']:
        label = test_df[test_df['path'] == test_file]['label'].values[0]
        dest_folder = os.path.join(test_path, label, os.path.basename(test_file))
        Path(os.path.join(test_path, label)).mkdir(parents=True, exist_ok=True)
        shutil.copytree(test_file, dest_folder)

    # Save metadata for train and test sets
    train_metadata = metadata[metadata['patient_id'].isin(train_df['patient_id'])]
    test_metadata = metadata[metadata['patient_id'].isin(test_df['patient_id'])]

    train_metadata.to_csv(os.path.join(train_path, 'metadata.csv'), index=False)
    test_metadata.to_csv(os.path.join(test_path, 'metadata.csv'), index=False)

    print(f"Fold {fold} created successfully with train and test metadata.")

print("All folds created successfully.")
