import pandas as pd
import os

# Paths to the metadata files
metadata_train_path = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/metadata_train.csv'
metadata_test_path = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/metadata_test.csv'
combined_metadata_path = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/metadata_combined.csv'
target_path = '/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/F_AML/TCIA_data_prepared/data'

# Load the metadata files
metadata_train = pd.read_csv(metadata_train_path)
metadata_test = pd.read_csv(metadata_test_path)

# Combine the metadata files
combined_metadata = pd.concat([metadata_train, metadata_test])

# Shuffle the combined metadata
combined_metadata = combined_metadata.sample(frac=1).reset_index(drop=True)

# Save the combined and shuffled metadata file
combined_metadata.to_csv(combined_metadata_path, index=False)

# Copy the combined metadata file to the target directory
os.system(f'cp {combined_metadata_path} {target_path}')
