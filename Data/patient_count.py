import os

import pickle

pkl_file_path = "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/mixed_uncertain_fold_1_seed42/max_20_percent/file_paths.pkl"

# Load the pkl file
with open(pkl_file_path, 'rb') as file:
    file_paths = pickle.load(file)

# Count the number of patient names in each class
for class_name, paths in file_paths.items():
    num_patients = len(paths)
    print(f"Class {class_name}: {num_patients} patient(s)")

# Count the total number of patient names across all classes
total_file_paths = sum(len(paths) for paths in file_paths.values())
print(f"Total number of patient names: {total_file_paths}")

print(f"Total number of file paths: {total_file_paths}")
"""
def count_subfolders(main_folder):
    subfolders_count = {}

    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            count = len([name for name in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, name))])
            subfolders_count[subfolder] = count

    return subfolders_count

def main():
    main_folder = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/mixed_uncertain_fold_0_seed42/max_20_percent'
    subfolders_count = count_subfolders(main_folder)

    total_count = 0
    for subfolder, count in subfolders_count.items():
        print(f"{subfolder}: {count} subfolders")
        total_count += count

    print(f"Total number of subfolders: {total_count}")

if __name__ == "__main__":
    main()
"""
