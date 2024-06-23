import random
import os
import re
import shutil
import pandas as pd
import numpy as np
import pickle

# Seed initialization
seed = 20
random.seed(seed)
np.random.seed(seed)

def copyfiles_real(src_folder, dest_folder):
    paths = {}
    patients_per_folder = {}
    for class_name in os.listdir(src_folder):
        class_path = os.path.join(src_folder, class_name)
        for file in os.listdir(class_path):
            src_path = os.path.join(class_path, file)
            patient_name = get_patient_name(src_path)
            folder_path = os.path.join(dest_folder, class_name)
            patients_per_folder[folder_path] = patients_per_folder.get(folder_path, 0) + 1
            if class_name not in paths.keys():
                paths[class_name] = []
            paths[class_name].append(src_path)
    return patients_per_folder, paths

def get_patient_name(path):
    return re.search(r"/data/\w+/([0-9a-zA-Z_]*)", path).group(1)

def get_class_name(path):
    return re.search(r"/data/(\w+)", path).group(1)

def copyfiles_art(src_folder, dest_folder, patient_counts):
    selected_patients = []
    class_names = sorted(os.listdir(src_folder))
    paths = {}
    for class_index, class_name in enumerate(class_names):
        if class_name not in patient_counts:
            break
        patients = os.listdir(os.path.join(src_folder, class_name))
        num_patients = min(patient_counts[class_name], len(patients))
        selected_patient_names = random.sample(patients, num_patients)
        for patient_name in selected_patient_names:
            selected_patients.append((patient_name, class_name))
            patient_path = os.path.join(src_folder, class_name, patient_name)
            if class_name not in paths.keys():
                paths[class_name] = []
            paths[class_name].append(patient_path)
    return selected_patients, paths

# Specify your source and destination folders
src_data_folder = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/data'
src_artificialdata_folder = f'/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata/experiment_3/data'
mixeddata_folder = f'/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/mixed_seed{seed}'
percentage = [10, 20, 30, 50]

for p in percentage:
    folder_name = f"{p}_percent"
    dest_mixeddata_folder = os.path.join(mixeddata_folder, folder_name)
    os.makedirs(dest_mixeddata_folder, exist_ok=True)
    art_percentage = p
    patients_per_folder, paths_real_patients = copyfiles_real(src_data_folder, dest_mixeddata_folder)

    num_art_patient = {}
    for key, value in patients_per_folder.items():
        folder_name = os.path.basename(key)
        num_art_patient[folder_name] = round((value * art_percentage) / (100 - art_percentage))

    selected_patients, paths_artificial_patients = copyfiles_art(src_artificialdata_folder, dest_mixeddata_folder, num_art_patient)

    paths_mixed_patients = {}
    for key, value in paths_real_patients.items():
        paths_mixed_patients[key] = value
        paths_mixed_patients[key] += paths_artificial_patients.get(key, [])

    df1 = pd.read_csv("/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/metadata.csv")
    df2 = pd.read_csv(os.path.dirname(src_artificialdata_folder) + "/metadata.csv")

    selected_patients_df = pd.DataFrame(selected_patients, columns=['patient_id', 'bag_label'])
    filtered_df2 = df2.merge(selected_patients_df, on=['patient_id', 'bag_label'], how='inner')

    conc_df = pd.concat([df1, filtered_df2], ignore_index=True)

    conc_df.to_csv(dest_mixeddata_folder + '/metadata.csv', index=False)
    with open(dest_mixeddata_folder + '/file_paths.pkl', 'wb') as f:
        pickle.dump(paths_mixed_patients, f)

    print("Experiment is done!")
