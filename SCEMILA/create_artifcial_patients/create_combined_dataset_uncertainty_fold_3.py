import os
import re
import torch
import torch.nn.functional as F
import sys
module_path = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/SCEMILA/ml_pipeline'
if module_path not in sys.path:
    sys.path.append(module_path)
from dataset_mixed import *  # dataset
from model import *  # actual MIL model
from sklearn import metrics as metrics
import csv
import shutil
import pandas as pd
import numpy as np
import pickle

CLASSES = ['PML_RARA',
        'NPM1',
        'CBFB_MYH11',
        'RUNX1_RUNX1T1',
        'control']
num_classes = 5
seed = 42
experiment_source = 'experiment_3'
real_data_source = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/Folds/fold_3/train/data'
SOURCE_FOLDER = f'/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata_fold_3/'+experiment_source
TARGET_FOLDER = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/result_fold_3'
output_folder = f'/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/mixed_uncertain_fold_3_seed{seed}'

def get_patient_name(path):
    return re.search(r"/data/\w+/([0-9a-zA-Z_]*)", path).group(1)

def get_class_name(path):
    return re.search(r"/data/(\w+)", path).group(1)

# Load patient data
patients = {}
with open(os.path.join(SOURCE_FOLDER, 'metadata.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        key = line[0] + "_" + line[3]
        patients[key] = [os.path.join(SOURCE_FOLDER, "data", line[3], line[0], "fnl34_bn_features_layer_7.npy"), line[3]]

# Function to update misclassification count
def update_misclassification_count(probability_vector, one_hot_target, current_misclassification_count):
    one_hot_prediction = torch.zeros_like(probability_vector)
    one_hot_prediction[0, torch.argmax(probability_vector).item()] = 1
    if torch.argmax(one_hot_prediction).item() != torch.argmax(one_hot_target).item():
        current_misclassification_count += 1
    return current_misclassification_count

# Number of Monte Carlo samples
num_samples = 50

# Load class converter
class_converter = {}
with open(os.path.join(TARGET_FOLDER, 'class_conversion.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        class_converter[line[1]] = int(line[0])

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load(os.path.join(TARGET_FOLDER, "state_dictmodel.pt"), map_location="cpu")
model = model.to(device)
model.train()

# Initialize arrays to store uncertainties
all_uncertainties = {}
missclassification_counts = {}
max_uncertainties = {}
sum_uncertainties = {}

# Perform Monte Carlo Dropout
with torch.no_grad():
    for p in patients.keys():
        pred = []
        missclassification_count = 0
        for j in range(num_samples):
            path, lbl_name = patients[p]
            lbl = np.zeros(5)
            lbl[class_converter[lbl_name]] = 1

            bag = np.load(path)
            bag = torch.tensor(bag).to(device)
            bag = torch.unsqueeze(bag, 0)
            prediction, _, _, _, = model(bag)
            pred.append(F.softmax(prediction, dim=1).cpu().detach().numpy())

            missclassification_count = update_misclassification_count(F.softmax(prediction, dim=1), torch.tensor(lbl), missclassification_count)

        pred_tensor = [torch.from_numpy(arr) for arr in pred]
        pred_tensor = torch.stack(pred_tensor)

        mean_prediction = pred_tensor.mean(dim=0)
        uncertainty = pred_tensor.std(dim=0)

        uncertainty_value_max = torch.max(uncertainty).item()
        uncertainty_value_sum = torch.sum(uncertainty).item()
        path = os.path.dirname(path)

        max_uncertainties[p] = {'path': path, 'data': uncertainty.cpu().numpy().squeeze(), 'uncertainty': uncertainty_value_max}
        sum_uncertainties[p] = {'path': path, 'data': uncertainty.cpu().numpy().squeeze(), 'uncertainty': uncertainty_value_sum}
        missclassification_counts[p] = {'path': path, 'uncertainty': missclassification_count / num_samples}

def sort_and_print(uncertainties):
    sorted_uncertainties = dict(sorted(uncertainties.items(), key=lambda item: item[1]['uncertainty'], reverse=True))
    for p, data in sorted_uncertainties.items():
        print(f"Patient {p}: Uncertainty - {data['uncertainty']:.4}")

sort_and_print(max_uncertainties)

print(len(max_uncertainties.keys()))
print(len(set(max_uncertainties.keys())))

def select_paths(uncertainties, percentage):
    sorted_uncertainties = dict(sorted(uncertainties.items(), key=lambda item: item[1]['uncertainty'], reverse=True))
    num_paths = int(len(sorted_uncertainties) * (percentage / 100.0))
    selected_paths = {p: data['path'] for p, data in list(sorted_uncertainties.items())[:num_paths]}
    return selected_paths

metadata_path = os.path.join(SOURCE_FOLDER, "metadata.csv")

def create_metadata(selected_paths, target_folder):
    metadata_df = pd.read_csv(metadata_path)
    metadata_dict = {}
    for key in selected_paths:
        path = selected_paths[key]
        match = re.search(r'\d', key)
        if match:
            index_of_second_digit = key.find('_', match.start() + 1)
            if index_of_second_digit != -1:
                patient_id = key[:index_of_second_digit + 1]
                patient_id = patient_id[:-1]
                bag_label = key[index_of_second_digit + 1:]
            else:
                patient_id = key
                bag_label = ''
        else:
            patient_id = key
            bag_label = ''
        metadata_row = metadata_df[(metadata_df['patient_id'] == patient_id) & (metadata_df['bag_label'] == bag_label)]
        if not metadata_row.empty:
            metadata_dict[key] = metadata_row.iloc[0].to_dict()
        else:
            print(f"No metadata found for {key}: {path}")

    new_metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index')
    new_metadata_path = os.path.join(target_folder, 'metadata_uncertain_patients.csv')
    new_metadata_df.to_csv(new_metadata_path, index=False)
    return new_metadata_path

def get_realpatients_filepaths_dictionary(src_folder):
    paths = {}
    for class_name in os.listdir(src_folder):
        class_path = os.path.join(src_folder, class_name)
        for file in os.listdir(class_path):
            src_path = os.path.join(class_path, file)
            patient_name = get_patient_name(src_path)
            if class_name not in paths.keys():
                paths[class_name] = []
            paths[class_name].append(src_path)
    return paths

import copy

def save_patient_filepaths(selected_paths, new_folder, paths_real_patients):
    print(f"Save file paths for uncertain patients in {new_folder}")
    os.makedirs(new_folder, exist_ok=True)
    paths_uncertain_patients = {}
    for class_name in CLASSES:
        if class_name not in paths_uncertain_patients.keys():
            paths_uncertain_patients[class_name] = []

    for p, path in selected_paths.items():
        class_name = get_class_name(path)
        paths_uncertain_patients[class_name].append(path)

    paths_mixed_patients = copy.deepcopy(paths_real_patients)
    for key, value in paths_mixed_patients.items():
        paths_mixed_patients[key] += paths_uncertain_patients[key]
    for key in paths_mixed_patients.keys():
        len_before = len(paths_mixed_patients[key])
        paths_mixed_patients[key] = list(set(paths_mixed_patients[key]))
        print(f"Removed {len_before - len(paths_mixed_patients[key])} duplicates")
    with open(new_folder + '/file_paths.pkl', 'wb') as f:
        pickle.dump(paths_mixed_patients, f)

# Iterate over different percentages from 10 to 50 and save uncertain patients
paths_real_patients = get_realpatients_filepaths_dictionary(real_data_source)
for percentage in [10, 20, 30, 50]:
    new_folder_max = output_folder + f'/max_{percentage}_percent'
    selected_max_paths = select_paths(max_uncertainties, percentage)
    save_patient_filepaths(selected_max_paths, new_folder_max, paths_real_patients)
    create_metadata(selected_max_paths, new_folder_max)

def concatenate_metadata(original_metadata_path, uncertain_patients_folder):
    original_metadata_df = pd.read_csv(original_metadata_path)
    for folder in os.listdir(uncertain_patients_folder):
        folder_path = os.path.join(uncertain_patients_folder, folder)
        if os.path.isdir(folder_path):
            metadata_file_path = os.path.join(folder_path, 'metadata_uncertain_patients.csv')
            print("metadata is:")
            print(metadata_file_path)
            if os.path.exists(metadata_file_path):
                uncertain_metadata_df = pd.read_csv(metadata_file_path)
                concatenated_metadata_df = pd.concat([original_metadata_df, uncertain_metadata_df], ignore_index=True)
                output_folder_new = uncertain_patients_folder + f'/{folder}'
                os.makedirs(output_folder_new, exist_ok=True)
                output_file_path = os.path.join(output_folder_new, 'metadata.csv')
                print("Output is:")
                print(output_file_path)
                concatenated_metadata_df.to_csv(output_file_path, index=False)

original_metadata_path = "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/Folds/fold_3/train/metadata.csv"
concatenate_metadata(original_metadata_path, output_folder)

print("Finished")
""""
# Check for duplicates
percentages = [10, 20, 30, 50]
for percentage in percentages:
    folder = f'/mnt/volume/shared/data_file/mixed_uncertain_fixbug_seed{seed}/max_{percentage}_percent'
    with open(folder + '/file_paths.pkl', 'rb') as f:
        file_paths_dict = pickle.load(f)
    for key, value_list in file_paths_dict.items():
        if len(value_list) != len(set(value_list)):
            print(f"Duplicates found for seed {seed}, {percentage}%, and key {key}. Duplicates: {set([item for item in value_list if value_list.count(item) > 1])}")
"""