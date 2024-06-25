import os.path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from dataset import *  # dataset
from model import *  # actual MIL model
from sklearn import metrics as metrics
import csv
import os
import shutil
import numpy as np

# Configuration
CLASSES = ['control', 'RUNX1_RUNX1T1', 'NPM1', 'CBFB_MYH11', 'PML_RARA']
num_classes = 5

SOURCE_FOLDER = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata/experiment_3'
TARGET_FOLDER = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/result_folder_1'

# Load patient data
patients = {}
with open(os.path.join(SOURCE_FOLDER, 'metadata.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        key = line[0] + "_" + line[3]
        patients[key] = [os.path.join(SOURCE_FOLDER, "data", line[3], line[0], "fnl34_bn_features_layer_7.npy"), line[3]]


print(patients)

# Function to update misclassification count
def update_misclassification_count(probability_vector, one_hot_target, current_misclassification_count):
    one_hot_prediction = torch.zeros_like(probability_vector)
    one_hot_prediction[0, torch.argmax(probability_vector).item()] = 1

    target_index = torch.argmax(one_hot_target).item()

    if torch.argmax(one_hot_prediction).item() != target_index:
        current_misclassification_count += 1

    return current_misclassification_count

# Number of Monte Carlo samples
num_samples = 10

class_converter = {}
with open(os.path.join(TARGET_FOLDER, 'class_conversion.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        class_converter[line[1]] = int(line[0])

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

        max_uncertainties[p] = {'path': path, 'data': uncertainty.cpu().numpy().squeeze(), 'uncertainty': uncertainty_value_max}
        sum_uncertainties[p] = {'path': path, 'data': uncertainty.cpu().numpy().squeeze(), 'uncertainty': uncertainty_value_sum}
        missclassification_counts[p] = {'path': path, 'uncertainty': missclassification_count / num_samples}

# Function to sort and print uncertainties
def sort_and_print(uncertainties):
    sorted_uncertainties = dict(sorted(uncertainties.items(), key=lambda item: item[1]['uncertainty'], reverse=True))
    for p, data in sorted_uncertainties.items():
        uncertainty_rate = data['uncertainty']
        print(f"Patient {p}: Uncertainty - {uncertainty_rate:.4}")

sort_and_print(max_uncertainties)
sort_and_print(sum_uncertainties)
sort_and_print(missclassification_counts)

# Function to select paths based on uncertainties
def select_paths(uncertainties, criterion):
    sorted_uncertainties = dict(sorted(uncertainties.items(), key=lambda item: item[1]['uncertainty'], reverse=True))
    selected_paths = {p: data['path'] for p, data in list(sorted_uncertainties.items())[:criterion]}
    return selected_paths

# Function to save uncertain patients
def save_uncertain_patients(selected_paths, new_folder):
    for p, path in selected_paths.items():
        extraction_index = path.find('fnl34_bn_features_layer_7.npy')
        data_index = path.find('/artificialdata/experiment_3_seed1_41/data/') + len('/artificialdata/experiment_3_seed1_41')
        extracted_path = path[data_index:extraction_index-1]
        source_path = path[:extraction_index-1]

        target_folder = new_folder + extracted_path
        os.makedirs(target_folder, exist_ok=True)

        additional_files = ["image_file_paths", "images.txt"]
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file in additional_files or file.endswith("fnl34_bn_features_layer_7.npy"):
                    source_file = os.path.join(root, file)
                    relative_path = os.path.relpath(source_file, source_path)
                    destination_file = os.path.join(target_folder, relative_path)
                    os.makedirs(os.path.dirname(destination_file), exist_ok=True)
                    shutil.copy2(source_file, destination_file)

# Save uncertain patients
new_folder = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata/uncertain_patients_misscl'
selected_misscl_paths = select_paths(missclassification_counts, 10)
save_uncertain_patients(selected_misscl_paths, new_folder)

new_folder = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata/uncertain_patients_max'
selected_max_paths = select_paths(max_uncertainties, 10)
save_uncertain_patients(selected_max_paths, new_folder)

new_folder = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata/uncertain_patients_sum'
selected_sum_paths = select_paths(sum_uncertainties, 10)
save_uncertain_patients(selected_sum_paths, new_folder)
