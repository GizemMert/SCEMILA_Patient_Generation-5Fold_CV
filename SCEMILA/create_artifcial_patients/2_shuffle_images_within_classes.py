import os
import pickle
import numpy as np
import glob
import random
import re
from collections import Counter
import pandas as pd

class_labels = ["CBFB_MYH11","control","NPM1","PML_RARA","RUNX1_RUNX1T1"]
n_classes = len(class_labels)

def get_image_path_list(folder_path):
    tif_files = glob.glob(f"{folder_path}/*.tif")
    return tif_files

"""
shuffle list into n new lists:
num_lists: number of new lists
items_per_list: item per each new list
"""

def shuffle_into_lists(original_list, num_lists, items_per_list, seed=4):
    random.seed(seed)
    random.shuffle(original_list)
    total_items = len(original_list)
    if num_lists * items_per_list > total_items:
        raise ValueError("Invalid parameters: Not enough items in the original list.")
    result_lists = [original_list[i:i+items_per_list] for i in range(0, num_lists*items_per_list, items_per_list)]
    return result_lists

def extract_number_image(file_path):
    match = re.search(r'image_(\d+).tif', file_path)
    return int(match.group(1))

n_patients = 5
n_images = 10
experiment_name = "experiment_2"

data_directory = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/data'
output_folder = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata/'+experiment_name+'/data'
print("Output Folder:", output_folder)

class_folders = [folder for folder in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, folder))]
shuffled_images = {}

for class_folder in class_labels:
    class_path = os.path.join(data_directory, class_folder)
    print(class_path)
    patient_folders = [folder for folder in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, folder))]
    class_image_paths = []
    for patient_folder in patient_folders:
        patient_path = os.path.join(class_path, patient_folder)
        print(patient_path)
        image_paths = get_image_path_list(patient_path)
        class_image_paths.extend(image_paths)
    shuffled_lists = shuffle_into_lists(class_image_paths, n_patients, n_images)
    shuffled_images[class_folder] = shuffled_lists

print(shuffled_images)

patient_classes = []
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for class_folder, shuffled_lists in shuffled_images.items():
    output_class_folder = os.path.join(output_folder, class_folder)
    print("Output Class Folder:", output_class_folder)
    os.makedirs(output_class_folder, exist_ok=True)
    for i, shuffled_patient_folder in enumerate(shuffled_lists):
        new_patient_folder = os.path.join(output_class_folder, f'patient_{i+1}')
        os.makedirs(new_patient_folder, exist_ok=True)
        patient_classes.append(class_folder)
        shuffled_patient_folder.sort()
        txt_file_path = os.path.join(new_patient_folder, 'images.txt')
        print(txt_file_path)
        with open(txt_file_path, 'w') as txt_file:
            for image_path in shuffled_patient_folder:
                txt_file.write(image_path + '\n')

for class_folder, shuffled_lists in shuffled_images.items():
    for patient, shuffled_patient_folder in enumerate(shuffled_lists):
        print(f"Save patient {patient + 1} features")
        array_list = []
        previous_patient_id = None
        for image_path in shuffled_patient_folder:
            patient_id = image_path[:image_path.find("/image")]
            if previous_patient_id != patient_id:
                features = np.load(os.path.join(patient_id, "fnl34_bn_features_layer_7.npy"))
                previous_patient_id = patient_id
            array_list.append(features[extract_number_image(image_path)])
        artificial_features = np.stack(array_list, axis=0)
        output_npy_folder = os.path.join(output_folder, class_folder, f"patient_{patient+1}")
        os.makedirs(output_npy_folder, exist_ok=True)
        output_npy_file = os.path.join(output_npy_folder, "fnl34_bn_features_layer_7.npy")
        np.save(output_npy_file, artificial_features)

columns = ['patient_id', 'sex_1f_2m', 'age', 'bag_label', 'instance_count',
           'leucocytes_per_µl', 'pb_myeloblast', 'pb_promyelocyte',
           'pb_myelocyte', 'pb_metamyelocyte', 'pb_neutrophil_band',
           'pb_neutrophil_segmented', 'pb_eosinophil', 'pb_basophil',
           'pb_monocyte', 'pb_lymph_typ', 'pb_lymph_atyp_react',
           'pb_lymph_atyp_neopl', 'pb_other', 'pb_total']

artifcialmetadata = pd.DataFrame(columns=columns)
artifcialmetadata['patient_id'] = [f"patient{i%5 + 1}" for i in range(n_patients*n_classes)]
artifcialmetadata['bag_label'] = patient_classes
artifcialmetadata['instance_count'] = n_images
artifcialmetadata.to_csv(os.path.dirname(output_folder)+'/metadata.csv', index=False)

experiment_1_directory = output_folder

for aml_subtype in class_labels:
    class_folder = os.path.join(experiment_1_directory, aml_subtype)
    if os.path.exists(class_folder):
        patient_folders = [folder for folder in os.listdir(class_folder) if folder.startswith("patient_")]
        for patient_folder in patient_folders:
            patient_number = int(patient_folder[len("patient_"):])
            input_patient_folder = os.path.join(class_folder, patient_folder)
            txt_file_path = os.path.join(input_patient_folder, 'images.txt')
            with open(txt_file_path, 'r') as txt_file:
                image_file_paths = [line.strip() for line in txt_file.readlines()]
            with open(os.path.join(input_patient_folder, 'image_file_paths'), 'wb') as fp:
                pickle.dump(image_file_paths, fp)
