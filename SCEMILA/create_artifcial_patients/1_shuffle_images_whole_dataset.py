import numpy as np
import glob
import os
import random
import re
from collections import Counter
import pandas as pd
import pickle

class_labels = ["CBFB_MYH11", "control", "NPM1", "PML_RARA", "RUNX1_RUNX1T1"]


# Function to get list of image_paths in one folder
def get_image_path_list(folder_path):
    tif_files = glob.glob(f"{folder_path}/*.tif")
    return tif_files


"""
shuffle list into n new lists:
num_lists: number of new lists
items_per_list: item per each new list
"""


def shuffle_into_lists(original_list, num_lists, items_per_list, seed=4):
    # Shuffle the original list in-place
    random.seed(seed)
    random.shuffle(original_list)
    total_items = len(original_list)

    # Check if the specified number of lists and items per list are valid
    if num_lists * items_per_list > total_items:
        raise ValueError("Invalid parameters: Not enough items in the original list.")

    result_lists = [original_list[i:i + items_per_list] for i in range(0, num_lists * items_per_list, items_per_list)]
    return result_lists


# Extracts the number of image in the file_path e.g. "image_123.tif"
def extract_number_image(file_path):
    # Use a regular expression to find the number after "image_" and before ".tif"
    match = re.search(r'image_(\d+).tif', file_path)
    return int(match.group(1))


# Get the most common class label in a list of file_paths
def get_most_common_class(file_paths):
    class_labels = [re.search(datafile_path + r'/data/(\w+)/', path).group(1) for path in file_paths]
    class_counts = Counter(class_labels)
    most_common_class = random.choice(class_counts.most_common())[0]
    return most_common_class


datafile_path = "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data"
parent_folder = datafile_path + "/data"
image_path_list = []

for folder_class in os.listdir(parent_folder):
    folder_class = os.path.join(parent_folder, folder_class)

    if os.path.isdir(folder_class):
        print(folder_class)
        for folder_patient in os.listdir(folder_class):
            folder_patient = os.path.join(folder_class, folder_patient)
            if os.path.isdir(folder_patient):
                image_path_list += get_image_path_list(folder_patient)

print(f"Number of all images {len(image_path_list)}")

# Shuffle into n patients with n images
n_patients = 10
n_images = 10
experiment_name = "experiment_1"
output_folder = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata/' + experiment_name + '/data'

shuffled_patients = shuffle_into_lists(image_path_list, n_patients, n_images, seed=4)
patient_classes = []

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the shuffled lists
for i, file_paths in enumerate(shuffled_patients):
    most_common_class = get_most_common_class(file_paths)
    output_file_path = os.path.join(output_folder + "/" + most_common_class, f"patient{i}")
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    print(f"Save Patient {i + 1} in class {most_common_class}")
    patient_classes.append(most_common_class)
    file_paths = sorted(file_paths)
    with open(output_file_path + "/images.txt", 'w') as file:
        for path in file_paths:
            file.write(path + '\n')

for class_label in class_labels:
    full_path = os.path.join(output_folder, class_label)

    if os.path.exists(full_path) and os.path.isdir(full_path):
        contents = os.listdir(full_path)
        num_patients = sum(os.path.isdir(os.path.join(full_path, item)) for item in contents)

        # Print the result
        print(f"Class {class_label} contains {num_patients} patients.")
    else:
        print(f"Path {full_path} does not exist or is not a directory.")

for patient, filepath_images_list in enumerate(shuffled_patients):
    print(f"Save patient {patient + 1} features")
    array_list = []
    previous_patient_id = None
    for filepath in filepath_images_list:
        patient_id = filepath[:filepath.find("/image")]
        if previous_patient_id != patient_id:
            features = np.load(patient_id + "/fnl34_bn_features_layer_7.npy")
        array_list.append([features[extract_number_image(filepath)]])
        previous_patient_id = patient_id
    # Concatenate all features for one artificial patient
    artificial_features = np.concatenate(array_list, axis=0)
    patient_class = patient_classes[patient]
    output_npy_file = output_folder + f"/{patient_class}/patient{patient}/fnl34_bn_features_layer_7.npy"
    # Save the array to the .npy file
    np.save(output_npy_file, artificial_features)

# Save metadata file
columns = ['patient_id', 'sex_1f_2m', 'age', 'bag_label', 'instance_count',
           'leucocytes_per_¬µl', 'pb_myeloblast', 'pb_promyelocyte',
           'pb_myelocyte', 'pb_metamyelocyte', 'pb_neutrophil_band',
           'pb_neutrophil_segmented', 'pb_eosinophil', 'pb_basophil',
           'pb_monocyte', 'pb_lymph_typ', 'pb_lymph_atyp_react',
           'pb_lymph_atyp_neopl', 'pb_other', 'pb_total']
artifcialmetadata = pd.DataFrame(columns=columns)
artifcialmetadata['patient_id'] = [f"patient{i}" for i in range(n_patients)]
artifcialmetadata['bag_label'] = patient_classes
artifcialmetadata['instance_count'] = n_images
artifcialmetadata.to_csv(os.path.dirname(output_folder) + '/metadata.csv', index=False)

experiment_1_directory = output_folder

# Iterate through each AML subtype
for aml_subtype in class_labels:
    class_folder = os.path.join(experiment_1_directory, aml_subtype)

    # Check if the input folder exists
    if os.path.exists(class_folder):
        # Find the number of patients in the input folder
        patient_folders = [folder for folder in os.listdir(class_folder) if folder.startswith("patient")]

        # Iterate through each patient
        for patient_folder in patient_folders:
            patient_number = int(patient_folder[len("patient"):])  # Extract the patient number from the folder name
            input_patient_folder = os.path.join(class_folder, patient_folder)

            # Read the existing images.txt file
            txt_file_path = os.path.join(input_patient_folder, 'images.txt')
            with open(txt_file_path, 'r') as txt_file:
                image_file_paths = [line.strip() for line in txt_file.readlines()]

            # Save the image_file_paths using pickle
            with open(os.path.join(input_patient_folder, 'image_file_paths'), 'wb') as fp:
                pickle.dump(image_file_paths, fp)

