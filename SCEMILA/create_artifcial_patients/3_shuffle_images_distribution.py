import numpy as np
import glob
import os
import random
import re
from collections import Counter
import pandas as pd
import pickle

class_labels = ['PML_RARA',
        'NPM1',
        'CBFB_MYH11',
        'RUNX1_RUNX1T1',
        'control']
n_classes = len(class_labels)

#Function to get list of image_paths in one folder
def get_image_path_list(folder_path):
    tif_files = glob.glob(f"{folder_path}/*.tif")
    return tif_files


#exracts the number of image in the file_path e.g. "image_123.tif"
def extract_number_image(file_path):
    # Use a regular expression to find the number after "image_" and before ".tif"
    match = re.search(r'image_(\d+).tif', file_path)
    return int(match.group(1))

def get_patient_name(path):
    return re.search(r"/data/\w+/([A-Z]{3})", path).group(1)

def get_class_name(path):
    return re.search(r"/data/(\w+)", path).group(1)

def get_classification_patient(patient_folder):
    probs_path = patient_folder + '/single_cell_probabilities.npy'
    sc_probs = np.load(probs_path)
    sc_class= np.argmax(sc_probs, axis=1)
    return sc_class

data_directory = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/Folds/fold_2/train/data'
subtype = data_directory + "/NPM1"
n_patients = 30
experiment_name = "experiment_3"
output_folder = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata_fold_2/' + experiment_name + '/data'
output_folder_csv = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/artificialdata_fold_2/' + experiment_name

#Iterate over real dataset and store image paths in a dataframe df
df = pd.DataFrame(columns=["patient","AML_subtype", "SC_Label", "image_path"])
for folder_class in class_labels:
    folder_class = os.path.join(data_directory, folder_class)
    if os.path.isdir(folder_class):
       #print(folder_class)
       AML_subtype=get_class_name(folder_class)
       for folder_patient in os.listdir(folder_class):
            folder_patient = os.path.join(folder_class, folder_patient)
            if os.path.isdir(folder_patient):
                images=get_image_path_list(folder_patient)
                sc_classes=get_classification_patient(folder_patient)
                #print(sc_classes)
                for image in images:
                    number=extract_number_image(image)
                    df.loc[len(df)]=[get_patient_name(folder_patient), AML_subtype, sc_classes[number],image]

#calculate mean and std for each cell type that will be later used to sample data with normal distribution
sc_class_labels= ['eosinophil granulocyte', 'reactive lymphocyte',
       'neutrophil granulocyte (segmented)', 'typical lymphocyte',
       'other', 'neutrophil granulocyte (band)', 'monocyte',
       'large granulated lymphocyte', 'atypical promyelocyte',
       'basophil granulocyte', 'smudge cell', 'neoplastic lymphocyte',
       'promyelocyte', 'myelocyte', 'myeloblast', 'metamyelocyte',
       'normo', 'plasma cell', 'hair cell', 'bilobed M3v',
       'mononucleosis']
#If single_cell_results does not exist yet run notebook create_single_cell_results
df_sc_res=pd.read_csv("/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/Folds/fold_2/train/single_cell_results.csv",index_col=0).drop("patient", axis=1)
df_meanstd = df_sc_res.groupby(["AML_subtype"]).agg(["mean","std"])

#This cell creates artificial patients and stores the single cell couunts per patient in cell_type_counts_dict, also it counts the selected_images_per_patient as a sanity check
# Dictionary stores cell type counts (how often each cell type appears)
cell_type_counts_dict = {}

# Dictionary stores the selected images and counts them per SC class per patient
selected_images_per_patient = {}

# Iterate over all AML subtypes
for aml_subtype in class_labels:
    output_class_folder = output_folder + "/" + aml_subtype
    # Get distribution
    class_means = df_meanstd.loc[aml_subtype, :].loc[:, "mean"].values
    class_variances = df_meanstd.loc[aml_subtype, :].loc[:, "std"].values
    for patient_number in range(n_patients):
        print(f"Generating data for patient {patient_number+1} of subtype {aml_subtype}...")
        # Calculate how many images from each SC class
        generated_data = np.random.normal(class_means, class_variances, 21).astype(int)
        generated_data = generated_data * (generated_data > 0)
        image_file_paths = []
        # Randomly choose SC image and concatenate them into image_file_paths
        selected_images_count = {}
        for cell_type_number, cell_type in enumerate(sc_class_labels):
            df_cell_type = df[df["SC_Label"] == cell_type_number]
            # print(f"\tImages for cell type {cell_type}...")
            file_path = df_cell_type["image_path"].values
            image_paths = np.random.choice(file_path, size=generated_data[cell_type_number]).tolist()
            print(f"\t\tSelected {len(image_paths)} images for {cell_type}")
            image_file_paths += image_paths
            # Store selected images count per SC class
            selected_images_count[cell_type] = len(image_paths)
        # Store selected images count per SC class per patient
        patient_id = f"{aml_subtype}/patient_{patient_number+1}"
        selected_images_per_patient[patient_id] = selected_images_count
        # Store patient classes and number images for metadata
        new_patient_folder = os.path.join(output_class_folder, f'patient_{patient_number+1}')
        os.makedirs(new_patient_folder, exist_ok=True)

        # Sort the shuffled image paths for the current patient
        image_file_paths.sort()
        # Save the shuffled image paths into a text file
        txt_file_path = os.path.join(new_patient_folder, 'images.txt')
        with open(txt_file_path, 'w') as txt_file:
            for image_path in image_file_paths:
                txt_file.write(image_path + '\n')

        with open(os.path.join(new_patient_folder, "image_file_paths"), 'wb') as fp:
            pickle.dump(image_file_paths, fp)

        # Count cell types in the current patient
        cell_type_count = {cell_type: image_file_paths.count(cell_type) for cell_type in set(image_file_paths)}
        print(f"\tCell type count for patient {patient_number+1}: {cell_type_count}")

        # Add the cell type count to the dictionary with the patient as the key
        # and the cell type count as the value
        cell_type_counts_dict[(aml_subtype, patient_id)] = cell_type_count
        print(f"\tCell type counts dictionary for patient {patient_id}: {cell_type_count}")

# Print selected images count per SC class per patient
print("\nSelected Images Count per SC class per Patient:")
for patient_id, sc_counts in selected_images_per_patient.items():
    print(patient_id)
    for sc_class, count in sc_counts.items():
        print(f"\t{sc_class}: {count}")


#saving images in npy files
for aml_subtype in class_labels:
    output_class_folder=output_folder+"/"+aml_subtype
    for patient_number in range(n_patients):
        patient_folder = os.path.join(output_class_folder, f'patient_{patient_number+1}')
        print(patient_folder)
        with open( os.path.join(patient_folder,"image_file_paths") ,'rb') as fp:
            image_file_paths=pickle.load(fp)
        array_list=[]
        previous_patient_id=None
        # Iterate through each image path
        for image_path in image_file_paths:
            patient_id = image_path[:image_path.find("/image")]
            if previous_patient_id!=patient_id:
                #print(f"New patient: {patient_id}, old patient : {previous_patient_id}")
                features=np.load(patient_id+"/fnl34_bn_features_layer_7.npy")
            array_list.append([features[extract_number_image(image_path)]])
            previous_patient_id=patient_id
        #Concatenate all features for one artificial patient
        artificial_features = np.concatenate(array_list,axis=0)
        output_npy_file = output_folder+f"/{aml_subtype}/patient_{patient_number+1}/fnl34_bn_features_layer_7.npy"
        # Save the array to the .npy file
        np.save(output_npy_file, artificial_features)

# Create metadata including single cell types, not including age, gender and leucocytes_per_µl
# Create a list to hold the rows of the DataFrame
rows = []
'''sc_class_labels= ['eosinophil granulocyte', 'reactive lymphocyte',
       'neutrophil granulocyte (segmented)', 'typical lymphocyte',
       'other', 'neutrophil granulocyte (band)', 'monocyte',
       'large granulated lymphocyte', 'atypical promyelocyte',
       'basophil granulocyte', 'smudge cell', 'neoplastic lymphocyte',
       'promyelocyte', 'myelocyte', 'myeloblast', 'metamyelocyte',
       'normo', 'plasma cell', 'hair cell', 'bilobed M3v',
       'mononucleosis']'''

# Fill in the DataFrame with values from the dictionary
for patient_id, cell_counts in selected_images_per_patient.items():
    myeloblast_count = cell_counts.get('myeloblast', 0)
    # Extract counts for other cell types
    promyelocyte_count = cell_counts.get('promyelocyte', 0)
    myelocyte_count = cell_counts.get('myelocyte', 0)
    metamyelocyte_count = cell_counts.get('metamyelocyte', 0)
    neutrophil_band_count = cell_counts.get('neutrophil granulocyte (band)', 0)
    neutrophil_segmented_count = cell_counts.get('neutrophil granulocyte (segmented)', 0)
    eosinophil_count = cell_counts.get('eosinophil granulocyte', 0)
    basophil_count = cell_counts.get('basophil granulocyte', 0)
    monocyte_count = cell_counts.get('monocyte', 0)
    lymph_typ_count = cell_counts.get('typical lymphocyte', 0)
    lymph_atyp_react_count = cell_counts.get('reactive lymphocyte', 0)
    lymph_atyp_neopl_count = cell_counts.get('neoplastic lymphocyte', 0)
    other_count = cell_counts.get('other', 0)
    total_count = sum(cell_counts.values())  # check

    index = patient_id.find("/patient")
    bag = patient_id[:index] if index != -1 else patient_id

    id = patient_id.find("/")
    patient = patient_id[id + 1:] if index != -1 else ""

    age = 0  # dummy
    row = {'patient_id': patient,
           'sex_1f_2m': None,  # Placeholder value
           'age': age,
           'bag_label': bag,
           'instance_count': total_count,  # check
           'leucocytes_per_µl': None,  # Placeholder value
           'pb_myeloblast': round((myeloblast_count / total_count) * 100, 1),
           'pb_promyelocyte': round((promyelocyte_count / total_count) * 100, 1),
           'pb_myelocyte': round((myelocyte_count / total_count) * 100, 1),
           'pb_metamyelocyte': round((metamyelocyte_count / total_count) * 100, 1),
           'pb_neutrophil_band': round((neutrophil_band_count / total_count) * 100, 1),
           'pb_neutrophil_segmented': round((neutrophil_segmented_count / total_count) * 100, 1),
           'pb_eosinophil': round((eosinophil_count / total_count) * 100, 1),
           'pb_basophil': round((basophil_count / total_count) * 100, 1),
           'pb_monocyte': round((monocyte_count / total_count) * 100, 1),
           'pb_lymph_typ': round((lymph_typ_count / total_count) * 100, 1),
           'pb_lymph_atyp_react': round((lymph_atyp_react_count / total_count) * 100, 1),
           'pb_lymph_atyp_neopl': round((lymph_atyp_neopl_count / total_count) * 100, 1),
           'pb_other': round((other_count / total_count) * 100, 1),
           'pb_total': round((total_count / total_count) * 100, 1)}  # check
    rows.append(row)

# Create DataFrame from the list of rows
artificial_metadata = pd.DataFrame(rows)

# Save the DataFrame to a CSV file
metadata_csv_path = os.path.join(output_folder_csv, 'metadata.csv')
artificial_metadata.to_csv(metadata_csv_path, index=False)
print(f"Metadata saved to {metadata_csv_path}")
