import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import pandas as pd
import re
import seaborn as sns


def get_counts_vector(labels_vector):
    unique_labels, label_counts = np.unique(labels_vector, return_counts=True)
    counts_vector = np.zeros(21, dtype=int)
    counts_vector[unique_labels] = label_counts
    return counts_vector, unique_labels

sc_class_labels= ['eosinophil granulocyte', 'reactive lymphocyte',
       'neutrophil granulocyte (segmented)', 'typical lymphocyte',
       'other', 'neutrophil granulocyte (band)', 'monocyte',
       'large granulated lymphocyte', 'atypical promyelocyte',
       'basophil granulocyte', 'smudge cell', 'neoplastic lymphocyte',
       'promyelocyte', 'myelocyte', 'myeloblast', 'metamyelocyte',
       'normo', 'plasma cell', 'hair cell', 'bilobed M3v',
       'mononucleosis']

aml_class_labels = ["CBFB_MYH11","control","NPM1","PML_RARA","RUNX1_RUNX1T1"]
# Path to the folder containing your files

data_path = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/Folds/fold_4/train/data'
result_path = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/Folds/fold_4/train'


def get_patient_name(path):
    return re.search(r"/data/\w+/([A-Z]{3})", path).group(1)

def get_class_name(path):
    return re.search(r"/data/(\w+)", path).group(1)

def get_image_number(path):
    return re.search(r"image_(\d).tif", path).group(1)

def get_classification_patient(patient_folder):
    probs_path = patient_folder + '/single_cell_probabilities.npy'
    sc_probs = np.load(probs_path)
    sc_class= np.argmax(sc_probs, axis=1)
    return sc_class


df = pd.DataFrame(columns=["patient", "AML_subtype"] + sc_class_labels)
# Save class classification count for each patient in csv file
for folder_class in os.listdir(data_path):
    folder_class = os.path.join(data_path, folder_class)

    if os.path.isdir(folder_class):
        print(folder_class)
        for folder_patient in os.listdir(folder_class):
            folder_patient = os.path.join(folder_class, folder_patient)

            if os.path.isdir(folder_patient):
                if "single_cell_probabilities.npy" not in os.listdir(folder_patient):
                    print("Skipping patient folder without single_cell_probabilities.npy:", folder_patient)
                    continue
                sc_class = get_classification_patient(folder_patient)
                counts_vector, unique_labels = get_counts_vector(sc_class)
                df.loc[len(df)] = np.array(
                    [get_patient_name(folder_patient), get_class_name(folder_patient)] + counts_vector.tolist())

df[sc_class_labels]=df[sc_class_labels].astype(int)
df[["patient","AML_subtype"]]=df[["patient","AML_subtype"]].astype(str)
df.to_csv(result_path+"/single_cell_results.csv")
