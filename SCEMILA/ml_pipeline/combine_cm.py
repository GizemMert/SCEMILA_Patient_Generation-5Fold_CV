import numpy as np
import os
import label_converter
import sys
sys.path.append('/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/SCEMILA/analysis/functions')
import confusion_matrix as cm_module  # Import the confusion matrix module

base_path = "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV"
fold_paths = [
    os.path.join(base_path, f"result_fold_{i}_mixed/mixed_seed42_max30/test_conf_matrix.npy")
    for i in range(5)
]

# Load and sum the confusion matrices from all folds
sum_confusion_matrix = None
for fold_path in fold_paths:
    if os.path.exists(fold_path):
        cm_data = np.load(fold_path)
        if sum_confusion_matrix is None:
            sum_confusion_matrix = cm_data
        else:
            sum_confusion_matrix += cm_data

# Define the reorder list for confusion matrix display
reorder = ['PML_RARA', 'NPM1', 'CBFB_MYH11', 'RUNX1_RUNX1T1', 'control']

# Define the function to save the confusion matrix
def save_confusion_matrix(confusion_data, lbl_conv_obj, fig_export_path):
    # Save confusion matrix as npy file
    np.save(os.path.join(fig_export_path, 'confusion_matrix.npy'), confusion_data)

    # Plot and save confusion matrix as SVG
    cm_module.show_pred_mtrx(
        pred_mtrx=confusion_data,
        class_conversion=lbl_conv_obj.df,
        reorder=reorder,
        fig_size=(8.1, 4.5),
        path_save=os.path.join(fig_export_path, 'confusion_matrix_250_30.svg')
    )

# Define the path to the label conversion file
label_conv_obj = label_converter.LabelConverter(
    path_preload=os.path.join(base_path, "result_fold_0/class_conversion.csv")
)

# Define the path to save the final confusion matrix
fig_export_path = "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/combined_cm"

# Create the directory if it does not exist
os.makedirs(fig_export_path, exist_ok=True)

# Save the summed confusion matrix
save_confusion_matrix(sum_confusion_matrix, label_conv_obj, fig_export_path)
