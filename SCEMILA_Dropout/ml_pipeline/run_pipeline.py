from model_train import *   # model training function
from model import *         # actual MIL model
from dataset import *       # dataset
# makes conversion from string label to one-hot encoding easier
import label_converter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.multiprocessing
import torch
import sys
import os
import time
import argparse as ap
from sklearn.model_selection import KFold
import numpy as np
from dataset import MllDataset
from data_split import split_in_folds, return_folds



torch.multiprocessing.set_sharing_strategy('file_system')

# import from other, own modules


# 1: Setup. Source Folder is parent folder for both mll_data_master and
# the /data folder
#Normal Scemila
# results will be stored here
#TARGET_FOLDER = r'/mnt/c/Users/Hillary Hauger/Documents/Studium/WS23-24/Computational Methods for Single-cell Biology/smalldataset/data/output'
# path to dataset
#SOURCE_FOLDER = r'/mnt/c/Users/Hillary Hauger/Documents/Studium/WS23-24/Computational Methods for Single-cell Biology/smalldataset'
#Random shuffe: Experiment 1

# get arguments from parser, set up folder
# parse arguments
parser = ap.ArgumentParser()

# Algorithm / training parameters
parser.add_argument(
    '--fold',
    help='offset for cross-validation (0-4). Change to cross-validate',
    required=False,
    default=0)  # shift folds for cross validation. Increasing by 1 moves all folds by 1.
parser.add_argument(
    '--lr',
    help='used learning rate',
    required=False,
    default=0.00005)                                     # learning rate
parser.add_argument(
    '--ep',
    help='max. amount after which training should stop',
    required=False,
    default=2)               # epochs to train
parser.add_argument(
    '--es',
    help='early stopping if no decrease in loss for x epochs',
    required=False,
    default=20)          # epochs without improvement, after which training should stop.
parser.add_argument(
    '--multi_att',
    help='use multi-attention approach',
    required=False,
    default=1)                          # use multiple attention values if 1

# Data parameters: Modify the dataset
parser.add_argument(
    '--prefix',
    help='define which set of features shall be used',
    required=False,
    default='fnl34_')        # define feature source to use (from different CNNs)
# pass -1, if no filtering acc to peripheral blood differential count
# should be done
parser.add_argument(
    '--filter_diff',
    help='Filters AML patients with less than this perc. of MYB.',
    default=20)
# Leave out some more samples, if we have enough without them. Quality of
# these is not good, but if data is short, still ok.
parser.add_argument(
    '--filter_mediocre_quality',
    help='Filters patients with sub-standard sample quality',
    default=0)
parser.add_argument(
    '--bootstrap_idx',
    help='Remove one specific patient at pos X',
    default=-
    1)                             # Remove specific patient to see effect on classification

# Output parameters
parser.add_argument(
    '--result_folder',
    help='store folder with custom name',
    required=True)                                 # custom output folder name
parser.add_argument(
    '--save_model',
    help='choose wether model should be saved',
    required=False,
    default=1)                  # store model parameters if 1
args = parser.parse_args()

# Ensure the fold number is within the valid range
if not (0 <= args.fold < 5):
    raise ValueError("Fold number must be between 0 and 4.")

TARGET_FOLDER = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/target'
SOURCE_FOLDER = '/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/F_AML/TCIA_data_prepared/data'
FEATURES_ZIP = '/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/F_AML/TCIA_data_prepared/TCIA-features.zip'


# store results in target folder
TARGET_FOLDER = os.path.join(TARGET_FOLDER, args.result_folder)
if not os.path.exists(TARGET_FOLDER):
    os.mkdir(TARGET_FOLDER)
start = time.time()

# Dataset Initialization
print("\nInitialize datasets...")
label_conv_obj = label_converter.LabelConverter()
set_dataset_path(SOURCE_FOLDER)
define_dataset(num_folds=5, prefix_in=args.prefix, label_converter_in=label_conv_obj, filter_diff_count=int(args.filter_diff), filter_quality_minor_assessment=int(args.filter_mediocre_quality))

# Extract patient IDs
def get_patient_ids(source_folder):
    data = {}
    for subtype in os.listdir(source_folder):
        subtype_path = os.path.join(source_folder, subtype)
        if os.path.isdir(subtype_path):
            patients = [os.path.join(subtype, patient) for patient in os.listdir(subtype_path) if os.path.isdir(os.path.join(subtype_path, patient))]
            data[subtype] = patients
    return data

patient_data = get_patient_ids(SOURCE_FOLDER)
split_in_folds(patient_data, num_folds=5)

# Initialize cross-validation results
all_fold_losses = []
all_fold_accuracies = []

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Perform 5-Fold Cross-Validation
for current_fold in range(5):
    print(f"Starting fold {current_fold + 1}...")

    # Retrieve the current fold data
    fold_data = return_folds(current_fold)

    train_ids = []
    val_ids = []

    for key, ids in fold_data.items():
        train_ids.extend(ids['train'])
        val_ids.extend(ids['val'])

    datasets = {
        'train': MllDataset(folds=train_ids, aug_im_order=True, split='train', patient_bootstrap_exclude=args.bootstrap_idx, features_zip=FEATURES_ZIP),
        'val': MllDataset(folds=val_ids, aug_im_order=False, split='val', features_zip=FEATURES_ZIP)
    }

    df = label_conv_obj.df
    df.to_csv(os.path.join(TARGET_FOLDER, f"class_conversion_fold_{current_fold + 1}.csv"), index=False)
    class_count = len(df)
    print("Data distribution: ")
    print(df)

    # Initialize dataloaders
    print("Initialize dataloaders...")
    dataloaders = {}
    class_sizes = list(df.size_tot)
    label_freq = [class_sizes[c] / sum(class_sizes) for c in range(class_count)]
    individual_sampling_prob = [(1 / class_count) * (1 / label_freq[c]) for c in range(class_count)]

    idx_sampling_freq_train = torch.tensor(individual_sampling_prob)[datasets['train'].labels]
    idx_sampling_freq_val = torch.tensor(individual_sampling_prob)[datasets['val'].labels]

    sampler_train = WeightedRandomSampler(weights=idx_sampling_freq_train, replacement=True, num_samples=len(idx_sampling_freq_train))

    dataloaders['train'] = DataLoader(datasets['train'], sampler=sampler_train)
    dataloaders['val'] = DataLoader(datasets['val'])

    # Model Initialization
    ngpu = torch.cuda.device_count()
    print(f"Found {ngpu} GPU(s)")

    model = AMiL(class_count=class_count, multicolumn=int(args.multi_att), device=device)
    if ngpu > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    print("Setup complete.\n")

    # Optimizer and Scheduler Setup
    optimizer = optim.SGD(model.parameters(), lr=float(args.lr), momentum=0.9, nesterov=True)
    scheduler = None

    # Launch Training
    train_obj = ModelTrainer(model=model, dataloaders=dataloaders, epochs=int(args.ep), optimizer=optimizer, scheduler=scheduler, class_count=class_count, early_stop=int(args.es), device=device)
    model, conf_matrix, data_obj = train_obj.launch_training()

    # Collect metrics for this fold
    all_fold_losses.append(train_obj.best_loss)
    all_fold_accuracies.append(train_obj.best_acc)

    if int(args.save_model):
        torch.save(model, os.path.join(TARGET_FOLDER, f'model_fold_{current_fold}.pt'))
        torch.save(model, os.path.join(TARGET_FOLDER, f'state_dictmodel_fold_{current_fold}.pt'))


# Calculate and print average metrics across all folds
avg_loss = np.mean(all_fold_losses)
avg_accuracy = np.mean(all_fold_accuracies)
print(f"Average Loss across folds: {avg_loss}")
print(f"Average Accuracy across folds: {avg_accuracy}")

end = time.time()
runtime = end - start
time_str = f"{int(runtime // 3600)}h{int((runtime % 3600) // 60)}min{int(runtime % 60)}s"

# Final Report
print("\n------------------------Final report--------------------------")
print('prefix', args.prefix)
print('Runtime', time_str)
print('max. Epochs', args.ep)
print('Learning rate', args.lr)
print(f'Average Loss: {avg_loss:.4f}')
print(f'Average Accuracy: {avg_acc:.4f}')
