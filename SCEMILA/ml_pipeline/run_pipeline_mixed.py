from model_train import *   # model training function
from model import *         # actual MIL model
from dataset_mixed import *       # dataset
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

torch.multiprocessing.set_sharing_strategy('file_system')

# import from other, own modules
# get the number of patients in each class counts
def get_class_sizes(folder,dictionary=None):
    class_sizes = []
    for i,class_label in enumerate(['PML_RARA','NPM1','CBFB_MYH11','RUNX1_RUNX1T1','control']):
        if dictionary is None:
            count = len(os.listdir(folder+"/"+class_label))
            class_sizes.append(count)
        else:
            class_sizes.append(len(dictionary[class_label]))
    return class_sizes

# 1: Setup. Source Folder is parent folder for both mll_data_master and
# the /data folder
# results will be stored here
TARGET_FOLDER = "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/results/mixed_seed42_max20"
# path to dataset
SOURCE_FOLDER = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/mixed_uncertain_fixbug_seed42/max_20_percent'


# get arguments from parser, set up folder
# parse arguments
parser = ap.ArgumentParser()

# Algorithm / training parameters
parser.add_argument(
    '--fold',
    help='offset for cross-validation (1-5). Change to cross-validate',
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
    default=150)               # epochs to train
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
    default=20) #previously set to 20 
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

#Data and output folder
parser.add_argument(
    '--target_folder',
    help='Target folder: where results are shaves',
    required=True,
    default="/mnt/volume/shared/all_results/debug") 

#Data and output folder
parser.add_argument(
    '--source_folder',
    help='Source folder: where data is stored',
    required=True,
    default='/mnt/volume/shared/data_file/data') 


args = parser.parse_args()

# the /data folder
# results will be stored here
TARGET_FOLDER = args.target_folder
# path to dataset
SOURCE_FOLDER = args.source_folder

# store results in target folder
TARGET_FOLDER = os.path.join(TARGET_FOLDER, args.result_folder)
if not os.path.exists(TARGET_FOLDER):
    os.mkdir(TARGET_FOLDER)
start = time.time()


# 2: Dataset
# Initialize datasets, dataloaders, ...
print("")
print('Initialize datasets...')
with open(SOURCE_FOLDER+'/file_paths.pkl', 'rb') as f:
    mixed_data_filepaths = pickle.load(f)
label_conv_obj = label_converter.LabelConverter()
set_dataset_path(SOURCE_FOLDER)
define_dataset(
    num_folds=5,
    prefix_in=args.prefix,
    label_converter_in=label_conv_obj,
    filter_diff_count=int(args.filter_diff),
    filter_quality_minor_assessment=int(args.filter_mediocre_quality),
    merge_dict_processed=mixed_data_filepaths
)

results = {
    'train': [],
    'val': [],
    'test': []
}

# Ensure the target folder exists
os.makedirs(args.target_folder, exist_ok=True)

# File to save results
results_file = os.path.join(args.target_folder, 'cross_validation_results.txt')

# Clear results file if it exists
if os.path.exists(results_file):
    os.remove(results_file)

with open(results_file, 'a') as f:
    f.write("Fold\tTrain Accuracy\tVal Accuracy\tTest Accuracy\n")

for fold in range(5):
    datasets = {}

    # Set up folds for cross-validation, including the test set
    num_folds = 5
    folds = {'train': [], 'val': [], 'test': []}

    # Determine the fold numbers
    all_folds = np.arange(num_folds)
    val_fold = fold
    test_fold = (val_fold + 1) % num_folds  # Use the next fold as the test set
    train_folds = [f for f in all_folds if f != val_fold and f != test_fold]

    # Set the fold indices
    folds['val'] = [val_fold]
    folds['test'] = [test_fold]
    folds['train'] = train_folds

    # Initialize datasets
    datasets['train'] = MllDataset(folds=folds['train'], aug_im_order=True, split='train', patient_bootstrap_exclude=int(args.bootstrap_idx))
    datasets['val'] = MllDataset(folds=folds['val'], aug_im_order=False, split='val')
    datasets['test'] = MllDataset(folds=folds['test'], aug_im_order=False, split='test')

    # Ensure balanced sampling for training
    class_sizes = get_class_sizes(args.prefix, mixed_data_filepaths)
    class_count = len(class_sizes)
    label_freq = [class_sizes[c] / sum(class_sizes) for c in range(class_count)]
    individual_sampling_prob = [(1 / class_count) * (1 / label_freq[c]) for c in range(class_count)]

    idx_sampling_freq_train = torch.tensor(individual_sampling_prob)[datasets['train'].labels]
    sampler_train = WeightedRandomSampler(weights=idx_sampling_freq_train, replacement=True, num_samples=len(idx_sampling_freq_train))

    dataloaders = {
        'train': DataLoader(datasets['train'], sampler=sampler_train),
        'val': DataLoader(datasets['val']),  # Without sampler
        'test': DataLoader(datasets['test'])  # Without sampler
    }

    # Initialize model, optimizer, and scheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()
    print("Found device: ", ngpu, "x ", device)

    model = AMiL(class_count=class_count, multicolumn=int(args.multi_att), device=device)

    if ngpu > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    print("Setup complete.")
    print("")

    optimizer = optim.SGD(model.parameters(), lr=float(args.lr), momentum=0.9, nesterov=True)
    scheduler = None

    # Launch training
    train_obj = ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        epochs=int(args.ep),
        optimizer=optimizer,
        scheduler=scheduler,
        class_count=class_count,
        early_stop=int(args.es),
        device=device
    )
    model, conf_matrix, data_obj = train_obj.launch_training()

    # Append results for this fold
    results['train'].append(train_obj.train_accuracy)
    results['val'].append(train_obj.val_accuracy)
    results['test'].append(train_obj.test_accuracy)

    # Print and save results for this fold
    fold_results = f"Fold {fold}: Train Accuracy: {train_obj.train_accuracy:.2f}, Val Accuracy: {train_obj.val_accuracy:.2f}, Test Accuracy: {train_obj.test_accuracy:.2f}\n"
    print(fold_results)
    with open(results_file, 'a') as f:
        f.write(f"{fold}\t{train_obj.train_accuracy:.2f}\t{train_obj.val_accuracy:.2f}\t{train_obj.test_accuracy:.2f}\n")

    # Save confusion matrix from test set, all the data, model, print parameters
    np.save(os.path.join(args.target_folder, f'test_conf_matrix_fold{fold}.npy'), conf_matrix)
    pickle.dump(data_obj, open(os.path.join(args.target_folder, f'testing_data_fold{fold}.pkl'), "wb"))

    if int(args.save_model):
        torch.save(model, os.path.join(args.target_folder, f'model_fold{fold}.pt'))
        torch.save(model.state_dict(), os.path.join(args.target_folder, f'state_dictmodel_fold{fold}.pt'))

# Calculate average accuracy across all folds
avg_train_accuracy = np.mean(results['train'])
avg_val_accuracy = np.mean(results['val'])
avg_test_accuracy = np.mean(results['test'])

# Print and save average results
avg_results = f"Average Train Accuracy: {avg_train_accuracy:.2f}\nAverage Validation Accuracy: {avg_val_accuracy:.2f}\nAverage Test Accuracy: {avg_test_accuracy:.2f}\n"
print(avg_results)
with open(results_file, 'a') as f:
    f.write("\nAverages\n")
    f.write(f"Train\t{avg_train_accuracy:.2f}\n")
    f.write(f"Val\t{avg_val_accuracy:.2f}\n")
    f.write(f"Test\t{avg_test_accuracy:.2f}\n")

end = time.time()
runtime = end - start
time_str = str(int(runtime // 3600)) + "h" + str(int((runtime %
                                                      3600) // 60)) + "min" + str(int(runtime % 60)) + "s"

# other parameters
print("")
print("------------------------Final report--------------------------")
print('prefix', args.prefix)
print('Runtime', time_str)
print('max. Epochs', args.ep)
print('Learning rate', args.lr)
