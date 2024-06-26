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

datasets = {}

# Set up folds for cross-validation, including the test set
num_folds = 5
folds = {'train': [], 'val': [], 'test': []}

# Determine the fold numbers
all_folds = np.arange(num_folds)
val_fold = int(args.fold)
test_fold = (val_fold + 1) % num_folds  # Use the next fold as the test set
train_folds = [fold for fold in all_folds if fold != val_fold and fold != test_fold]

# Set the fold indices
folds['val'] = [val_fold]
folds['test'] = [test_fold]
folds['train'] = train_folds

# Initialize datasets
datasets['train'] = MllDataset(folds=folds['train'], aug_im_order=True, split='train', patient_bootstrap_exclude=int(args.bootstrap_idx))
datasets['val'] = MllDataset(folds=folds['val'], aug_im_order=False, split='val')
datasets['test'] = MllDataset(folds=folds['test'], aug_im_order=False, split='test')


# store conversion from true string labels to artificial numbers for
# one-hot encoding
df = label_conv_obj.df
df.to_csv(os.path.join(TARGET_FOLDER, "class_conversion.csv"), index=False)
class_count = 5
print("Data distribution: ")
print(df)
print(df.size_tot)
# Initialize dataloaders
print("Initialize dataloaders...")
dataloaders = {}

# ensure balanced sampling
# get total sample sizes
class_sizes = get_class_sizes(SOURCE_FOLDER,mixed_data_filepaths)
# calculate label frequencies
label_freq = [class_sizes[c] / sum(class_sizes) for c in range(class_count)]
# balance sampling frequencies for equal sampling
individual_sampling_prob = [
    (1 / class_count) * (1 / label_freq[c]) for c in range(class_count)]
print(datasets['train'])
idx_sampling_freq_train = torch.tensor(individual_sampling_prob)[
    datasets['train'].labels]
idx_sampling_freq_val = torch.tensor(individual_sampling_prob)[
    datasets['val'].labels]

sampler_train = WeightedRandomSampler(
    weights=idx_sampling_freq_train,
    replacement=True,
    num_samples=len(idx_sampling_freq_train))
# sampler_val = WeightedRandomSampler(weights=idx_sampling_freq_val, replacement=True, num_samples=len(idx_sampling_freq_val))

dataloaders['train'] = DataLoader(
    datasets['train'],
    sampler=sampler_train)
dataloaders['val'] = DataLoader(
    datasets['val'])  # , sampler=sampler_val)
dataloaders['test'] = DataLoader(datasets['test'])
print("")


# 3: Model
# initialize model, GPU link, training

# set up GPU link and model (check for multi GPU setup)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = torch.cuda.device_count()
print("Found device: ", ngpu, "x ", device)

model = AMiL(
    class_count=class_count,
    multicolumn=int(
        args.multi_att),
    device=device)

if(ngpu > 1):
    model = torch.nn.DataParallel(model)
model = model.to(device)
print("Setup complete.")
print("")

# set up optimizer and scheduler
optimizer = optim.SGD(
    model.parameters(),
    lr=float(
        args.lr),
    momentum=0.9,
    nesterov=True)
scheduler = None

# launch training
train_obj = ModelTrainer(
    model=model,
    dataloaders=dataloaders,
    epochs=int(
        args.ep),
    optimizer=optimizer,
    scheduler=scheduler,
    class_count=class_count,
    early_stop=int(
        args.es),
    device=device)
model, conf_matrix, data_obj = train_obj.launch_training()


# 4: aftermath
# save confusion matrix from test set, all the data , model, print parameters

np.save(os.path.join(TARGET_FOLDER, 'test_conf_matrix.npy'), conf_matrix)
pickle.dump(
    data_obj,
    open(
        os.path.join(
            TARGET_FOLDER,
            'testing_data.pkl'),
        "wb"))

if(int(args.save_model)):
    torch.save(model, os.path.join(TARGET_FOLDER, 'model.pt'))
    torch.save(model, os.path.join(TARGET_FOLDER, 'state_dictmodel.pt'))

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
