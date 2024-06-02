import random as r
from sklearn.model_selection import KFold

'''Script splits data into x folds, and returns the specific sets
with the function return_folds'''

data_split = None


def split_in_folds(data, num_folds):
    '''splits data into num_folds shares. Split
    data can then be retrieved using return_folds.
    Data comes in a dict which has to be formatted as:

        data[entity] = [patient_list]

    So that in the end every fold of num_folds contains a
    stratified part of patients for every entity.
    '''

    global data_split

    data_split = dict()
    percent_per_split = 1 / num_folds

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # iterate over all entities
    for key, value in data.items():

        # following routine makes sure data is always split in the same way
        r.seed(42)
        ordered_patients = sorted(value)
        r.shuffle(ordered_patients)

        patient_ids = list(range(len(ordered_patients)))

        for fold, (train_index, val_index) in enumerate(kf.split(patient_ids)):
            if fold not in data_split:
                data_split[fold] = dict()
            train_ids = [ordered_patients[i] for i in train_index]
            val_ids = [ordered_patients[i] for i in val_index]

            data_split[fold][key] = {'train': train_ids, 'val': val_ids}

    print("Data split into {} folds.".format(num_folds))


def return_folds(folds):
    '''Returns all data from data_split for the specified fold of the
    previously calculated split.
    '''
    global data_split

    if isinstance(folds, int):
        folds = [folds]

    data_final = dict()

    # merge together multiple folds to return one dictionary
    for fold in folds:
        if fold not in data_split:
            raise ValueError(f"Fold {fold} not found in data_split")

        for key, value in data_split[fold].items():
            if key not in data_final:
                data_final[key] = {'train': [], 'val': []}
            data_final[key]['train'].extend(value['train'])
            data_final[key]['val'].extend(value['val'])

    return data_final
