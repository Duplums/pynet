import numpy as np
import sys
sys.path.append('../pynet')
import os, pickle, operator, math
from pynet.datasets.core import DataManager
from pynet.transforms import LabelMapping, Crop, Normalize, Padding
from json_config import CONFIG
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

def train_linear_model(X_train, y_train, X_val, y_val, reg=True, hyperparams=None, scaler=None):
    n = len(X_train)
    n_val = len(X_val)
    # Train the model and pick the best hyperparameters
    X_train, y_train = np.array(X_train).reshape(n, -1), np.array(y_train).reshape(n, -1)
    X_val, y_val = np.array(X_val).reshape(n_val, -1), np.array(y_val).reshape(n_val, -1)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
    if hyperparams is None:
        hyperparams = dict()
    params_grid = ParameterGrid(hyperparams)
    best_model, best_score, best_params = None, -math.inf, None
    for params in params_grid:
        model = Ridge(**params) if reg==True else LogisticRegression(**params)
        model = model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        if best_score < score:
            best_model = model
            best_params = params
            best_score = score
    return best_model, best_params, best_score


## Defines the experimental setup parameters
root = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/'
saving_dir = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/LinearModel'
data_dirs = {'': {'input_path': 'cat12vbm/all_t1mri_mwp1_gs-raw_data32_tocheck.npy',
                  'metadata_path': 'cat12vbm/all_t1mri_mwp1_participants.tsv'},
             'quasi_raw': {'input_path': 'quasi_raw/all_t1mri_quasi_raw_data32_1.5mm_skimage.npy',
                           'metadata_path': 'quasi_raw/all_t1mri_quasi_raw_participants.tsv'
                           }
             }
preprocs = ['', 'quasi_raw']
pbs = ["Age", "Sex", "Dx"]
models_hyperparams = [{'alpha': [1e-2, 1e-1, 1], 'fit_intercept': [False]},
                      {'C': [1e2, 1e1, 1], 'solver': ['liblinear']},
                      {'C': [1e2, 1e1, 1], 'solver': ['liblinear']}]
dbs = ["HCP_IXI", "HCP_IXI", "SCZ_VIP"]
dbs_config = ["healthy", "healthy", "tiny_scz_kfolds"]
labels = ['age', 'sex', 'diagnosis']
strat_labels = ['age', 'site', 'diagnosis']
all_sites = ['PRAGUE', 'pittsburgh', 'Boston', 'NU', 'Hartford', 'sandiego', 'udine', 'WashU', 'Detroit', 'MRN',
             'mannheim', 'creteil', 'galway', 'WUSTL','LONDON','geneve', 'Baltimore', 'ICM', 'Sainte-Anne', 'vip',
             'grenoble', 'Dallas']
strat_label_mappings= [LabelMapping(), LabelMapping(**{s: i for (i,s) in enumerate(all_sites)}),
                       LabelMapping(schizophrenia=1, control=0)]
label_mappings = [LabelMapping(), LabelMapping(), LabelMapping(schizophrenia=1, control=0)]
nb_training_samples = [[100, 300, 500, 1000, 1600],[100, 300, 500, 1000, 1600], [100, 300, 500]]
total_nb_folds = [[10, 10, 10, 5, 5], [10, 10, 10, 5, 5], [10, 10, 10]]
scalers = [None, StandardScaler(copy=True)]

## Actual loading of data + training
for (scaler, preproc) in zip(scalers, preprocs):
    input_path = os.path.join(root, data_dirs[preproc]['input_path'])
    metadata_path = os.path.join(root, data_dirs[preproc]['metadata_path'])
    print('Loading data...', flush=True)
    data = np.load(input_path, mmap_mode='r')
    print('Data loaded !', flush=True)
    for i_pb, (pb, label_map, label, strat_label_map, strat_label, db, db_config, hyperparams) in \
        enumerate(zip(pbs, label_mappings, labels, strat_label_mappings,
                      strat_labels, dbs, dbs_config, models_hyperparams)):
        if pb == 'Age' and preproc == '':
            continue
        for (N, nb_folds) in zip(nb_training_samples[i_pb], total_nb_folds[i_pb]):
            manager = DataManager(None, metadata_path,
                                  number_of_folds=nb_folds,
                                  labels=[label],
                                  labels_transforms=[label_map],
                                  stratify_label=strat_label,
                                  stratify_label_transforms=[strat_label_map],
                                  categorical_strat_label=(pb!="Age"),
                                  N_train_max=N,
                                  custom_stratification=CONFIG['db'][db_config], sep='\t')
            for fold in range(nb_folds):
                train_indices = manager.dataset['train'][fold].indices
                val_indices = manager.dataset['validation'][fold].indices
                ## Actual loading into memory
                X_train = data[train_indices]
                y_train = np.array([label_map(label) for label in manager.labels[train_indices]])
                X_val = data[val_indices]
                y_val = np.array([label_map(label) for label in manager.labels[val_indices]])

                model, best_params, best_score = train_linear_model(X_train, y_train, X_val, y_val,
                                                                    reg=(pb == 'Age'),
                                                                    hyperparams=hyperparams, scaler=scaler)
                print('{} Prediction (fold {}, N={}, preproc={}):\n\t*Best score: {}\n\t*Best params: {}'.format(
                    pb, fold, N, preproc or 'cat12', best_score, best_params), flush=True)
                del(X_train, X_val, y_train, y_val)

                ## Test it on independent dataset
                test_indices = manager.dataset['test'].indices
                X_test = data[test_indices].reshape(len(test_indices), -1)
                if scaler is not None:
                    X_test = scaler.fit_transform(X_test)
                y_test = np.array([label_map(label) for label in manager.labels[test_indices]])
                if pb=="Age":
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict_proba(X_test)[:, 1]
                del(X_test)

                ## Saves the result on disk
                exp_name = 'Test_{net}_{pb}_{db}_fold{k}_epoch{e}.pkl'.format(net=type(model).__name__,
                                                                              pb=pb, db=db, k=fold,e=100)
                with open(os.path.join(saving_dir, preproc, 'N_%i'%N, pb, exp_name), 'wb') as f:
                    pickle.dump({'y_pred': y_pred, 'y_true': y_test}, f)
            del(manager)



