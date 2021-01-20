import os
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pynet.utils import get_pickle_obj
from pynet.plotting.image import plot_data_reduced, plot_losses
from pynet.datasets.core import DataManager
from json_config import CONFIG
from pynet.history import History
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import pearsonr
import pandas as pd
import pickle
seaborn.set_style("darkgrid")

## Analysis of the generalization power of DenseNet pretrained on different pb (supervised and unsupervised) for dx classification


def train_linear_model(data_train, data_test, reg=False, **kwargs):
    # Train the model with a logistic regression
    X, y = np.array(data_train['y']).reshape(len(data_train['y']), -1), np.array(data_train['y_true']).ravel()
    if reg:
        model = Ridge(**kwargs).fit(X, y)
    else:
        model = LogisticRegression(solver='liblinear', **kwargs).fit(X, y)
    # Test the model
    X_test, y_test = np.array(data_test['y']).reshape(len(data_test['y']), -1), \
                     np.array(data_test['y_true']).ravel()
    if reg:
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict_proba(X_test)[:, 1]
    return y_test, y_pred



blocks = ["block1", "block2", "block3", "block4"]
root = "/neurospin/psy_sbox/bd261576/checkpoints"

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15,5))


### Compute the baseline: only logistic regression on the input images

# data, meta = np.load(CONFIG['cat12']['input_path'], mmap_mode='r'), \
#              pd.read_csv(CONFIG['cat12']['metadata_path'], sep='\t')

## Age prediction
# mask_train = DataManager.get_mask(meta, CONFIG['db']['healthy']['train'])
# mask_test = DataManager.get_mask(meta, CONFIG['db']['healthy']['test'])
# model = Ridge().fit(data[mask_train].reshape(mask_train.sum(),-1),meta[mask_train].age)
# pred = model.predict(data[mask_test].reshape(mask_test.sum(),-1))
# true = data[mask_test].reshape(mask_test)
## DX prediction
# mask_train = DataManager.get_mask(meta, CONFIG['db']['tiny_scz_kfolds']['train'])
# mask_test = DataManager.get_mask(meta, CONFIG['db']['tiny_scz_kfolds']['test'])
# model = LogisticRegression(solver='liblinear').fit(data[mask_train].reshape(mask_train.sum(),-1),
#                                                    (meta[mask_train].diagnosis=='schizophrenia'))
# pred = model.predict_proba(data[mask_test].reshape(mask_test.sum(),-1))[:,1]
# true = (meta[mask_test].diagnosis=='schizophrenia')
# axes[0].scatter(0, roc_auc_score(true, pred), marker='*', color='red', label='Baseline')
# axes[1].scatter(0, balanced_accuracy_score(true, pred>0.5), marker='*', color='red', label='Baseline')

# AUC = 0.7914432989690722, bAcc = 0.7039690721649485
# Plots the reference line (linear baseline)
axes[0].axhline(0.7914432989690722, linestyle="--", color='gray', linewidth=.7, label='Logistic Regression')
axes[1].axhline(0.7914432989690722, linestyle="--", color='gray', linewidth=.7, label='Logistic Regression')
axes[2].axhline(0.7914432989690722, linestyle="--", color='gray', linewidth=.7, label='Logistic Regression')


paths = [
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/Age/block_outputs",
    "scz_prediction/Benchmark_Transfer/Age_Pretraining/schizconnect_vip/DenseNet/block_outputs",
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/Sex_Age/block_outputs",
    "scz_prediction/Benchmark_Transfer/Sex_Age_Pretraining/schizconnect_vip/DenseNet/block_outputs",
    "self_supervision/simCLR/DenseNet/N_1600/exp_3/age_supervision/noisy_spike_motion_crop_DA/block_outputs",
    "scz_prediction/Benchmark_Transfer/Self_Supervision/schizconnect_vip/SimCLR_Exp3/noisy_spike_motion_crop_DA/age_supervision/block_outputs",
    "self_supervision/simCLR/DenseNet/N_1600/exp_3/unsupervised/noisy_spike_motion_crop_DA/block_outputs",
    "scz_prediction/Benchmark_Transfer/Self_Supervision/schizconnect_vip/SimCLR_Exp3/noisy_spike_motion_crop_DA/unsupervised/block_outputs",
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/block_outputs",
    "scz_prediction/Benchmark_Transfer/No_Pretraining/schizconnect_vip/DenseNet/block_outputs"
]

training_filenames = [
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch99.pkl",
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch49.pkl",
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch99.pkl",
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch49.pkl",
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch199.pkl",
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch49.pkl",
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch299.pkl",
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch49.pkl",
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch0.pkl", # Random init
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch49.pkl"
]
testing_filenames = [
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch99.pkl",
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch49.pkl",
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch99.pkl",
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch49.pkl",
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch199.pkl",
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch49.pkl",
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch299.pkl",
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch49.pkl",
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch0.pkl", # Random init
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch49.pkl"
]

exp_names = [
    "Age",
    "Age",
    "Sex+Age",
    "Sex+Age",
    "SimCLR Supervised on Age",
    "SimCLR Supervised on Age",
    "Unsupervised SimCLR",
    "Unsupervised SimCLR",
    "Random",
    "Random",
]

nb_folds = [1, 5, 1, 5, 1, 5, 1, 5, 5, 5]
colors = ["blue", "green", "orange", "cyan", "red"]

metrics = {'auc': {path: {b: [] for b in blocks} for path in paths},
           'balanced_accuracy': {path: {b: [] for b in blocks} for path in paths}}
metrics = get_pickle_obj('metrics_TL_sup_unsup.pkl')
## Plots the performance of pretrained/fine-tuned networks after each block
for i, (path, train_file, test_file, exp) in enumerate(zip(paths, training_filenames, testing_filenames, exp_names)):
    for j, b in enumerate(blocks):
        for fold in range(nb_folds[i]):

            # Train the model with a logistic regression
            data = get_pickle_obj(os.path.join(root, path, train_file.format(b=j+1, f=fold)))
            testing_data = get_pickle_obj(os.path.join(root, path, test_file.format(b=j+1, f=fold)))

            y_test, y_pred = train_linear_model(data, testing_data)
            metrics['auc'][path][b].append(roc_auc_score(y_test, y_pred))
            metrics['balanced_accuracy'][path][b].append(balanced_accuracy_score(y_test, y_pred>0.5))
    seaborn.lineplot(x=[b+1 for b in range(len(blocks)) for _ in range(nb_folds[i])],
                     y=[metrics['auc'][path][b][k] for b in blocks for k in range(nb_folds[i])],
                     label=exp, ax=axes[i%2], marker='o', color=colors[i//2])
    axes[0].lines[-1].set_linestyle('--')

# with open('metrics_TL_sup_unsup.pkl', 'wb') as f:
#     pickle.dump(metrics, f)

### Plots the performance after fine-tuning of several parts of the network
blocks = [ "Block1", "Block2", "Block3", "Block4"]
root = "/neurospin/psy_sbox/bd261576/checkpoints/scz_prediction/Benchmark_Transfer"
pretrainings = ["Age_Pretraining", "Sex_Age_Pretraining", "Self_Supervision", "Self_Supervision", "No_Pretraining"]
exp_names = ["Age", "Age+Sex", "SimCLR Supervised on Age", "Unsupervised SimCLR", "Random"]
settings = ["DenseNet", "DenseNet", "SimCLR_Exp3/noisy_spike_motion_crop_DA/age_supervision",
            "SimCLR_Exp3/noisy_spike_motion_crop_DA/unsupervised", "DenseNet"]
db = "schizconnect_vip"
network = "DenseNet"


metrics_frozen_blocks = {exp: {b: dict() for b in blocks} for exp in exp_names}
for i, (exp_name, pretraining, setting) in enumerate(zip(exp_names, pretrainings, settings)):
    for block in blocks:
        path = os.path.join(root, pretraining, db, setting)
        if block == "None":
            tests = [get_pickle_obj(os.path.join(path, "Test_DenseNet_Dx_SCZ_VIP_fold{k}_epoch49.pkl".format(k=k)))
                     for k in range(5)]
        else:
           tests = [get_pickle_obj(os.path.join(path, block+'_frozen',
                                    "Test_DenseNet_Dx_SCZ_VIP_fold{k}_epoch49.pkl".format(k=k))) for k in range(5)]
        metrics_frozen_blocks[exp_name][block]["auc"] = [roc_auc_score(t['y_true'], t['y_pred']) for t in tests]

    seaborn.lineplot(x=[i for i, _ in enumerate(blocks) for _ in range(5)],
                     y=[metrics_frozen_blocks[exp_name][b]['auc'][k] for b in blocks for k in range(5)],
                     label=exp_name, marker='o', ax=axes[2], color=colors[i])

axes[0].set_xlabel('Feature Extraction until Block...')
axes[0].set_ylabel('AUC')
axes[0].set_xticks(range(len(blocks)+1))
axes[0].set_xticklabels(range(0, len(blocks)+1))
axes[0].set_title('(1) Linear Probe on Frozen Encoder', fontweight="bold")

axes[1].tick_params(labelleft=True)
axes[1].set_xticks(range(len(blocks)+1))
axes[1].set_xticklabels(range(0, len(blocks)+1))
axes[1].set_xlabel('Feature Extraction until Block...')
axes[1].set_ylabel('AUC')
axes[1].set_title('(2) Linear Probe After Fine-Tuning on Dx', fontweight="bold")

axes[2].tick_params(labelleft=True)
axes[2].set_xlabel('Frozen Blocks')
axes[2].set_ylabel('AUC')
axes[2].set_xticks(range(len(blocks)))
axes[2].set_xticklabels(blocks)
axes[2].set_title("(3) After Fine-Tuning on Dx\nWith Frozen Blocks", fontweight="bold")
fig.savefig('TL_DenseNet_hidden_representations.png')
plt.show()


## Exp i -- Fine tuning only the last layer
# Train the model with a logistic regression

## SimCLR Pretraining
root = '/neurospin/psy_sbox/bd261576/checkpoints/self_supervision/simCLR/DenseNet/N_1600'
exp_3 = ['unsupervised/noisy_spike_motion_crop_DA/', 'age_supervision/noisy_spike_motion_crop_DA',
         'age_supervision/noisy_spike_motion_DA/', 'unsupervised/noisy_spike_motion_cutout_DA',
         'unsupervised/crop_DA', 'unsupervised/cutout_DA', 'unsupervised/affine_DA', 'unsupervised/crop_cutout_DA',
         'age_implicit_supervision/cutout_DA', 'age_implicit_supervision/crop_DA/sigma_5',
         'age_supervision/cutout_DA', 'age_supervision/crop_DA',
         'age_implicit_supervision/crop_DA/sigma_2/BatchSize_64',
         'age_implicit_supervision/crop_DA/sigma_5/BatchSize_32',
         'age_implicit_supervision/cutout_DA/sigma_5/BatchSize_32']

sub_exp_3 = [[], [], [], [], ['window-0.25', 'window-0.5', 'window-0.75', 'window-0.5_unresized'],
             ['window-0.25', 'window-0.5', 'window-0.75'], [], [], ['window-0.5'], ['window-0.75', 'window-0.5'],
             ['window-0.5'], ['window-0.5'], ['window-0.75'], ['window-0.75'], ['window-0.25']]
epochs = [[99], [50], [10], [299, 199, 99, 80, [200, 200, 30, 299], [10, 299, 99], # Exp 0, 1, 2, 3
                             250, 299, 160, [160, 160], 299, 299, 299, 299, 299]
          ]
for i, list_e in enumerate(epochs):
    for k, e in enumerate(list_e):
        if i!= 3 or k < 12:
            continue
        path_spec = 'exp_%i' if i < 3 else 'exp_%i/'+exp_3[k]
        if type(e) == int:
            e = [e]
        for j, sub_e in enumerate(e):
            baseline = History.load(
                os.path.join(root, path_spec%i, 'Validation_DenseNet_HCP_IXI_{sub_exp}0_epoch_{e}.pkl'.
                             format(sub_exp=sub_exp_3[k][j]+'_' if i==3 and len(sub_exp_3[k])>0 else '', e=sub_e)))
            if 'MAE on validation set' in baseline.metrics:
                print('Exp {i} ({exp}: {sub_exp}):\n\tPretext task MAE={mae}, Acc={acc}'.
                      format(i=i, exp=exp_3[k] if i==3 else 'None',
                             sub_exp=sub_exp_3[k][j] + '_' if i == 3 and len(sub_exp_3[k]) > 0 else '',
                             mae=baseline['MAE on validation set'][1][-1],
                             acc=baseline['accuracy on validation set'][1][-1]))
            else:
                print('Exp {i} ({exp}: {sub_exp}):\n\tPretext task Acc={acc}'.
                      format(i=i,  exp=exp_3[k] if i==3 else 'None',
                             sub_exp=sub_exp_3[k][j]+'_' if i==3 and len(sub_exp_3[k])>0 else '',
                             acc=baseline['accuracy on validation set'][1][-1]))
            dbs_training = ['SCZ_VIP'] + (['HCP_IXI'] if i<3 or 'unsupervised' in exp_3[k] else [])
            dbs_test = ['BSNIP'] + (['BSNIP_CTL'] if i<3 or 'unsupervised' in exp_3[k] else [])
            is_reg = [False] + ([True] if i<3 or 'unsupervised' in exp_3[k] else [])
            for l, (db_training, db_test) in enumerate(zip(dbs_training, dbs_test)):
                data_train = get_pickle_obj(
                    os.path.join(root, path_spec%i, 'block_outputs/DenseNet_Block4_{db_train}_{sub_exp}fold0_epoch{e}.pkl'.
                                 format(db_train=db_training,
                                        sub_exp=sub_exp_3[k][j]+'_' if i==3 and len(sub_exp_3[k])>0 else '', e=sub_e)))
                data_test = get_pickle_obj(
                    os.path.join(root, path_spec%i, 'block_outputs/DenseNet_Block4_{db_test}_{sub_exp}fold0_epoch{e}.pkl'.
                                 format(db_test=db_test,
                                        sub_exp=sub_exp_3[k][j]+'_' if i==3 and len(sub_exp_3[k])>0 else '', e=sub_e)))
                y_test, y_pred = train_linear_model(data_train, data_test, reg=is_reg[l])
                if is_reg[l]:
                    print("\t(Age) MAE={mae}".format(mae=np.mean(np.abs(y_test-y_pred))))
                else:
                    print("\t(Dx) AUC={auc}\n\tBacc={bacc}".format(auc=roc_auc_score(y_test, y_pred),
                                                              bacc=balanced_accuracy_score(y_test, y_pred>0.5)))

## Plots the results of several DA with implicit age supervision and varying sigma on HC vs SCZ + HC vs BIP downstream tasks
# n="10K"
# baseline_n = [get_pickle_obj("/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/"
#                                "DenseNet/N_%s/Age/old_training/Test_DenseNet_SCZ_VIP_block4_fold%s_epoch149.pkl"%(n,f))
#                 for f in range(1)]
# Age Pretraining:
# SCZ Results:
# For N=10K, AUC = 0.745 +/- 0.016; Bacc = 0.69 +/- 0.016
# For N=1600: AUC = 0.745 +/- 0.010 ; BAcc = 0.69 +/- 0.015
# For N=500: AUC = 0.74 +/- 0.020 ; BAcc = 0.68 +/- 0.016
# Bipolar Results:
# For N=10K, AUC = 0.63 +/- 0.02; Bacc = 0.60 +/- 0.014
# For N=1600: AUC = 0.64 +/- 0.005; Bacc = 0.59 +/- 0.012
# For N=500: AUC = 0.61 +/- 0.015; BAcc = 0.58 +/- 0.020


root_ = os.path.join('/neurospin/psy_sbox/bd261576/checkpoints/self_supervision/simCLR/DenseNet/N_%s/'
                     'exp_3/age_sex_implicit_supervision/')
augmentations = ['cutout_DA']#, 'crop_DA', 'noisy_spike_motion_crop_DA']
augmentation_names = ['Cutout patch 25% p=100%']#,'Crop 75% p=100%', 'Flip-Blur-Noise-Motion-\nSpike-Ghosting-Crop [75%] p=50%']
hyperparams = ['window-0.25_', 'window-0.75_', '']
dbs = ['SCZ_VIP', 'BIOBD']#, 'HCP_IXI_age', 'HCP_IXI_sex']
prediction_tasks = ['SCZ vs CTL', 'Bipolar vs CTL', 'Age Prediction', 'Sex Prediction']
sigmas = [5] # [0, 0.5, 1, 2, 3, 5]
epochs = [299, 299, 299, 299, 299, 299]
all_N = ['10K']#[500, 1600, '10K']
nb_folds = 1
tested_epochs = list(range(10, 100, 10))# list(range(10, 300, 10))+[299]


baseline = {'SCZ_VIP': {'b_acc': 0.72, 'auc': 0.78},
            'BIOBD': {'b_acc': 0.63, 'auc': 0.68},
           }

baseline_age = {
                '10K': {'SCZ_VIP': {'b_acc': 0.69, 'auc': 0.745},
                        'BIOBD': {'b_acc': 0.60, 'auc': 0.63},
                        'HCP_IXI_age': {'mae': 4.02, 'r': 0.93}},
                1600: {'SCZ_VIP': {'b_acc': 0.69, 'auc': 0.745},
                        'BIOBD': {'b_acc': 0.59, 'auc': 0.64},
                        'HCP_IXI_age': {'mae': 5.65, 'r': 0.86}},
                500: {'SCZ_VIP': {'b_acc': 0.68, 'auc': 0.74},
                      'BIOBD': {'b_acc': 0.58, 'auc': 0.61},
                      'HCP_IXI_age': {'mae': 6.02, 'r': 0.83}}
                }

baseline_sex = {
                '10K': {'SCZ_VIP': {'b_acc': 0.65, 'auc': 0.72},
                        'BIOBD': {'b_acc': 0.55, 'auc': 0.57},
                        'HCP_IXI_sex': {'b_acc': 0.92, 'auc': 0.98}},
                1600: {'SCZ_VIP': {'b_acc': 0.65, 'auc': 0.70},
                        'BIOBD': {'b_acc': 0.55, 'auc': 0.58},
                        'HCP_IXI_sex': {'b_acc': 0.88, 'auc': 0.95}},
                500: {'SCZ_VIP': {'b_acc': 0.68, 'auc': 0.74},
                      'BIOBD': {'b_acc': 0.59, 'auc': 0.62},
                      'HCP_IXI_sex': {'b_acc': 0.82, 'auc': 0.91}}
                }


results = {aug: dict() for aug in augmentations}
results_batch_size = {aug: dict() for aug in augmentations}

# Representation Quality at N={500, 1600} for CROP 75%/CUTOUT25% during contrastive training
res = {N: {aug+hyper: {db: {s: [[get_pickle_obj(os.path.join(root_ % N, aug, 'sigma_' + str(s),
                                                       "Test_DenseNet_%s_%sblock4_fold%i_epoch%s.pkl" % (
                                                           db if (N != '10K' or db != 'HCP_IXI') else 'Big_Healthy',
                                                           hyper, f, e))) for f in range(nb_folds)]
                          for e in tested_epochs]
                      for s in sigmas}
                 for db in dbs
                 } for (aug, hyper) in zip(augmentations, hyperparams)}
       for N in all_N}


metric = {'SCZ_VIP': roc_auc_score, 'BIOBD': roc_auc_score,
          'HCP_IXI_age': lambda y_true, y: np.mean(np.abs(y.ravel()-y_true.ravel())),
          'HCP_IXI_sex': roc_auc_score
          }
res_metric = {N: {aug+hyper: {db: {s: [[metric[db](res[N][aug+hyper][db][s][e][f]['y_true'],
                                             res[N][aug+hyper][db][s][e][f]['y'] if db=='HCP_IXI_age' else
                                                   res[N][aug+hyper][db][s][e][f]['y'][:, 1]) for f in range(nb_folds)]
                              for e in range(len(tested_epochs))]
                          for s in sigmas} for db in dbs}
               for aug,hyper in zip(augmentations, hyperparams)}
           for N in all_N}

for N in all_N:
    fig, big_axes = plt.subplots(len(augmentations), 1, figsize=(5*len(dbs), 5*len(augmentations)), sharey='col', squeeze=False)
    for row, (big_ax, aug_name) in enumerate(zip(big_axes[:,0], augmentation_names), start=1):
        big_ax.set_title(aug_name, fontweight='bold', fontsize=16)
        big_ax.axis('off')
        big_ax._frameon = False
        big_ax.title.set_position([.5, 1.08])
    for k, (aug, hyper) in enumerate(zip(augmentations, hyperparams)):
        for i, (db, task) in enumerate(zip(dbs, prediction_tasks)):
            ax = fig.add_subplot(len(augmentations), len(dbs), k*len(dbs)+i+1)
            for s in sigmas:
                seaborn.lineplot(x=[e for e in tested_epochs for f in range(nb_folds)],
                                 y=[res_metric[N][aug+hyper][db][s][e][f] for e in range(len(tested_epochs))
                                    for f in range(nb_folds)],
                                 marker='o', label='$\sigma=%.1f$' % s, ax=ax)
            ax.set_title('%s ($N_{pretrained}=%s$)' % (task, N))
            ax.set_xlabel('Contrastive training epochs')
            if db == "HCP_IXI_age":
                ax.set_ylabel('MAE')
                ax.axhline(baseline_age[N][db]['mae'], color='red', linestyle='dotted',
                                   label="Standard Age Pretraining")
            elif db == "HCP_IXI_sex":
                ax.set_ylabel('AUC')
                ax.axhline(baseline_sex[N][db]['auc'], color='orange', linestyle='dotted',
                                   label="Standard Sex Pretraining")
            else:
                ax.set_ylabel('AUC')
                ax.axhline(baseline[db]['auc'], color='gray', linestyle='dotted', label="Supervised on %s"%db)
                ax.axhline(baseline_age[N][db]['auc'], color='red', linestyle='dotted',
                                   label="Standard Age Pretraining")
                ax.axhline(baseline_sex[N][db]['auc'], color='orange', linestyle='dotted',
                                   label="Standard Sex Pretraining")
            ax.legend()
    fig.tight_layout(pad=1)
    fig.savefig('scz_bip_perf_contrastive_learning_N%s_AgeSex.png'%N)



# Performance on SCZ vs HC + BIP vs HC when the sigma varies at N_pretraining = 1600 fixed
N_pretraining = 1600
sigmas = [0, 1, 2, 3, 5]
fig, axes = plt.subplots(1, len(dbs), figsize=(len(dbs)*5, 5))
for i, (db, task) in enumerate(zip(dbs, prediction_tasks)):

    for j, (aug, aug_name, hyper) in enumerate(zip(augmentations, augmentation_names, hyperparams)):
        e = -1 if aug != "cutout_DA" else 2
        seaborn.lineplot(x=[s for s in sigmas for _ in range(nb_folds)],
                         y=[res_metric[N_pretraining][aug+hyper][db][s][e][f] for s in sigmas for f in range(nb_folds)],
                         label=aug_name, ax=axes[i], marker='o')
    if db != "HCP_IXI_age" and db != "HCP_IXI_sex":
        axes[i].axhline(baseline[db]['auc'], color='gray', linestyle='dotted', label="Supervised on %s" % db)
        axes[i].axhline(baseline_age[N_pretraining][db]['auc'], color='red', linestyle='dotted',
                    label="Standard Age Pretraining")
        axes[i].axhline(baseline_sex[N_pretraining][db]['auc'], color='orange', linestyle='dotted',
                    label="Standard Sex Pretraining")
    elif db == "HCP_IXI_age":
        axes[i].axhline(baseline_age[N_pretraining][db]['mae'], color='gray', linestyle='dotted',
                    label="Supervision on Age")
    elif db == "HCP_IXI_sex":
        axes[i].axhline(baseline_sex[N_pretraining][db]['auc'], color='gray', linestyle='dotted',
                    label="Supervision on Sex")
    axes[i].set_xlabel("$\sigma$", fontsize=14)
    axes[i].set_ylabel('AUC' if db != "HCP_IXI_age" else "MAE", fontsize=14)
    axes[i].set_title(task, fontweight='bold', fontsize=14)
    axes[i].legend()

fig.subplots_adjust(top=0.80)
fig.suptitle('Age-Aware Contrastive Learning Performance vs $\sigma$', fontweight='bold', fontsize=16)
fig.savefig('contrastive-age-aware_perf_sigma.png')


## Performance on pretraining task (age prediction) + downstream task (only linear probe) as we vary the number
# of pre-trained samples.
# Pre-training: Age Prediction (l1 loss), Age-Aware Contrastive Learning with N = 500, 1600, 10K
# Downstream tasks: SCZ vs HC, Bipolar vs HC

root = "/neurospin/psy_sbox/bd261576/checkpoints/"
pretraining_paths = [
    # Contrastive learning
    "self_supervision/simCLR/DenseNet/N_{n}/exp_3/age_implicit_supervision/cutout_DA/sigma_0",
    # Age-Aware Contrastive
    "self_supervision/simCLR/DenseNet/N_{n}/exp_3/age_implicit_supervision/cutout_DA/sigma_5",
    # Age Pretraining
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/N_{n}/Age",
]
supervised_path = [
    # Supervision from scratch SCZ vs CTL
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/N_500/Dx",
    # Supervision from scratch BIP vs CTL
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/N_500/Bipolar"
]
pretraining_names = ['Contrastive Learning', 'Age-Aware Contrastive Learning', 'Age Supervision ($l1$-loss)']

filename = "Test_DenseNet_{pb}{db}{hyper}_fold{f}_epoch{e}.pkl"
N_pretrainings = [500, 1600, '10K']
epochs = [[299, 30, 30], [299, 30, 30],[299, 299, 299]] # nb pretrainings X #N
folds = [[3, 3, 3], [3, 3, 3], [5, 5, 3]]
hypers = [["_window-0.25_N500_block4", "_window-0.25_block4", "_window-0.25_block4"],
          ["_window-0.25_N500_block4", "_window-0.25_block4", "_window-0.25_block4"],
          ["", "", ""]]
pbs = [["", "", ""], ["", "", ""], ["Age_", "Age_", "Age_"]]
results_age_sup = {
    name: {
        N: [get_pickle_obj(os.path.join(root, path.format(n=N), filename.format(db='HCP_IXI' if N != '10K' else 'Big_Healthy',
                                                                                hyper=hyper, f=f,e=e, pb=pb)))
            for f in range(nb_folds)]
        for (N, e, nb_folds, hyper, pb) in zip(N_pretrainings, epochs[i], folds[i], hypers[i], pbs[i])
    }
    for i, (name, path) in enumerate(zip(pretraining_names, pretraining_paths))
}
# Contrastive: Test_DenseNet_SCZ_VIP_window-0.25_block4_fold2_epoch200.pkl
# Age: Test_DenseNet_SCZ_VIP_block4_fold2_epoch299.pkl
hypers = ["_window-0.25_block4", "_window-0.25_block4", "_block4"]
results_dx = {
        db: {name: {
            N: [get_pickle_obj(
                os.path.join(root, path.format(n=N), filename.format(db=db,  hyper=hyper, f=f, e=e, pb="")))
                for f in range(nb_folds)]
            for (N, e, nb_folds) in zip(N_pretrainings, epochs[i], folds[i])
        }
        for i, (name, path, hyper) in enumerate(zip(pretraining_names, pretraining_paths, hypers))
        }
    for db in ['SCZ_VIP', 'BIOBD']
}
# Test_DenseNet_Bipolar_BIOBD_fold0_epoch299.pkl
baselines = {
    db: [get_pickle_obj(os.path.join(root, p, filename.format(pb=pb, db=db, hyper="",f=f,e=299))) for f in range(5)]
    for (db, pb, p) in zip(["SCZ_VIP", "BIOBD"], ["Dx_", "Bipolar_"], supervised_path)
}
metrics = {
    "MAE": lambda r, y: np.mean(np.abs(np.array(r[y]) - np.array(r['y_true']))),
    "RMSE":  lambda r, y: np.sqrt(np.mean(np.abs(np.array(r[y]) - np.array(r['y_true']))**2)),
    "r": lambda r, y: pearsonr(np.array(r[y]).flatten(), np.array(r['y_true']).flatten())[0],
    "AUC": lambda r, y: roc_auc_score(np.array(r['y_true']), np.array(r[y])[:,1]),
    "AUC_single" : lambda r, y: roc_auc_score(np.array(r['y_true']), np.array(r[y]))
}

fig, axes = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
axes[0].set_title("Age Prediction", fontweight="bold")
axes[0].set_ylim([1, 9])
axes[0].set_ylabel("MAE")
axes[1].set_title("SCZ vs HC", fontweight="bold")
axes[1].set_ylabel("AUC")
axes[1].set_ylim([0.5, 0.9])
axes[2].set_title("BIPOLAR vs HC", fontweight="bold")
axes[2].set(xscale="log")
axes[2].set_xticks([500, 1600, 10**4])
axes[2].set_xticklabels(["$5\\times 10^2$", "$1.6\\times 10^3$", '$10^4$'])
axes[2].set_ylabel("AUC")
axes[2].set_ylim([0.4, 0.7])
axes[2].set_xlabel("$N_{pretrained}$")
for i, name in enumerate(pretraining_names):
    seaborn.lineplot(x=[N if N!='10K' else 10**4 for _ in range(nb_folds) for (N, nb_folds) in zip(N_pretrainings, folds[i])],
                     y=[metrics["MAE"](results_age_sup[name][N][k], "y_pred" if "Supervision" in name else "y")
                        for k in range(nb_folds) for (N, nb_folds) in zip(N_pretrainings, folds[i])],
                     ax=axes[0], label=name, marker='o',err_style='bars', err_kws={'capsize': 10})
    for j, db in enumerate(["SCZ_VIP", "BIOBD"], start=1):
        seaborn.lineplot(x=[N if N!='10K' else 10**4 for _ in range(nb_folds) for (N, nb_folds) in zip(N_pretrainings, folds[i])],
                         y=[metrics["AUC"](results_dx[db][name][N][k], "y")
                            for k in range(nb_folds) for (N, nb_folds) in zip(N_pretrainings, folds[i])],
                         ax=axes[j], label=name, marker='o',err_style='bars', err_kws={'capsize': 10})
        base = np.mean([metrics["AUC_single"](baselines[db][k], "y_pred") for k in range(5)])
        if i==0: axes[j].axhline(base, color='red', linestyle='dotted', label="Baseline")
fig.savefig("unsupervised_perf_age_scz_bip_N_varies.png")


## Performance on SCZ vs HC and BIP vs HC when N_fine_tuned varies (nb of training samples for the fine-tuning)
## Pre-training: Age Prediction (N=10K) or Contrastive Learning (cutout) or
#  Age-Aware Contrastive Learning (cutout) (N=10K) or Age-Sex-Aware Contrastive Learning
seaborn.set_style('darkgrid')
N_pretraining = '10K'
N_finetuning = [100, 300, 500]
root = "/neurospin/psy_sbox/bd261576/checkpoints/"
pretraining_paths = [
    # Contrastive learning
    "self_supervision/simCLR/DenseNet/N_{n}/exp_3/age_implicit_supervision/cutout_DA/sigma_0",
    # Age-Aware Contrastive
    "self_supervision/simCLR/DenseNet/N_{n}/exp_3/age_implicit_supervision/cutout_DA/sigma_5",
    # (Age, Sex)-Aware Contrastive
    "self_supervision/simCLR/DenseNet/N_{n}/exp_3/age_sex_implicit_supervision/cutout_DA/sigma_5",
    # Age Pretraining
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/N_{n}/Age",
    # Supervision from scratch SCZ vs CTL/BIP vs CTL
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/N_{n_finetune}/{pb}",
]
nb_folds = [5, 5, 5, 5, 5]
hyperparams = ['_window-0.25', '_window-0.25', '_window-0.25', '', '_Dx', '']
blocks = ['_block4', '_block4', '_block4', '_block4', '', '']
dbs = ['SCZ_VIP', 'BIOBD']
pbs = ['Dx', 'Bipolar']
exp_names = ["Contrastive Learning", "Age-Aware Contrastive Learning", "(Age, Sex)-Aware Contrastive Learning",
             "Age Supervision", "Supervised on target task"]
all_epochs = [[30], [30], [30], [100], [299]]
results = {db: {name: {} for name in exp_names} for db in dbs}

fig, axes = plt.subplots(1, 2, figsize=(13, 8))
axes[0].set_title("SCZ vs CTL", fontweight="bold", fontsize=14)
axes[1].set_title("BIPOLAR vs CTL", fontweight="bold", fontsize=14)
for i, (pb, db) in enumerate(zip(pbs, dbs)):
    for (name, path, epochs, folds, hyper, b) in zip(exp_names, pretraining_paths, all_epochs, nb_folds, hyperparams, blocks):
        for e in epochs:
            for n_finetuning in N_finetuning:
                filename = "Test_DenseNet_{db}{hyper}_N{n_finetuning}{block}_ModelFold0_fold{f}_epoch{e}.pkl" if name != "Supervised on target task"\
                            else "Test_DenseNet_{pb}_{db}_fold{f}_epoch{e}.pkl"
                results[db][name][n_finetuning] = [get_pickle_obj(
                    os.path.join(root, path.format(n=N_pretraining, n_finetune=n_finetuning, pb=pb), filename.
                                 format(db=db, pb=pb, hyper=hyper, n_finetuning=n_finetuning, block=b, f=fold,e=e)))
                                                    for fold in range(folds)]
            ax = seaborn.lineplot(x=[n for n in N_finetuning for _ in range(folds)],
                                  y=[roc_auc_score(results[db][name][n][f]['y_true'],
                                              np.array(results[db][name][n][f]['y_pred']) if name=="Supervised on target task" else
                                              results[db][name][n][f]['y'][:, 1])
                                for n in N_finetuning for f in range(folds)],
                             ax=axes[i],
                             err_style='bars', err_kws={'capsize': 10},
                             label=name)
    axes[i].set_ylabel("AUC")
    axes[i].set_xlabel("$N_{target}$")

fig.savefig("unsupervised_perf_scz_bip_AgeSex_N10K.png")


## t-SNE visualization of BSNIP according to different pre-trainings

# 1) take all SCZ/BIP/CTL from BSNIP and  plots t-SNE for i) Sup Contrastive Loss (sigma=5),
# ii) Contrastive Loss (sigma=0) and iii) Age Prediction (l1-loss)
nb_folds = 3
sigmas = [0, 5]
tested_epochs = [30]
features_contrastive = {
    s: [[get_pickle_obj("/neurospin/psy_sbox/bd261576/checkpoints/self_supervision/simCLR/DenseNet/N_10K/"
                        "exp_3/age_implicit_supervision/cutout_DA/sigma_{s}/"
                        "features_DenseNet_BSNIP_window-0.25_fold{f}_epoch{e}.pkl".format(s=s, f=f, e=e))
         for f in range(nb_folds)] for e in tested_epochs] for s in sigmas}
# features_contrastive_scz = [
#     [get_pickle_obj("/neurospin/psy_sbox/bd261576/checkpoints/self_supervision/simCLR/DenseNet/N_500/"
#                     "exp_3/age_dx_implicit_supervision/cutout_DA/sigma_5/"
#                     "features_DenseNet_BSNIP_window-0.25_fold{f}_epoch299.pkl".format(f=f))
#      for f in range(nb_folds)]
# ]
tested_epochs =  [299]
features_age = [
    [get_pickle_obj("/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/N_10K/Age/"
                    "features_DenseNet_BSNIP_fold{f}_epoch{e}.pkl".format(f=f, e=e))
     for f in range(nb_folds)] for e in tested_epochs]
tested_epochs =  [299]
features_dx = {pb: [
    [get_pickle_obj("/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/N_500/{pb}/"
                    "features_DenseNet_BSNIP_fold{f}_epoch{e}.pkl".format(pb=pb, f=f, e=e))
     for f in range(nb_folds)] for e in tested_epochs]
    for pb in ['Dx', 'Bipolar']
}

seaborn.set_style('white')
fig, axes = plt.subplots(2, 3, figsize=(19, 13))
l_mapping = {0: 'control', 1: "schizophrenia", 2: "bipolar"}
mapping = lambda x: l_mapping[x]

for i, s in enumerate(sigmas):
    epoch_i = -1
    plot_data_reduced(features_contrastive[s][epoch_i][0]['y'][:, 0, :],
                      ax=axes[0, i],
                      labels=features_contrastive[s][epoch_i][0]['y_true'][:, 0, 1],
                      labels_title="Dx",
                      continuous_labels=features_contrastive[s][epoch_i][0]['y_true'][:, 0, 0]**2/20,
                      inv_continuous_labels_tf=lambda x: np.sqrt(x*20),
                      continuous_labels_title="Age",
                      labels_mapping_fn=mapping, cmap=plt.cm.winter,
                      title="Contrastive Learning" if s==0 else "Age-Aware Contrastive Learning",
                      reduction='t_sne', metric='cosine', perplexity=30)
epoch_i = -1
mask = (features_dx['Dx'][epoch_i][0]['y_true'][:,0,1] < 2)
plot_data_reduced(features_dx['Dx'][epoch_i][0]['y'][mask][:, 0, :],
                  ax=axes[1,0],
                  labels=features_dx['Dx'][epoch_i][0]['y_true'][mask][:, 0, 1],
                  labels_title="Dx",
                  continuous_labels=features_dx['Dx'][epoch_i][0]['y_true'][mask][:, 0, 0]**2/20,
                  inv_continuous_labels_tf=lambda x: np.sqrt(x * 20),
                  continuous_labels_title="Age",
                  labels_mapping_fn=mapping, cmap=plt.cm.winter,
                  title="Supervision on SCZ vs HC",
                  reduction='t_sne', metric='cosine', perplexity=30)
mask = (features_dx['Dx'][epoch_i][0]['y_true'][:,0,1] != 1)
plot_data_reduced(features_dx['Bipolar'][epoch_i][0]['y'][mask][:, 0, :],
                  ax=axes[1,1],
                  labels=features_dx['Bipolar'][epoch_i][0]['y_true'][mask][:, 0, 1],
                  labels_title="Dx",
                  continuous_labels=features_dx['Bipolar'][epoch_i][0]['y_true'][mask][:, 0, 0]**2/20,
                  inv_continuous_labels_tf=lambda x: np.sqrt(x * 20),
                  continuous_labels_title="Age",
                  labels_mapping_fn=mapping, cmap=plt.cm.winter,
                  title="Supervision on BIPOLAR vs HC",
                  reduction='t_sne', metric='cosine', perplexity=30)
plot_data_reduced(features_age[epoch_i][0]['y'][:, 0, :],
                  ax=axes[1,2],
                  labels=features_age[epoch_i][0]['y_true'][:, 0, 1],
                  labels_title="Dx",
                  continuous_labels=features_age[epoch_i][0]['y_true'][:, 0, 0]**2/20,
                  inv_continuous_labels_tf=lambda x: np.sqrt(x * 20),
                  continuous_labels_title="Age",
                  labels_mapping_fn=mapping, cmap=plt.cm.winter,
                  title="Supervision on Age",
                  reduction='t_sne', metric='cosine', perplexity=30)

fig.savefig("t-SNE_contrastive-age-aware.png")

## Plots losses
# fig, axes = plot_losses(h, h_val,
#             patterns_to_del=['validation_', ' on validation set'],
#             metrics=['accuracy', 'loss'],
#             experiment_names=['$\sigma=0$', '$\sigma=0.5$', '$\sigma=1$', '$\sigma=2$', '$\sigma=3$', '$\sigma=5$'],
#             ylabels={'accuracy': 'Accuracy', 'loss': 'loss'},
#             ylim={'loss': [0,7.5]},
#             titles={'loss': 'Flip-Blur-Noise-Motion-Spike-Ghosting-Crop [75%] p=50% With Age Prior $\sigma$'},
#             same_plot=True,
#             figsize=(15,15), saving_path='losses_simCLR_implicit_age_sup_all.png')


## Age + Sex/Age Pretraining
root = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/'
pb = ['Age', 'Sex', 'Sex_Age']
nb_samples = ['', 'N_10K']
nb_epochs = [99, 149]
dbs = ['HCP_IXI', 'Big_Healthy']

for p in pb:
    for (samples,e, db) in zip(nb_samples, nb_epochs, dbs):
        baseline = History.load(os.path.join(root, samples, '{pb}/Validation_DenseNet_{pb}_{db}_0_epoch_{e}.pkl'.
                                             format(pb=p, db=db, e=e)))
        data_train = get_pickle_obj(
            os.path.join(root, samples, '{pb}/block_outputs/DenseNet_Block4_SCZ_VIP_fold0_epoch{e}.pkl'.
                         format(pb=p, e=e)))
        data_test = get_pickle_obj(
            os.path.join(root, samples, '{pb}/block_outputs/DenseNet_Block4_BSNIP_fold0_epoch{e}.pkl'.
                         format(pb=p, e=e)))
        y_test, y_pred = train_linear_model(data_train, data_test)
        loss, p_auc = None, None
        if 'validation_loss' in baseline.metrics:
            loss = baseline['validation_loss'][1][-1]
        if 'roc_auc on validation set' in baseline.metrics:
            p_auc = baseline['roc_auc on validation set'][1][-1]
        print('Exp {p}:\n\tPretext task loss={loss}, AUC={pretext_auc}\n\tAUC={auc}, Bacc={bacc}'.
              format(p=p, loss=loss,
                     pretext_auc=p_auc,
                     auc=roc_auc_score(y_test, y_pred),
                     bacc=balanced_accuracy_score(y_test, y_pred>0.5)))

## Color histogram of Images cropped and with cutout
from pynet.transforms import *
from pynet.augmentation import cutout
from nilearn.plotting import plot_anat
from nibabel import Nifti1Image

input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128]), Normalize()]
manager = DataManager(CONFIG['cat12']['input_path'], CONFIG['cat12']['metadata_path'],
                      batch_size=4,
                      number_of_folds=5,
                      custom_stratification=CONFIG['db']["healthy"],
                      input_transforms=input_transforms,
                      pin_memory=True,
                      drop_last=False,
                      device='cuda')

train_iter = iter(manager.get_dataloader(train=True).train)
X_train = next(train_iter).inputs.detach().cpu().numpy()
tfs = [lambda x: x,
       Crop((1, 64, 64, 64), "random", resize=True),
       Crop((1, 64, 64, 64), "random", resize=False, keep_dim=True),
       lambda x: cutout(x, patch_size=32, localization=(64, 64, 80))]
reps = [1, 2, 2, 2]
tf_names = ['Original Image', 'Crop+Resize', 'Crop', 'Cutout']

cols = 1 + np.sum(reps)
fig, axes = plt.subplots(len(X_train) + 1, cols, figsize=(cols * 5, (len(X_train) + 1) * 5), sharey=True)
for i, x in enumerate(X_train, start=1):
    axes[i, 0].text(0.5, 0.5, 'Image %i' % i, fontsize=20)
    axes[i, 0].axis('off')
    for c, (tf, rep, name) in enumerate(zip(tfs, reps, tf_names)):
        for j in range(rep):
            current_j = np.concatenate([[0], np.cumsum(reps)])[c] + j + 1
            x_tf = tf(x)
            histogram, bin_edges = np.histogram(x_tf, bins=100, range=(-0.7, 0.1))
            axes[i, current_j].plot(bin_edges[:-1], histogram / 128 ** 3)
            axes[i, current_j].set_ylim([0, 1])
            axes[0, current_j].set_title('{name} {r}'.format(name=name, r=j + 1) if rep > 1 else name, fontsize=20)
            if i == 1:
                axes[0, 0].axis('off')
                plot_anat(Nifti1Image(x_tf[0], np.eye(4)), cut_coords=[50], display_mode='x',
                          axes=axes[0, current_j], annotate=True,
                          draw_cross=False, black_bg='auto', vmax=5, vmin=0)
fig.savefig('color_histogram_crop_cutout.png')


## Sanity Check between Linear Regression and Fine-Tuning
from pynet.models.densenet import *
from sklearn.linear_model import LogisticRegression
from pynet.transforms import *
import torch

# Loads the pretrained model and compute the linear regression on the features given by this model
age_pretraining = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Age/' \
                  'DenseNet_Age_HCP_IXI_0_epoch_99_without_classifier.pth'
model = densenet121(num_classes=1, out_block="block4").to('cuda')
pretrained_weights = torch.load(age_pretraining)
dx_mapping = LabelMapping(schizophrenia=1, control=0)
model.load_state_dict(pretrained_weights['model'], strict=False)
model.eval()
input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128]), Normalize()]
manager = DataManager(CONFIG['cat12']['input_path'], CONFIG['cat12']['metadata_path'],
                      batch_size=4,
                      number_of_folds=5,
                      labels=["diagnosis"],
                      labels_transforms=[dx_mapping],
                      custom_stratification=CONFIG['db']["tiny_scz_kfolds"],
                      input_transforms=input_transforms,
                      pin_memory=True,
                      drop_last=False,
                      device='cuda')
loader = manager.get_dataloader(train=True, test=True)
X_train, y_train = [], []
for dataitem in loader.train:
    X_train.extend(model(dataitem.inputs.to('cuda')).detach().cpu().numpy())
    y_train.extend(dataitem.labels.detach().cpu().numpy())
X_test, y_test = [], []
for dataitem in loader.test:
    X_test.extend(model(dataitem.inputs.to('cuda')).detach().cpu().numpy())
    y_test.extend(dataitem.labels.detach().cpu().numpy())

linear_model = LogisticRegression(solver='newton-cg', C=1/(5e-5)).fit(np.array(X_train), np.array(y_train))
y_pred = linear_model.predict_proba(np.array(X_test))[:,1]
print('AUC for linear model on top of DenseNet_Block4: %f'%roc_auc_score(y_test, y_pred))



