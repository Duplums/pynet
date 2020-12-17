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

## Plots the results of several DA with implicit age supervision and varying sigma on HC vs SCZ downstream task
# n=1600
# baseline_n = [get_pickle_obj("/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/"
#                                "DenseNet/N_%i/Age/Test_DenseNet_BIOBD_block4_fold%s_epoch299.pkl"%(n,f))
#                 for f in range(5)]
# Age Pretraining:
# SCZ Results:
# For N=1600: AUC = 0.745 +/- 0.010 ; BAcc = 0.69 +/- 0.015
# For N=500: AUC = 0.74 +/- 0.020 ; BAcc = 0.68 +/- 0.016
# Bipolar Results:
# For N=1600: AUC = 0.64 +/- 0.005; Bacc = 0.59 +/- 0.012
# For N=500: AUC = 0.61 +/- 0.015; BAcc = 0.58 +/- 0.020


root_ = os.path.join('/neurospin/psy_sbox/bd261576/checkpoints/self_supervision/simCLR/DenseNet/N_%s/'
                     'exp_3/age_implicit_supervision/')
augmentations = ['crop_DA'] #['cutout_DA', 'crop_DA', 'noisy_spike_motion_crop_DA']
augmentation_names = ['Crop 75% p=100%']#['Cutout patch 25% p=100%', 'Crop 75% p=100%',
                                        #'Flip-Blur-Noise-Motion-\nSpike-Ghosting-Crop [75%] p=50%']
hyperparams = ['window-0.75_'] #['window-0.25_', 'window-0.75_', '']
dbs = ['SCZ_VIP', 'BIOBD', 'HCP_IXI']
sigmas = [0, 0.5, 1, 2, 3, 5]
epochs = [299, 299, 299, 299, 299, 299]
all_N = [500, 1600]
nb_folds = 3
tested_epochs = list(range(10, 300, 10)) + [299]


baseline = {'SCZ_VIP': {'b_acc': 0.72, 'auc': 0.78},
            'BIOBD': {'b_acc': 0.63, 'auc': 0.68},
           }

baseline_age = {1600: {'SCZ_VIP': {'b_acc': 0.69, 'auc': 0.745},
                        'BIOBD': {'b_acc': 0.59, 'auc': 0.64},
                        'HCP_IXI': {'mae': 5.65, 'r': 0.86}},
                500: {'SCZ_VIP': {'b_acc': 0.68, 'auc': 0.74},
                      'BIOBD': {'b_acc': 0.58, 'auc': 0.61},
                      'HCP_IXI': {'mae': 6.02, 'r': 0.83}}
                }
results = {aug: dict() for aug in augmentations}
results_batch_size = {aug: dict() for aug in augmentations}

# Representation Quality at N={500, 1600} for CROP 75%/CUTOUT25% during contrastive training
res = {N: {aug+hyper: {db: {s: [[get_pickle_obj(os.path.join(root_ % N, aug, 'sigma_' + str(s),
                                                       "Test_DenseNet_%s_%sblock4_fold%i_epoch%s.pkl" % (
                                                           db, hyper, f, e))) for f in range(nb_folds)]
                          for e in tested_epochs]
                      for s in sigmas}
                 for db in dbs
                 } for (aug, hyper) in zip(augmentations, hyperparams)}
       for N in all_N}


metric = {'SCZ_VIP': roc_auc_score, 'BIOBD': roc_auc_score,
          'HCP_IXI': lambda y_true, y: np.mean(np.abs(y.ravel()-y_true.ravel()))}
res_metric = {N: {aug+hyper: {db: {s: [[metric[db](res[N][aug+hyper][db][s][e][f]['y_true'],
                                             res[N][aug+hyper][db][s][e][f]['y'] if db=='HCP_IXI' else
                                                   res[N][aug+hyper][db][s][e][f]['y'][:, 1]) for f in range(nb_folds)]
                              for e in range(len(tested_epochs))]
                          for s in sigmas} for db in dbs}
               for aug,hyper in zip(augmentations, hyperparams)}
           for N in all_N}

for N in all_N:
    fig, big_axes = plt.subplots(len(augmentations), 1, figsize=(12, 15), sharey='col', squeeze=False)
    for row, (big_ax, aug_name) in enumerate(zip(big_axes[:,0], augmentation_names), start=1):
        big_ax.set_title(aug_name, fontweight='bold', fontsize=16)
        big_ax.axis('off')
        big_ax._frameon = False
        big_ax.title.set_position([.5, 1.08])
    for k, (aug, hyper) in enumerate(zip(augmentations, hyperparams)):
        for i, db in enumerate(dbs):
            ax = fig.add_subplot(len(augmentations), len(dbs), k*len(dbs)+i+1)
            for s in [0, 1, 2, 3, 5]:
                seaborn.lineplot(x=[e for e in tested_epochs for f in range(nb_folds)],
                                 y=[res_metric[N][aug+hyper][db][s][e][f] for e in range(len(tested_epochs))
                                    for f in range(nb_folds)],
                                 marker='o', label='$\sigma=%.1f$' % s, ax=ax)
            ax.set_title('Performance on %s with $N_{pretrained}=%i$ ' % (db, N))
            ax.set_xlabel('Contrastive training epochs')
            if db == "HCP_IXI":
                ax.set_ylabel('MAE')
                ax.axhline(baseline_age[N][db]['mae'], color='red', linestyle='dotted',
                                   label="Standard Age Pretraining")
            else:
                ax.set_ylabel('AUC')
                ax.axhline(baseline[db]['auc'], color='gray', linestyle='dotted', label="Supervised on %s"%db)
                ax.axhline(baseline_age[N][db]['auc'], color='red', linestyle='dotted',
                                   label="Standard Age Pretraining")
            ax.legend()
    fig.tight_layout(pad=1)
    fig.savefig('scz_bip_perf_contrastive_learning_N%i.png'%N)

# Final performance on SCZ vs HC and BIPOLAR vs HC at 300 epochs for all the transformations and N={500, 1600}
# training samples in pretraining.
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey='row')
for i,db in enumerate(dbs):
    for j, (aug, aug_name, hyper) in enumerate(zip(augmentations, augmentation_names, hyperparams)):
        for sigma, e in zip(sigmas, epochs):
            batch_size = ['']
            # if sigma == 5: ## Stress out that the batch size does not change the representation
            #     batch_size = ['', 'BatchSize_32']
            for b_size in batch_size:
                if b_size != '': e = 299
                h_val = History.load(os.path.join(root_, aug, 'sigma_'+ str(sigma), b_size,
                                                  "Validation_DenseNet_HCP_IXI_%s2_epoch_%s.pkl"%(hyper, e))).to_dict()
                res = [get_pickle_obj(os.path.join(root_, aug, 'sigma_'+str(sigma),
                                     "Test_DenseNet_%s_%sblock4_fold%i_epoch%s.pkl"%(db, hyper, f, e))) for f in range(3)]
                roc_auc = [roc_auc_score(res[k]['y_true'], res[k]['y'][:,1]) for k in range(3)]

                if b_size == '': # Implicit batch size == 16
                    results[aug][sigma] = {'acc_pretext': np.mean([h_val['accuracy on validation set'][k][-1] for k in range(3)]),
                                           'auc': roc_auc
                                           }
                # else:
                #     results_batch_size[aug] = {
                #         'acc_pretext': h_val['accuracy on validation set'][1][-1],
                #         'auc': roc_auc_score(y_test, y_pred),
                #         'b_acc': balanced_accuracy_score(y_test, y_pred > 0.5)}
        axes[i,j].errorbar([results[aug][s]['acc_pretext'] for s in sigmas],
                         [np.mean(results[aug][s]['auc']) for s in sigmas],
                         yerr=[np.std(results[aug][s]['auc']) for s in sigmas],
                         marker='o',
                        label='Batch size 16')
        # axes[i].scatter(results_batch_size[aug]['acc_pretext'], results_batch_size[aug]['auc'], marker='*',
        #                 label='Batch Size 32')
        # axes[i].annotate("$\sigma=5.0$", (0.001 + results_batch_size[aug]['acc_pretext'],
        #                                   0.001 + results_batch_size[aug]['auc']))
        for s in sigmas:
            axes[i,j].annotate("$\sigma=%.1f$"%s,
                               (0.001+results[aug][s]['acc_pretext'], 0.001+np.mean(results[aug][s]['auc'])))

        axes[i,j].axhline(baseline[db]['auc'], color='gray', linestyle='dotted', label="Supervised on SCZ vs HC")
        axes[i,j].axhline(baseline_age[N][db]['auc'], color='red', linestyle='dotted', label="Standard Age Pretraining")
        axes[i,j].set_xlabel('Accuracy on pretext task', fontsize=14)
        axes[i,j].set_ylabel('AUC on %s'%db, fontsize=14)
        axes[i,j].set_title(aug_name, fontweight='bold', fontsize=14)
        axes[i,j].legend()
axes[0,0].set_ylim([0.7, 0.8])
axes[1,0].set_ylim([0.6, 0.7])
fig.subplots_adjust(top=0.80)
fig.suptitle('Unsupervised SimCLR With Age Similarity Prior $\sigma$', fontweight='bold', fontsize=16)
fig.savefig('simclr_perf_implicit_age_sup.png')

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
       lambda x: cutout(x, patch_size=40)]
reps = [1, 2, 2, 2]
tf_names = ['Original Image', 'Crop+Resize', 'Crop', 'Cutout']

cols = 1+np.sum(reps)
fig, axes = plt.subplots(len(X_train)+1, cols, figsize=(cols*5, (len(X_train)+1)*5), sharey=True)
for i, x in enumerate(X_train, start=1):
    axes[i, 0].text(0.5, 0.5, 'Image %i'%i, fontsize=20)
    axes[i, 0].axis('off')
    for c, (tf, rep, name) in enumerate(zip(tfs, reps, tf_names)):
        for j in range(rep):
            current_j = np.concatenate([[0], np.cumsum(reps)])[c]+j+1
            x_tf = tf(x)
            histogram, bin_edges = np.histogram(x_tf, bins=100, range=(-0.7, 0.1))
            axes[i, current_j].plot(bin_edges[:-1], histogram/128**3)
            axes[i, current_j].set_ylim([0, 1])
            axes[0, current_j].set_title('{name} {r}'.format(name=name, r=j+1) if rep > 1 else name, fontsize=20)
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



