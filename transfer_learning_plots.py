import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pynet.utils import get_pickle_obj
from pynet.plotting.image import plot_data_reduced
from pynet.datasets.core import DataManager
from json_config import CONFIG
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
seaborn.set_style("darkgrid")

## Analysis of the generalization power of DenseNet pretrained on different pb (supervised and unsupervised) for dx classification

blocks = ["block1", "block2", "block3", "block4"]
root = "/neurospin/psy_sbox/bd261576/checkpoints"

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10,5))


### Compute the baseline: only logistic regression on the input images

# data, meta = np.load(CONFIG['input_path'], mmap_mode='r'), pd.read_csv(CONFIG['metadata_path'], sep='\t')
# mask_train = DataManager.get_mask(meta, CONFIG['db']['tiny_scz_kfolds']['train'])
# mask_test = DataManager.get_mask(meta, CONFIG['db']['tiny_scz_kfolds']['test'])
# model = LogisticRegression(solver='liblinear').fit(data[mask_train].reshape(mask_train.sum(),-1),
#                                                    (meta[mask_train].diagnosis=='schizophrenia'))
# pred = model.predict_proba(data[mask_test].reshape(mask_test.sum(),-1))[:,1]
# true = (meta[mask_test].diagnosis=='schizophrenia')
# axes[0].scatter(0, roc_auc_score(true, pred), marker='*', color='red', label='Baseline')
# axes[1].scatter(0, balanced_accuracy_score(true, pred>0.5), marker='*', color='red', label='Baseline')

# AUC = 0.7914432989690722, bAcc = 0.7039690721649485
axes[0].scatter(0, 0.7914432989690722, marker='*', color='red', label='Baseline')
axes[1].scatter(0, 0.7039690721649485, marker='*', color='red', label='Baseline')


paths = [
    #          "self_supervision/simCLR/DenseNet/exp_1/block_outputs",
    #          "scz_prediction/Benchmark_Transfer/Self_Supervision/schizconnect_vip/SimCLR_Exp1/block_outputs",
    #          "self_supervision/simCLR/DenseNet/exp_2/block_outputs",
    #          "scz_prediction/Benchmark_Transfer/Self_Supervision/schizconnect_vip/SimCLR_Exp2/DenseNet/block_outputs",
    # "regression_age_sex/Benchmark_IXI_HCP/DenseNet/Sex_Age/block_outputs",
    # "scz_prediction/Benchmark_Transfer/Sex_Age_Pretraining/schizconnect_vip/DenseNet/block_outputs",
    # "regression_age_sex/Benchmark_IXI_HCP/DenseNet/Age/block_outputs",
    # "scz_prediction/Benchmark_Transfer/Age_Pretraining/schizconnect_vip/DenseNet/block_outputs",
    # "self_supervision/simCLR/DenseNet/exp_3/noisy_spike_motion_crop_DA/block_outputs",
    # "scz_prediction/Benchmark_Transfer/Self_Supervision/schizconnect_vip/SimCLR_Exp3/noisy_spike_motion_crop_DA/block_outputs",
    # "regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/block_outputs"
    "self_supervision/simCLR/DenseNet/exp_3/noisy_spike_motion_crop_DA/unsupervised/block_outputs"
]

training_filenames = [
    # "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch99.pkl",
    # "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch49.pkl",
    # "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch10.pkl",
    # "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch49.pkl",
    # "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch99.pkl",
    # "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch49.pkl",
    # "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch99.pkl",
    # "DenseNet_Block{b}_Dx_SCZ_VIP_{f}_epoch49.pkl",
    # "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch199.pkl",
    # "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch49.pkl",
    # "DenseNet_Block{b}_Dx_SCZ_VIP_{f}_epoch49.pkl"
    "DenseNet_Block{b}_SCZ_VIP_fold{f}_epoch299.pkl"
]
testing_filenames = [
    # "DenseNet_Block{b}_BSNIP_fold{f}_epoch99.pkl",
    # "DenseNet_Block{b}_BSNIP_fold{f}_epoch49.pkl",
    # "DenseNet_Block{b}_BSNIP_fold{f}_epoch10.pkl",
    # "DenseNet_Block{b}_BSNIP_fold{f}_epoch49.pkl",
    # "DenseNet_Block{b}_BSNIP_fold{f}_epoch99.pkl",
    # "DenseNet_Block{b}_BSNIP_fold{f}_epoch49.pkl",
    # "DenseNet_Block{b}_BSNIP_fold{f}_epoch99.pkl",
    # "DenseNet_Block{b}_Dx_BSNIP_{f}_epoch49.pkl",
    # "DenseNet_Block{b}_BSNIP_fold{f}_epoch199.pkl",
    # "DenseNet_Block{b}_BSNIP_fold{f}_epoch49.pkl",
    # "DenseNet_Block{b}_Dx_BSNIP_{f}_epoch49.pkl"
    "DenseNet_Block{b}_BSNIP_fold{f}_epoch299.pkl"
]

exp_names = [
             # "SimCLR\n (swap+flip+blur+motion+noise+spike)",
             # "SimCLR Transfer (1)",
             # "SimCLR\n (flip+blur+motion+noise+spike+ghosting)",
             # "SimCLR Transfer (2)",
             # "Sex+Age",
             # "Sex+Age Transfer",
             # "Age",
             # "Age Transfer",
             # "Age Supervision SimCLR",
             # "SimCLR Transfer"
             # "Trained on Dx",
    "SimCLR"
             ]
nb_folds = [1]#[1, 5, 1, 5, 1, 5, 5]
colors = ["blue", "green", "orange", "red"]

metrics = {'auc': {exp: {b: [] for b in blocks} for exp in exp_names},
           'balanced_accuracy': {exp: {b: [] for b in blocks} for exp in exp_names}}
#pca_fig, pca_axes = plt.subplots(len(paths)//2, len(blocks), figsize=(4*len(blocks), 4*len(paths)//2), squeeze=False)
for i, (path, train_file, test_file, exp) in enumerate(zip(paths, training_filenames, testing_filenames, exp_names)):
    for j, b in enumerate(blocks):
        for fold in range(nb_folds[i]):

            # Train the model with a logistic regression
            data = get_pickle_obj(os.path.join(root, path, train_file.format(b=j+1, f=fold)))
            X, y = np.array(data['y']).reshape(len(data['y']),-1), np.array(data['y_true']).ravel()
            model = LogisticRegression(solver='liblinear').fit(X, y)

            # Test the model on BSNIP
            testing_data = get_pickle_obj(os.path.join(root, path, test_file.format(b=j+1, f=fold)))
            X_test, y_test = np.array(testing_data['y']).reshape(len(testing_data['y']),-1), \
                             np.array(testing_data['y_true']).ravel()
            y_pred = model.predict_proba(X_test)[:,1]

            # if fold == 0 and i%2 == 0:
            #     plot_data_reduced(X_test, ['control' if y_t==0 else 'schizophrenia' for y_t in y_test],
            #                       ax=pca_axes[i//2,j])
            #     if i==0:
            #         pca_axes[i//2,j].set_title(b)
            #     if j==0:
            #         pca_axes[i//2,j].set_ylabel(exp)

            metrics['auc'][exp][b].append(roc_auc_score(y_test, y_pred))
            metrics['balanced_accuracy'][exp][b].append(balanced_accuracy_score(y_test, y_pred>0.5))
    seaborn.lineplot(x=[b+1 for b in range(len(blocks)) for _ in range(nb_folds[i])],
                     y=[metrics['auc'][exp][b][k] for b in blocks for k in range(nb_folds[i])],
                     label=exp, ax=axes[0], marker='o', color=colors[i//2])
    seaborn.lineplot(x=[b+1 for b in range(len(blocks)) for _ in range(nb_folds[i])],
                     y=[metrics['balanced_accuracy'][exp][b][k] for b in blocks for k in range(nb_folds[i])],
                     label=exp, ax=axes[1], marker='o', color=colors[i//2])
    if i%2 == 0 and i != len(paths)-1: # dashed line
        axes[0].lines[-1].set_linestyle('--')
        axes[1].lines[-1].set_linestyle('--')

axes[0].set_xlabel('Block')
axes[0].set_ylabel('AUC')
axes[0].set_xticks(range(len(blocks)+1))
axes[0].set_xticklabels(range(0, len(blocks)+1))
axes[1].tick_params(labelleft=True)
axes[1].set_xticks(range(len(blocks)+1))
axes[1].set_xticklabels(range(0, len(blocks)+1))
axes[1].set_xlabel('Block')
axes[1].set_ylabel('Balanced Accuracy')
plt.show()
plt.savefig('densenet_hidden_representations.png')


### Plots the AUC for different pretrainings as a function of the number of blocks frozen
fig, ax = plt.subplots(1,1, figsize=(5,5))
blocks = [ "Block1", "Block2", "Block3", "Block4"]
root = "/neurospin/psy_sbox/bd261576/checkpoints/scz_prediction/Benchmark_Transfer"
pretrainings = ["No_Pretraining", "Age_Pretraining", "Sex_Pretraining", "Sex_Age_Pretraining", "Self_Supervision"]
exp_names = ["Random", "Age Init", "Sex Init", "Age+Sex Init", "Age Sup. SimCLR Init"]
settings = ["DenseNet", "DenseNet", "DenseNet", "DenseNet", "SimCLR_Exp3/noisy_spike_motion_crop_DA/age_supervision"]
db = "schizconnect_vip"
network = "DenseNet"

# Plots the reference line (linear baseline)
ax.plot(range(len(blocks)), len(blocks)*[0.7914432989690722], linestyle="--", color='gray', label='Logistic Regression')

metrics = {exp: {b: dict() for b in blocks} for exp in exp_names}
for (exp_name, pretraining, setting) in zip(exp_names, pretrainings, settings):
    for block in blocks:
        path = os.path.join(root, pretraining, db, setting)
        if block == "None":
            tests = [get_pickle_obj(os.path.join(path, "Test_DenseNet_Dx_SCZ_VIP_fold{k}_epoch49.pkl".format(k=k)))
                     for k in range(5)]
        else:
           tests = [get_pickle_obj(os.path.join(path, block+'_frozen',
                                    "Test_DenseNet_Dx_SCZ_VIP_fold{k}_epoch49.pkl".format(k=k))) for k in range(5)]
        metrics[exp_name][block]["auc"] = [roc_auc_score(t['y_true'], t['y_pred']) for t in tests]

    seaborn.lineplot(x=[i for i, _ in enumerate(blocks) for _ in range(5)],
                     y=[metrics[exp_name][b]['auc'][k] for b in blocks for k in range(5)],
                     label=exp_name, marker='o', ax=ax)

ax.set_xlabel('Frozen Blocks')
ax.set_ylabel('AUC')
ax.set_xticks(range(len(blocks)))
ax.set_xticklabels(blocks)
ax.set_title("DenseNet Performance with Frozen Blocks on Dx", fontweight="bold")
plt.show()
plt.savefig('densenet_transfer_block_frozen.png')
