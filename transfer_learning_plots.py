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
import pickle
seaborn.set_style("darkgrid")

## Analysis of the generalization power of DenseNet pretrained on different pb (supervised and unsupervised) for dx classification

blocks = ["block1", "block2", "block3", "block4"]
root = "/neurospin/psy_sbox/bd261576/checkpoints"

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15,5))


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
# Plots the reference line (linear baseline)
axes[0].axhline(0.7914432989690722, linestyle="--", color='gray', linewidth=.7, label='Logistic Regression')
axes[1].axhline(0.7914432989690722, linestyle="--", color='gray', linewidth=.7, label='Logistic Regression')
axes[2].axhline(0.7914432989690722, linestyle="--", color='gray', linewidth=.7, label='Logistic Regression')


paths = [
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/Age/block_outputs",
    "scz_prediction/Benchmark_Transfer/Age_Pretraining/schizconnect_vip/DenseNet/block_outputs",
    "regression_age_sex/Benchmark_IXI_HCP/DenseNet/Sex_Age/block_outputs",
    "scz_prediction/Benchmark_Transfer/Sex_Age_Pretraining/schizconnect_vip/DenseNet/block_outputs",
    "self_supervision/simCLR/DenseNet/exp_3/noisy_spike_motion_crop_DA/age_supervision/block_outputs",
    "scz_prediction/Benchmark_Transfer/Self_Supervision/schizconnect_vip/SimCLR_Exp3/noisy_spike_motion_crop_DA/age_supervision/block_outputs",
    "self_supervision/simCLR/DenseNet/exp_3/noisy_spike_motion_crop_DA/unsupervised/block_outputs",
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
    # for j, b in enumerate(blocks):
    #     for fold in range(nb_folds[i]):
    #
    #         # Train the model with a logistic regression
    #         data = get_pickle_obj(os.path.join(root, path, train_file.format(b=j+1, f=fold)))
    #         X, y = np.array(data['y']).reshape(len(data['y']),-1), np.array(data['y_true']).ravel()
    #         model = LogisticRegression(solver='liblinear').fit(X, y)
    #
    #         # Test the model on BSNIP
    #         testing_data = get_pickle_obj(os.path.join(root, path, test_file.format(b=j+1, f=fold)))
    #         X_test, y_test = np.array(testing_data['y']).reshape(len(testing_data['y']),-1), \
    #                          np.array(testing_data['y_true']).ravel()
    #         y_pred = model.predict_proba(X_test)[:,1]
    #
    #         metrics['auc'][path][b].append(roc_auc_score(y_test, y_pred))
    #         metrics['balanced_accuracy'][path][b].append(balanced_accuracy_score(y_test, y_pred>0.5))
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



