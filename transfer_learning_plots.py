import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pynet.utils import get_pickle_obj
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style("darkgrid")

## Analysis of the generalization power of DenseNet pretrained on Age problem for dx classification

blocks = ["block1", "block2", "block3", "block4"]
root = "/neurospin/psy_sbox/bd261576/checkpoints"

paths = ["regression_age_sex/Benchmark_IXI_HCP/DenseNet/Age/block_outputs",
                "regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/block_outputs",
                "scz_prediction/Benchmark_Transfer/Age_Pretraining/schizconnect_vip/DenseNet/block_outputs"]
training_filenames = ["DenseNet_Block{b}_SCHIZCONNECT-VIP_Age_Pretrained_{f}_epoch49.pkl",
                      "DenseNet_Block{b}_Dx_SCZ_VIP_{f}_epoch49.pkl",
                      "DenseNet_Block{b}_Dx_SCZ_VIP_{f}_epoch49.pkl"]
testing_filenames = ["DenseNet_Block{b}_BSNIP_Age_Pretrained_{f}_epoch49.pkl",
                     "DenseNet_Block{b}_Dx_BSNIP_{f}_epoch49.pkl",
                     "DenseNet_Block{b}_Dx_BSNIP_{f}_epoch49.pkl"]

exp_names = ["Trained on Age", "Trained on Dx", "Fine-Tuned on Dx (Age Transfer)"]
nb_folds = [1, 5, 5]

metrics = {'auc': {exp: {b: [] for b in blocks} for exp in exp_names},
           'balanced_accuracy': {exp: {b: [] for b in blocks} for exp in exp_names}}

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10,5))
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

            metrics['auc'][exp][b].append(roc_auc_score(y_test, y_pred))
            metrics['balanced_accuracy'][exp][b].append(balanced_accuracy_score(y_test, y_pred>0.5))
    seaborn.lineplot(x=[b for b in range(len(blocks)) for _ in range(nb_folds[i])],
                     y=[metrics['auc'][exp][b][k] for b in blocks for k in range(nb_folds[i])],
                     label=exp, ax=axes[0], marker='o')
    seaborn.lineplot(x=[b for b in range(len(blocks)) for _ in range(nb_folds[i])],
                     y=[metrics['balanced_accuracy'][exp][b][k] for b in blocks for k in range(nb_folds[i])],
                     label=exp, ax=axes[1], marker='o')

axes[0].set_xlabel('Block')
axes[0].set_ylabel('AUC')
axes[0].set_xticks(range(len(blocks)))
axes[0].set_xticklabels(range(1, len(blocks)+1))
axes[1].tick_params(labelleft=True)
axes[1].set_xticks(range(len(blocks)))
axes[1].set_xticklabels(range(1, len(blocks)+1))
axes[1].set_xlabel('Block')
axes[1].set_ylabel('Balanced Accuracy')
plt.savefig('densenet_hidden_representations.png')
