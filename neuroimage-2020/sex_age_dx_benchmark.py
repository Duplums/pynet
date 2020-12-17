import numpy as np
from pynet.history import History
from pynet.plotting.image import plot_losses, linear_reg_plots
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from json_config import CONFIG
from pynet.utils import *
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import *
from pynet.metrics import ECE_score, get_binary_classification_metrics, get_regression_metrics
from pynet.models.densenet import *
from pynet.cca import CCAHook, svcca_distance
from pynet.datasets.core import DataManager
from pynet.transforms import *
from tqdm import tqdm
import pickle
import seaborn
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.special import expit

seaborn.set_style("darkgrid")

## Plots the metrics during the optimization
root = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP'
nets = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNeXt', 'DenseNet', 'ColeNet', 'VGG11', 'TinyDenseNet_Exp9', 'SFCN']
net_names = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNeXt', 'DenseNet', 'tiny-VGG', 'VGG11', 'tiny-DenseNet', 'SFCN']
path_nets = ['ResNet/ResNet18', 'ResNet/ResNet34', 'ResNet/ResNet50', 'ResNeXt', 'DenseNet', 'ColeNet', 'VGG/VGG11', 'TinyDenseNet', 'SFCN']
problem = "Age"

files = ['Train_{net}_{pb}_{db}_%s_epoch_{e}.pkl'.format(net=n, pb=problem, db='HCP_IXI', e=299) for n in nets]
val_files = ['Validation_{net}_{pb}_{db}_%s_epoch_{e}.pkl'.format(net=n, pb=problem, db='HCP_IXI', e=299) for n in nets]
test_files = ['Test_{net}_{pb}_{db}_fold%s_epoch{e}.pkl'.format(net=n, pb=problem, db='HCP_IXI', e=299) for n in nets]
h = [History.load(os.path.join(root, net, 'N_500', problem, file),folds=range(5)) for (net, file) in zip(path_nets, files)]
h_val = [History.load(os.path.join(root, net, 'N_500', problem, file),folds=range(5)) for (net, file) in zip(path_nets, val_files)]
tests = [get_pickle_obj(os.path.join(root, net, 'N_500', problem, file)%0) for (net, file) in zip(path_nets, test_files)]
metrics = None#['loss_prop']

plot_losses(h, h_val,
            patterns_to_del=['validation_', ' on validation set'],
            metrics=metrics,
            experiment_names=net_names,
            #titles={'loss': 'Age prediction'},
            ylabels={'loss': 'MAE'},
            ylim={'loss': [0, 20]},
            figsize=(15,15),
            same_plot=True,
            saving_path="age_N_500_cnn_convergence.png",
            )

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i, net in enumerate(nets):
    linear_reg_plots(np.array(tests[i]['y_pred']).reshape(-1, 1), np.array(tests[i]['y_true']).reshape(-1,1),
                     axes=axes[i%3, i//3], title=net_names[i])
plt.tight_layout()
plt.savefig('linear_reg_age_benchmark.png')

## Visualization of random MRI pictures with both CAT12 and QUASI-RAW preproc
from nibabel import Nifti1Image
from nilearn.plotting import plot_anat
import pandas as pd

data_quasi_raw = np.load(CONFIG['quasi_raw']['input_path'], mmap_mode='r')
df_quasi_raw = pd.read_csv(CONFIG['quasi_raw']['metadata_path'], sep='\t')
data_cat12 = np.load(CONFIG['cat12']['input_path'], mmap_mode='r')
df_cat12 = pd.read_csv(CONFIG['cat12']['metadata_path'], sep='\t')
img_quasi_raw = data_quasi_raw[0,0]
cat12_index = np.where(df_cat12.participant_id.eq(str(df_quasi_raw.participant_id[0])))[0][0]
img_cat12 = data_cat12[cat12_index,0] # get the same picture
img_names = ['Quasi-Raw', 'VBM']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (img, name) in enumerate(zip([img_quasi_raw, img_cat12], img_names)):
    current_image = Nifti1Image((img-img.mean())/img.std(), np.eye(4))
    for j, direction in enumerate(['x', 'y', 'z']):
        plot_anat(current_image, cut_coords=[50], display_mode=direction, axes=axes[i][j], annotate=True,
                  draw_cross=False, black_bg='auto', vmax=3 if i==0 else 5, vmin=0)
        if j == 1:
            axes[i][j].set_title(name, fontweight='bold')
axes[-1, -1].axis('off')
plt.subplots_adjust(wspace=0)
plt.savefig('cat12_quasi_raw_examples.png')


## Plots the convergence curves for all the networks: nb of iter steps until convergence function of
## samples size
root = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP'
nets = ['ResNet34', 'DenseNet', 'ColeNet', 'TinyDenseNet_Exp9']
net_names = ['ResNet34', 'DenseNet', 'tiny-VGG', 'tiny-DenseNet']
path_nets = ['ResNet/ResNet34', 'DenseNet', 'ColeNet', 'TinyDenseNet']
pbs = ["Age", "Sex", "Dx"]
dbs = ["HCP_IXI", "HCP_IXI", "SCZ_VIP"]
metrics = ['validation_loss', 'balanced_accuracy', 'balanced_accuracy'] # for CAT12 ['validation_loss', 'loss', 'validation_loss' ]
# for QUASI-RAW ['validation_loss', 'balanced_accuracy', 'balanced_accuracy']
modes = ['Validation', 'Validation', 'Validation']

preproc = "quasi_raw"

nb_folds = [5, 5, 5]
epochs = [299, 299, 299]
sliding_window_size = 20
thresholds = [0.65, 0.04, 0.05] # for CAT12 [0.3, 0.06, 0.03] for QUASI-RAW [0.6, 0.04, 0.05]
N = {"Age": [100, 300, 500, 1000, 1600], "Sex": [100, 300, 500, 1000, 1600], "Dx": [100, 300, 500]}

def get_stability_errors(loss, type='consecutive'):
    import pandas as pd
    if type == 'consecutive':
        return np.convolve(np.abs(loss[1:]-loss[:-1]),
                           1./sliding_window_size*np.ones(sliding_window_size, dtype=int), 'valid')
    if type == "std":
        s = pd.Series(loss).rolling(window=sliding_window_size).std().values[sliding_window_size:]
        return s


def get_stable_step(errors, threshold, offset=0):
    step = len(errors) - 1
    for i, err in enumerate(errors):
        if err <= threshold and np.all(errors[i:]<=threshold):
            return i + offset
    return step + offset

conv_fig, conv_axes = plt.subplots(1, len(pbs), figsize=(len(pbs)*5, 5))

for i, (pb, db, nb_f, epoch, metric, threshold, mode) in enumerate(zip(pbs, dbs, nb_folds, epochs, metrics, thresholds, modes)):
    hyperparams = len(N[pb])*['']
    # if preproc == 'quasi_raw' and pb == "Age":
    #     hyperparams = ["_step_size_scheduler_10" if n < 1000 else "" for n in N[pb]]

    h_val = [[History.load(os.path.join(root, preproc, path_net, 'N_%i'%n, pb,  'Validation_{net}_{pb}_{db}{hyper}_{fold}_epoch_{epoch}.pkl'.
                                                   format(net=net, pb=pb, db=db, hyper=hyperparams[l],fold=0, epoch=epoch)))
         for (path_net, net) in zip(path_nets, nets)] for l, n in enumerate(N[pb])]

    h = [[History.load(os.path.join(root, preproc, path_net, 'N_%i'%n, pb,  'Train_{net}_{pb}_{db}{hyper}_{fold}_epoch_{epoch}.pkl'.
                                                   format(net=net, pb=pb, db=db, hyper=hyperparams[l],fold=0, epoch=epoch)))
         for (path_net, net) in zip(path_nets, nets)] for l, n in enumerate(N[pb])]


    losses = [[[np.array(History.load(os.path.join(root, preproc, path_net, 'N_%i'%n, pb,
                                                   '{mode}_{net}_{pb}_{db}{hyper}_{fold}_epoch_{epoch}.pkl'.
                                                   format(mode=mode, net=net, pb=pb, db=db, hyper=hyperparams[l],fold=f, epoch=epoch))).
                             to_dict(patterns_to_del=' on validation set')[metric][-1]) for f in range(nb_f)]
                          for l,n in enumerate(N[pb])]
                         for (path_net, net) in zip(path_nets, nets)]

    sum_diff_errors = [[[get_stability_errors(val, 'std') for val in h_val_per_n]
                         for h_val_per_n in h_val]
                        for h_val in losses]

    nb_epochs_after_conv = [[[get_stable_step(errors, threshold, offset=sliding_window_size)
                              for errors in sum_diff_errors_per_n]
                            for sum_diff_errors_per_n in sum_diff_errors_per_net]
                            for sum_diff_errors_per_net in sum_diff_errors]

    for l, net in enumerate(net_names):
        seaborn.lineplot(x=[n for n in N[pb] for _ in range(nb_f)],
                         y=[e*n for epochs,n in zip(nb_epochs_after_conv[l], N[pb]) for e in epochs],
                         marker='o', label=net, ax=conv_axes[i])
        conv_axes[i].legend()
        conv_axes[i].set_xlabel('Number of training samples')
        conv_axes[i].set_title('%s Prediction'%pb.upper(), fontweight='bold')
        conv_axes[i].set_xticks(N[pb])
        conv_axes[i].set_xticklabels(N[pb])
        conv_axes[i].set_ylabel('# iterations until convergence')
    if pb == "Age":
        for k, n in enumerate(N[pb]):
            fig, axes = plot_losses(h[k], h_val[k],
                                    patterns_to_del=['validation_', ' on validation set'],
                                    metrics=None,
                                    experiment_names=[name+ ' N=%i'%n for name in net_names],
                                    figsize=(15, 15), same_plot=True)
            for l, net in enumerate(nets):
                axes[l%len(axes), l//len(axes)].axvline(nb_epochs_after_conv[l][k][0], color='red', linestyle='--')
conv_fig.tight_layout()
conv_fig.savefig('%s_convergence_speed_networks.png'%preproc)


## Robustness plots

fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
for k, pb in enumerate(['Age', 'Sex', 'Dx']):
    robustness_data = [get_pickle_obj(
        os.path.join(root, net, pb, 'Robustness_{net}_{pb}_{db}.pkl'.format(net=n, pb=pb,
                                                                            db=('SCZ_VIP' if pb=='Dx' else 'HCP_IXI'))))
        for net, n in zip(path_nets, nets)]
    for i, net in enumerate(net_names):
        std_noises = [std for std in robustness_data[i].keys() for _ in robustness_data[i][std]]
        if pb == 'Age':
            #score = [np.mean(np.abs(np.array(Y[0])-np.array(Y[1]))) for std in robustness_data[i]
            #         for Y in robustness_data[i][std]]
            score = [LinearRegression().fit(np.array(Y[1]).reshape(-1, 1), np.array(Y[0]).reshape(-1, 1)).
                         score(np.array(Y[1]).reshape(-1, 1), np.array(Y[0]).reshape(-1, 1))
                     for std in robustness_data[i] for Y in robustness_data[i][std]]
        elif pb in ['Sex', 'Dx']:
            score = [roc_auc_score(Y[1], np.array(Y[0])) for std in robustness_data[i] for Y in robustness_data[i][std]]
        seaborn.lineplot(x=std_noises, y=score, marker='x', label=net, ax=axes[0,k])
    if pb in ['Sex', 'Dx']:
        axes[0,k].set_ylim([0.4, 1])
    axes[0,k].set_xlabel('$\sigma$')
    axes[0,k].set_ylabel('$R^2$' if pb == 'Age' else 'AUC')
    axes[0,k].set_title('Robustness of various networks\n on {pb} Prediction problem'.format(pb=pb))
plt.savefig('robustness_curves_auc.png')
plt.show()

## Losses plots of the benchmark

problem = "Sex"

files = ['Train_{net}_{pb}_{db}_%s_epoch_{e}.pkl'.format(net=n, pb=problem, db='HCP_IXI', e=299) for n in nets]
val_files = ['Validation_{net}_{pb}_{db}_%s_epoch_{e}.pkl'.format(net=n, pb=problem, db='HCP_IXI', e=299) for n in nets]
test_files = ['Test_{net}_{pb}_{db}_fold{fold}_epoch{e}.pkl'.format(net=n, pb=problem, db='HCP_IXI', fold=0, e=299) for n in nets]
h = [History.load(os.path.join(root, net, 'N_500', problem, file),folds=range(5)) for (net, file) in zip(path_nets, files)]
h_val = [History.load(os.path.join(root, net, 'N_500', problem, file),folds=range(5)) for (net, file) in zip(path_nets, val_files)]
tests = [get_pickle_obj(os.path.join(root, net, 'N_500', problem, file)) for (net, file) in zip(path_nets, test_files)]
metrics = ['roc_auc', 'balanced_accuracy']


plot_losses(h, h_val,
            patterns_to_del=['validation_', ' on validation set'],
            metrics=metrics,
            experiment_names=net_names,
            #titles={'roc_auc': 'Gender prediction', 'balanced_accuracy': 'Gender Prediction'},
            ylabels={'roc_auc': 'AUC', 'balanced_accuracy': 'Balanced Accuracy'},
            ylim={'roc_auc': [0, 1], 'balanced_accuracy': [0, 1]},
            figsize=(15,15),
            same_plot=True,
            saving_path="sex_N_500_cnn_convergence.png")

problem = "Dx"
special_nets = ['ResNet34', 'DenseNet', 'ColeNet', 'TinyDenseNet_Exp9']
files = ['Train_{net}_{pb}_{db}_%s_epoch_{e}.pkl'.format(net=n, pb=problem, db='SCZ_VIP', e=99
if n not in special_nets else 100) for n in nets]
val_files = ['Validation_{net}_{pb}_{db}_%s_epoch_{e}.pkl'.format(net=n, pb=problem, db='SCZ_VIP', e=99
if n not in special_nets else 100) for n in nets]
test_files = ['Test_{net}_{pb}_{db}_fold%s_epoch{e}.pkl'.format(net=n, pb=problem, db='SCZ_VIP', e=99
if n not in special_nets else 100) for n in nets]
h = [History.load(os.path.join(root, net, 'N_500', problem, file), folds=range(5)) for (net, file) in zip(path_nets, files)]
h_val = [History.load(os.path.join(root, net, 'N_500', problem, file), folds=range(5)) for (net, file) in zip(path_nets, val_files)]
metrics = ['roc_auc', 'balanced_accuracy']

plot_losses(h, h_val,
            patterns_to_del=['validation_', ' on validation set'],
            metrics=metrics,
            experiment_names=net_names,
            #titles={'roc_auc': 'Gender prediction', 'balanced_accuracy': 'Gender Prediction'},
            ylabels={'roc_auc': 'AUC', 'balanced_accuracy': 'Balanced Accuracy'},
            ylim={'roc_auc': [0, 1], 'balanced_accuracy': [0, 1]},
            figsize=(15,15),
            same_plot=True,
            saving_path="dx_N_500_cnn_convergence.png")

# delta_age as predictor of the clinical status
from scipy.stats import ks_2samp

test_densenet = [get_pickle_obj(os.path.join(root, 'DenseNet', 'Age', 'Test_DenseNet_Age_HCP_IXI_fold0_epoch99.pkl')),
                  get_pickle_obj(os.path.join(root, 'DenseNet', 'Age', 'Test_DenseNet_Age_BSNIP_SCZ_fold0_epoch99.pkl'))]
mask = [np.array(test_densenet[i]['y_true'])  < 30 for i in range(2)]
absolute_error_min_age = [np.abs(np.array(test_densenet[i]['y_pred'])-np.array(test_densenet[i]['y_true']))[mask[i]] for i in range(2)]
absolute_error = [np.abs(np.array(test_densenet[i]['y_pred'])-np.array(test_densenet[i]['y_true'])) for i in range(2)]

# Significant KS-test for population with age < 30
ks_test_min_age = ks_2samp(absolute_error_min_age[0], absolute_error_min_age[1])
# ... But not after
ks_test = ks_2samp(absolute_error[0], absolute_error[1])

fig, axes = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)
seaborn.distplot(np.array(test_densenet[0]['y_pred'])[mask[0]], ax=axes[0,0], norm_hist=True, label='Predicted Age')
seaborn.distplot(np.array(test_densenet[0]['y_true'])[mask[0]], ax=axes[0,0], norm_hist=True, label='True Age')
seaborn.distplot(np.array(test_densenet[1]['y_pred'])[mask[1]], ax=axes[0,1], norm_hist=True, label='Predicted Age')
seaborn.distplot(np.array(test_densenet[1]['y_true'])[mask[1]], ax=axes[0,1], norm_hist=True, label='True Age')
seaborn.distplot(np.array(test_densenet[1]['y_pred']), ax=axes[1,0], norm_hist=True, label='Predicted Age')
seaborn.distplot(np.array(test_densenet[1]['y_true']), ax=axes[1,0], norm_hist=True, label='True Age')
seaborn.distplot(np.array(test_densenet[1]['y_pred']), ax=axes[1,1], norm_hist=True, label='Predicted Age')
seaborn.distplot(np.array(test_densenet[1]['y_true']), ax=axes[1,1], norm_hist=True, label='True Age')
axes[0,0].set_title('Age Prediction on BSNIP for HC \nwith Age<30 (N=%i)'%mask[0].sum())
axes[0,1].set_title('Age Prediction on BSNIP for SCZ \nwith Age<30 (N=%i)'%mask[1].sum())
axes[1,0].set_title('Age Prediction on BSNIP for HC (N=200)')
axes[1,1].set_title('Age Prediction on BSNIP for SCZ (N=194)')
axes[0,0].legend()
axes[0,1].legend()
axes[1,0].legend()
axes[1,1].legend()
plt.savefig('delta_age_hist_analysis.png')

fig, axes = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
axes[0,0].boxplot(absolute_error_min_age, notch=True, labels=['HC (N=%i)'%mask[0].sum(),
                                                              'SCZ (N=%i)'%mask[1].sum()])
axes[0,0].text(1, 22, 'KS Statistic=%1.2e\np-value=%1.2e'%
               (ks_test_min_age.statistic, ks_test_min_age.pvalue),
               bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
axes[0,0].set_title('Absolute Error for Age Prediction on BSNIP\n with Age<30 (N=%i)'%(mask[0].sum()+mask[1].sum()))
axes[0,1].boxplot(absolute_error, notch=True, labels=['HC (N=200)', 'SCZ (N=194)'])
axes[0,1].text(1, 22, 'KS Statistic=%1.2e\np-value=%1.2e'%
               (ks_test.statistic, ks_test.pvalue),
               bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
axes[0,1].set_title('Absolute Error for Age Prediction on BSNIP (N=394)')
plt.savefig('delta_age_err_analysis.png')


### Learning curves

root = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP'
net_names = ['ResNet34', 'DenseNet', 'tiny-VGG']#, 'tiny-DenseNet', 'Linear Model']
nets = ['ResNet34', 'DenseNet', 'ColeNet']#, 'TinyDenseNet_Exp9', "LinearModel"]
path_nets = ['ResNet/ResNet34', 'DenseNet', 'ColeNet']#, 'TinyDenseNet', 'LinearModel']
preprocessings = ['']

all_metrics = {preproc: {pb: dict() for pb in ['Age', 'Sex', 'Dx']} for preproc in preprocessings}
nb_training_samples = [[100, 300, 500, 1000, 1600, 10000],[100, 300, 500, 1000, 1600, 10000], [100, 300, 500]]
nb_epochs = [299]

X = [[n for n in training for k in range(3+2*(n<10000)+5*(n<500))] for i,training in enumerate(nb_training_samples)]

all_results = {preproc: {pb: {net if net!="LinearModel" else ('Ridge' if pb=='Age' else 'LogisticRegression'):
                                  [[[0 for k in range(3+2*(n<10000)+5*(n<500))]
                                     for n in nb_training_samples[n_pb]]
                                    for e in nb_epochs]
                              for net in nets}
                         for n_pb, pb in enumerate(['Age', 'Sex', 'Dx'])}
               for preproc in preprocessings}

for preproc in preprocessings:
    fig, axes = plt.subplots(len(nb_epochs), 3, sharey='col', sharex='col', squeeze=False, figsize=(15, 5*len(nb_epochs)))

    for n_pb, pb in enumerate(["Age", "Sex", "Dx"]):
        db = "HCP_IXI" if pb != "Dx" else "SCZ_VIP"
        for (name, net, path_net) in zip(net_names, nets, path_nets):
            if net == 'LinearModel':
                net = "Ridge" if pb == "Age" else "LogisticRegression"
            for i, e in enumerate(nb_epochs):
                if name == "Linear Model":
                    e = 100
                for j, n in enumerate(nb_training_samples[n_pb]):
                    for k in range(3+2*(n<10000)+5*(n<500)):
                        hyperparams = "_step_size_scheduler_10_gamma_0.7" \
                            if (net == "TinyDenseNet_Exp9" and pb == "Age" and n > 100 and n<1000) else "_step_size_scheduler_10"
                        try:
                            path = os.path.join(root, preproc, path_net, 'N_{n}', pb,
                                                'Test_{net}_{pb}_{db}{hyper}_fold{k}_epoch{e}.pkl')
                            all_results[preproc][pb][net][i][j][k] = get_pickle_obj(
                                path.format(net=net, pb=pb, db=db if n!=10000 else "Big_Healthy",
                                            hyper=hyperparams, k=k, n=n if n<10000 else '10K', e=e))
                        except FileNotFoundError:
                            path = os.path.join(root, preproc, path_net, 'N_{n}', pb,
                                                'Test_{net}_{pb}_{db}_fold{k}_epoch{e}.pkl')
                            all_results[preproc][pb][net][i][j][k] = get_pickle_obj(
                                path.format(net=net, pb=pb, db=db if n!=10000 else "Big_Healthy",
                                            k=k, n=n if n<10000 else '10K', e=e))

            if pb == 'Age': # Compute MAE
                all_metrics[preproc][pb][net] = [[np.mean(np.abs(np.array(all_results[preproc][pb][net][e][i][k]['y_true'])-
                                                                 np.array(all_results[preproc][pb][net][e][i][k]['y_pred'])))
                                                  for i,n in enumerate(nb_training_samples[n_pb])
                                                  for k in range(3+2*(n<10000)+5*(n<500)) ]
                                                 for e in range(len(nb_epochs))]
            if pb == 'Sex' or pb == "Dx": # Compute AUC
                all_metrics[preproc][pb][net] = [[roc_auc_score(all_results[preproc][pb][net][e][i][k]['y_true'],
                                                                all_results[preproc][pb][net][e][i][k]['y_pred'])
                                                  for i, n in enumerate(nb_training_samples[n_pb])
                                                  for k in range(3+2*(n<10000)+5*(n<500)) ]
                                                 for e in range(len(nb_epochs))]

            for k, epoch in enumerate(nb_epochs):
                seaborn.lineplot(x=X[n_pb], y=all_metrics[preproc][pb][net][k], marker='o', label=name, ax=axes[k, n_pb])
                if pb != "Dx":
                    axes[k, n_pb].set_x

    axes[0,0].set_ylim(bottom=3)
    axes[0,1].set_ylim(top=1)
    # axes[1,1].tick_params(labelleft=True)
    # axes[1,0].tick_params(labelleft=True)
    # axes[2,1].tick_params(labelleft=True)
    # axes[2,0].tick_params(labelleft=True)

    plt.legend()

    metrics = ['MAE', 'AUC', 'AUC']
    for i, _epoch in enumerate(nb_epochs):
        for j, _pb in enumerate(["Age", "Sex", "Dx"]):
            axes[i,j].set_ylabel(metrics[j])
            axes[i,j].set_xticks(nb_training_samples[j])
            if i == 0:
                axes[i,j].set_title("{pb} prediction\n at $N={n}$ epochs".format(pb=_pb, n=_epoch), fontweight='bold')
            else:
                axes[i,j].set_title("$N={n}$ epochs".format(n=_epoch), fontweight='bold')
            if i == len(nb_epochs)-1:
                axes[i,j].set_xlabel('Number of training samples')
    plt.tight_layout(w_pad=0.1, h_pad=0.2)
    fig.savefig('learning_curves_preproc_{}.png'.format(preproc), format='png')
    plt.show()


## Calibration Curves at N=500
root = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP'
nets = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNeXt', 'DenseNet', 'ColeNet', 'VGG11', 'TinyDenseNet_Exp9', 'SFCN']
net_names = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNeXt', 'DenseNet', 'tiny-VGG', 'VGG11', 'tiny-DenseNet', 'SFCN']
path_nets = ['ResNet/ResNet18', 'ResNet/ResNet34', 'ResNet/ResNet50', 'ResNeXt', 'DenseNet', 'ColeNet', 'VGG/VGG11', 'TinyDenseNet', 'SFCN']
N = 500
problems = ['Dx', 'Sex']
epochs = [99, 299]
dbs = ["SCZ_VIP", "HCP_IXI"]
for i, (pb, db, e) in enumerate(zip(problems, dbs, epochs)):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for j, (net, name, path) in enumerate(zip(nets, net_names, path_nets)):
        res = [get_pickle_obj(os.path.join(root, path, 'N_%i'%N, pb,
                                          "Test_{net}_{pb}_{db}_fold{fold}_epoch{e}.pkl".
                                           format(net=net,pb=pb,db=db,fold=k,
                                                  e=e+(pb == "Dx" and net in ["ResNet34", "DenseNet", "ColeNet", "TinyDenseNet_Exp9"]))))
               for k in range(5)]
        frac_pos, mean_pred_proba = calibration_curve(res[0]['y_true'], expit(res[0]['y_pred']))
        hist, bins = np.histogram(expit(res[0]['y_pred']), bins=5)
        axes[j%3, j//3].bar(bins[:-1], hist/len(res[0]['y_pred']), np.diff(bins), ls='--', fill=False, edgecolor='blue', align='edge')
        axes[j%3, j//3].plot(mean_pred_proba, frac_pos, 's-', color='red')
        axes[j%3, j//3].set_ylabel('Accuracy', color='red')
        axes[j%3, j//3].tick_params(axis='y', colors='red')
        sec_ax = axes[j%3,j//3].secondary_yaxis('right')
        sec_ax.tick_params(axis='y', colors='blue')
        sec_ax.set_ylabel('Fraction of Samples', color='blue')
        axes[j%3, j//3].set_xlabel('Confidence')
        axes[j%3, j//3].plot([0,1], [0,1], 'k:')
        axes[j%3, j//3].set_title(name, fontweight='bold')
    fig.tight_layout(pad=3.0)
    fig.savefig('%s_calibration_plots.png'%pb)

plt.show()

## Performance + Calibration of Ensemble models (DenseNet and tiny-DenseNet) at N=500
root = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP'
nets = ['DenseNet', 'TinyDenseNet_Exp9'] # TODO: add tiny-DenseNet
net_names = ['DenseNet (High capacity)', 'tiny-DenseNet (Low capacity)']
path_nets = ['DenseNet', 'TinyDenseNet']
T = list(range(1, 11))
N = 500
problems = ['Dx', 'Sex']
epochs = [99, 299]
dbs = ["SCZ_VIP", "HCP_IXI"]
colors = ['green', 'red']

# Calibration's improvement in terms of ECE
fig_ece, axes_ece = plt.subplots(1, len(problems), figsize=(len(problems)*5, 5), sharex=True, squeeze=False)
for j, (pb, db, e) in enumerate(zip(problems, dbs, epochs)):
    ax = axes_ece[0, j]
    for i, (net, name, path) in enumerate(zip(nets, net_names, path_nets)):
        res = [get_pickle_obj(os.path.join(root, path, 'N_%i'%N, pb,'Ensembling',
                                          "EnsembleTest_{net}_{pb}_{db}_fold{fold}_epoch{e}.pkl".
                                           format(net=net,pb=pb,db=db,fold=k, e=e))) for k in range(5)]
        y_pred, y_true = np.array([res[f]['y'] for f in range(5)]), np.array([res[f]['y_true'] for f in range(5)])[:, :,0]
        y_pred = expit(y_pred)
        AUC = [[roc_auc_score(y_true[k], y_pred[k,:,:t].mean(axis=1)) for k in range(5)] for t in T]
        ECE = [[ECE_score(y_pred[k,:,:t].mean(axis=1), y_true[k]) for k in range(5)] for t in T]
        ax.errorbar(T, [np.mean(ece) for ece in ECE], yerr=[np.std(ece) for ece in ECE], capsize=3, ecolor=colors[i],
                    color=colors[i], label=name)
    ax.set_ylabel('ECE $\downarrow$', color='black')
    ax.tick_params(axis='y', colors='black')
    ax.set_ylim([0, 0.25])
    ax.set_title("{pb} Prediction".format(pb=pb), fontsize=16, fontweight='bold')
    ax.set_xticks(ticks=T)
    ax.set_xlabel('Number of models T')
    ax.legend()

fig_ece.tight_layout(pad=2)
fig_ece.savefig('ensemble_calibration_performance.png')

# TODO: Calibration curves for DenseNet/tiny-DenseNet
# fig_cal_curves, big_axes = plt.subplots(2*len(problems), 1, figsize=(2 * 5, len(problems) * 5),
#                                         sharey=True, squeeze=False, gridspec_kw={})
# for row, (big_ax, pb_name) in enumerate(zip(big_axes[:, 0], problems), start=1):
#     big_ax.set_title('{pb} Prediction'.format(pb=pb_name), fontweight='bold', fontsize=16)
#     big_ax.axis('off')
#     big_ax._frameon = False
#     big_ax.title.set_position([.5, 1.08])
# for j, (pb, db, e) in enumerate(zip(problems, dbs, epochs)):
#     for l, t in enumerate([T[0], T[-1]], start=1):
#         calibration_curves_axis.append(fig.add_subplot(len(problems), 2, 2 * j + 1 + l))
#     for i, (net, name, path) in enumerate(zip(nets, net_names, path_nets)):
#         res = [get_pickle_obj(os.path.join(root, path, 'N_%i'%N, pb,'Ensembling',
#                                           "EnsembleTest_{net}_{pb}_{db}_fold{fold}_epoch{e}.pkl".
#                                            format(net=net,pb=pb,db=db,fold=k, e=e))) for k in range(5)]
#         y_pred, y_true = np.array([res[f]['y'] for f in range(5)]), np.array([res[f]['y_true'] for f in range(5)])[:, :,0]
#         y_pred = expit(y_pred)
#         AUC = [[roc_auc_score(y_true[k], y_pred[k,:,:t].mean(axis=1)) for k in range(5)] for t in T]
#         ECE = [[ECE_score(y_pred[k,:,:t].mean(axis=1), y_true[k]) for k in range(5)] for t in T]
#         ax.errorbar(T, [np.mean(ece) for ece in ECE], yerr=[np.std(ece) for ece in ECE], capsize=3, ecolor=colors[i],
#                     color=colors[i], label=name)
#         # ax2 = ax.twinx()
#         # ax2.errorbar(T, [np.mean(auc) for auc in AUC], yerr=[np.std(auc) for auc in AUC], capsize=3, ecolor='blue', color='blue')
#         # ax2.set_ylabel('AUC', color='blue')
#         # ax2.tick_params(axis='y', colors='blue')
#         # ax2.set_ylim([0.5,0.95])
#
#         for l, t in enumerate([T[0], T[-1]], start=0):
#             frac_pos_and_mean_pred_proba = [calibration_curve(y_true[fold], y_pred[fold,:,:t].mean(axis=1))
#                                          for fold in range(5)]
#             hist, bins = np.histogram(y_pred[0,:,:t].mean(axis=1), bins=5) # we assume they are all the same across the folds...
#             calibration_curves_axis[l].bar(bins[:-1], hist/len(y_true[0]), np.diff(bins), ls='--',
#                    fill=False, edgecolor=colors[i], align='edge')
#             seaborn.lineplot(x=[mean_pred_prob for _ in range(5) for mean_pred_prob in
#                                 np.mean([frac_mean_k[1] for frac_mean_k in frac_pos_and_mean_pred_proba], axis=0)],
#                              y=[m for frac_mean_k in frac_pos_and_mean_pred_proba for m in frac_mean_k[0]],
#                              marker='o', ax=calibration_curves_axis[l], color=colors[i], label=name)
#             #ax.plot(mean_pred_proba, frac_pos, 's-', color='red')
#             calibration_curves_axis[l].set_ylabel('Fraction of samples / Accuracy', color='black')
#             calibration_curves_axis[l].tick_params(axis='y', colors='black')
#             #sec_ax = calibration_curves_axis[l].secondary_yaxis('right')
#             #sec_ax.tick_params(axis='y', colors='black')
#             #sec_ax.set_ylabel('Fraction of Samples', color='black')
#             calibration_curves_axis[l].set_xlabel('Confidence')
#             calibration_curves_axis[l].plot([0,1], [0,1], 'k:')
#             calibration_curves_axis[l].set_title('Calibration curve at T=%i'%t, fontsize=13)
#             calibration_curves_axis[l].legend()
#     ax.legend()
#
# fig.tight_layout(pad=2)
# fig.savefig('ensemble_calibration_plots.png')

# Predictive uncertainty quality improvement with Deep Ensemble for both low and high capacity models
entropy_func = lambda sigma: - ((1-sigma) * np.log(1-sigma+1e-8) + sigma * np.log(sigma+1e-8))
colors = ['blue', 'green']
markers = ['o', '+']
T_models = [1, 10]
data_retained = np.arange(0.1, 1.01, 0.1)
fig, big_axes = plt.subplots(len(problems), 1, figsize=(7*len(nets), 7*len(problems)), sharex=True,
                             squeeze=False)
for row, (big_ax, pb_name) in enumerate(zip(big_axes[:,0], problems), start=1):
    big_ax.set_title('{pb} Prediction'.format(pb=pb_name), fontweight='bold', fontsize=16)
    big_ax.axis('off')
    big_ax._frameon=False
    big_ax.title.set_position([.5, 1.08])
for k, (pb, db, e ) in enumerate(zip(problems, dbs, epochs)):
    for i, (name, net, path) in enumerate(zip(net_names, nets, path_nets)):
        ax = fig.add_subplot(len(problems), len(nets), len(nets)*k+i+1)
        res = [get_pickle_obj(os.path.join(root, path, 'N_%i' % 500, pb, 'Ensembling',
                                           "EnsembleTest_{net}_{pb}_{db}_fold{fold}_epoch{e}.pkl".
                                           format(net=net, pb=pb, db=db, fold=k, e=e))) for k in range(5)]
        y_pred_ensemble, y_true = np.array([res[f]['y'] for f in range(5)]), np.array([res[f]['y_true'] for f in range(5)])[:, :, 0]

        for it_t, t in enumerate(T_models):
            y_pred = expit(y_pred_ensemble[:,:,:t]).mean(axis=2) # take the mean prob of Ensemble

            # Get the uncertainty (entropy) for correct/wrong predictions
            H_pred = entropy_func(y_pred)
            #MI = H_pred - entropy_func(expit(y_pred)).mean(axis=2)
            mask_corr = [(pred>0.5)==true for (pred, true) in zip(y_pred, y_true)]

            # Plot the performance (AUC, bAcc) as a function of the data retained based on the entropy
            H_pred_sorted = np.sort([H for H in H_pred])
            threshold = [[H[int(th*(len(y_pred[m])-1))] for th in data_retained] for m, H in enumerate(H_pred_sorted)]

            # Threshold based on the entropy directly
            #threshold = [data_retained for _ in range(5)]


            y_pred_thresholded = [pred[H<=th] for m, (pred, H) in enumerate(zip(y_pred, H_pred)) for th in threshold[m]]
            y_true_thresholded = [true[H<=th] for m, (true, H) in enumerate(zip(y_true, H_pred)) for th in threshold[m]]

            auc = [roc_auc_score(true, pred) for (pred, true) in zip(y_pred_thresholded, y_true_thresholded)]
            seaborn.lineplot(x=[th*100 for _ in y_pred for th in data_retained],
                             y=auc, marker=markers[it_t], label=(t>1)*'Ensemble '+'{net} (T={t})'.format(net=name,t=t),
                             ax=ax, color=colors[i])
            if it_t == 0:
                auc_random = [roc_auc_score(true, pred) for (pred, true) in zip(y_pred, y_true) for th in data_retained]
                seaborn.lineplot(x=[th * 100 for _ in y_pred for th in data_retained],
                                 y=auc_random, marker='.',
                                 label='Random case',
                                 ax=ax, color='black')

        ax.set_ylabel('AUC')
        ax.set_xlabel('% Data Retained based on $\mathcal{H}$')
        if pb == "Dx":
            ax.set_ylim([0.76, 0.86])
        if pb == "Sex":
            ax.set_ylim([0.80, 0.95])
        ax.legend()
        ax.set_title(name, fontweight='bold', fontsize=14)
fig.tight_layout(pad=2)
fig.savefig('models_uncertainty_estimation_ensemble.png', format='png')



# Performance's improvement
from scipy.stats import pearsonr

problems = ["Age", "Sex", "Dx"]
epochs = [299, 299, 99]
dbs = ["HCP_IXI", "HCP_IXI", "SCZ_VIP"]
metrics = ["MAE", "AUC", "AUC"]
fig, big_axes = plt.subplots(len(nets), 1, figsize=(len(problems) * 5, len(nets) * 5), sharex=True, squeeze=False)
for row, (big_ax, name) in enumerate(zip(big_axes[:, 0], net_names), start=1):
    big_ax.set_title('{net}'.format(net=name), fontweight='bold', fontsize=16)
    big_ax.axis('off')
    big_ax._frameon = False
    big_ax.title.set_position([.5, 1.08])
for i, (net, name, path) in enumerate(zip(nets, net_names, path_nets)):
    for j, (pb, db, e, metric) in enumerate(zip(problems, dbs, epochs, metrics)):
        res = [get_pickle_obj(os.path.join(root, path, 'N_%i' % N, pb, 'Ensembling',
                                           "EnsembleTest_{net}_{pb}_{db}_fold{fold}_epoch{e}.pkl".
                                           format(net=net, pb=pb, db=db, fold=k, e=e))) for k in range(5)]
        y_pred, y_true = np.array([res[f]['y'] for f in range(5)]), np.array([res[f]['y_true'] for f in range(5)])[:, :, 0]
        if pb == "Age":
            score = [[np.mean(np.abs(y_true[k]-y_pred[k, :, :t].mean(axis=1))) for k in range(5)] for t in T]
            #score = [[pearsonr(y_pred[k, :, :t].mean(axis=1), y_true[k])[0] for k in range(5)] for t in T]

        else:
            y_pred = expit(y_pred)
            score = [[roc_auc_score(y_true[k], y_pred[k, :, :t].mean(axis=1)) for k in range(5)] for t in T]
        ax = fig.add_subplot(len(nets), len(problems), len(problems)*i+j+1)
        seaborn.lineplot(x=[t for t in T for _ in range(5)], y=[s_k for s in score for s_k in s], marker='o', ax=ax)
        # ax.errorbar(T, [np.mean(s) for s in score], yerr=[np.std(s) for s in score], capsize=3, ecolor='blue',
        #             color='blue')
        ax.set_title('{pb}'.format(pb=pb if pb != "Dx" else "SCZ vs HC"), fontweight='bold')
        ax.set_xlabel('Number of models T')
        ax.set_ylabel('{metric}'.format(metric=metric))
fig.tight_layout(pad=2)
fig.savefig('ensemble_performance_age_sex_dx.png')
plt.show()



### Computes the entropy as a measure of (epistemic+aleatoric) uncertainty for wrong predictions and correct predictions
### + True Class Probability (TCP) as a histogram for well-classified/mis-classified examples


root = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP'
nets = ['TinyDenseNet_Exp9', 'DenseNet', 'Full_ColeNet', 'Full_ResNet34', 'DenseNet', 'ColeNet', 'ResNet34', 'DenseNet', 'ColeNet', 'ResNet34']
net_names = ['tiny-DenseNet', 'DenseNet', 'tiny-VGG', 'ResNet34', 'MC-Dropout DenseNet', 'MC-Dropout tiny-VGG', 'MC-Dropout ResNet34',
             'Ensemble DenseNet', 'Ensemble tiny-VGG', 'Ensemble ResNet34']
path_nets = ['TinyDenseNet', 'DenseNet', 'ColeNet', 'ResNet/ResNet34', 'DenseNet/Dx/Dropout/Concrete_Dropout',
             'ColeNet/Dx/Dropout/Concrete_Dropout', 'ResNet/ResNet34/Dx/Dropout/Concrete_Dropout',
             'DenseNet/Dx/Ensembling', 'ColeNet/Dx/Ensembling', 'ResNet/ResNet34/Dx/Ensembling']
problem = "Dx"
epochs = [49, 49, 49, 49, 49]

entropy_func = lambda sigma: - ((1-sigma) * np.log(1-sigma+1e-8) + sigma * np.log(sigma+1e-8))
colors = ['blue', 'green', 'orange']
markers = ['o', '+', '^']

fig, axes = plt.subplots(1, 1, squeeze=False, figsize=(7, 7))
fig2, axes2 = plt.subplots(3, 3, squeeze=False, sharey='row', figsize=(15, 15))
for i, (name, net, path) in enumerate(zip(net_names, nets, path_nets)):

    if 'Concrete_Dropout' in path or 'Ensembling' in path:
        test = "MC" if "Concrete_Dropout" in path else "Ensemble"
        res = [get_pickle_obj(os.path.join(root, path,  "{t}Test_{net}_Dx_SCZ_VIP_fold{k}_epoch{e}.pkl".
                                       format(t=test,net=net,k=k,e=e))) for (k,e) in enumerate(epochs)]
        y_pred, y_true = np.array([res[f]['y'] for f in range(5)]), np.array([res[f]['y_true'] for f in range(5)])[:,:, 0]
        y_pred = expit(y_pred).mean(axis=2) # take the mean prob of the MC-sampling or Ensemble
    else:
        res = [get_pickle_obj(os.path.join(root, path, problem,  "Test_{net}_Dx_SCZ_VIP_fold{k}_epoch{e}.pkl".
                                       format(net=net,k=k,e=e))) for (k,e) in enumerate(epochs)]
        y_pred, y_true = expit(np.array([res[f]['y_pred'] for f in range(5)])), np.array([res[f]['y_true'] for f in range(5)])

    # Get the uncertainty (entropy) for correct/wrong predictions
    H_pred = entropy_func(y_pred)
    #MI = H_pred - entropy_func(expit(y_pred)).mean(axis=2)
    mask_corr = [(pred>0.5)==true for (pred, true) in zip(y_pred, y_true)]

    # Plot the performance (AUC, bAcc) as a function of the data retained based on the entropy
    data_retained = np.arange(0.5, 1.01, 0.1)
    H_pred_sorted = np.sort([H for H in H_pred])
    threshold = [[H[int(th*(len(y_pred[m])-1))] for th in data_retained] for m, H in enumerate(H_pred_sorted)]

    y_pred_thresholded = [pred[H<=th] for m, (pred, H) in enumerate(zip(y_pred, H_pred)) for th in threshold[m]]
    y_true_thresholded = [true[H<=th] for m, (true, H) in enumerate(zip(y_true, H_pred)) for th in threshold[m]]

    b_acc = [balanced_accuracy_score(true, pred>0.5) for (pred, true) in zip(y_pred_thresholded, y_true_thresholded)]
    auc = [roc_auc_score(true, pred) for (pred, true) in zip(y_pred_thresholded, y_true_thresholded)]
    TCP_err = [pred[~corr] * (pred[~corr]<=0.5) + (1-pred[~corr]) * (pred[~corr]>0.5) for (pred, corr) in zip(y_pred, mask_corr)]
    TCP_true = [pred[corr] * (pred[corr]>0.5) + (1-pred[corr]) * (pred[corr]<=0.5) for (pred, corr) in zip(y_pred, mask_corr)]
    seaborn.distplot(TCP_true[1], kde=False, label="Successes", ax=axes2[i%3,i//3], color='green')
    seaborn.distplot(TCP_err[1], kde=False, label="Errors", ax=axes2[i%3,i//3], color='red')
    axes2[i%3,i//3].set_title(format(name))
    axes2[i%3,i//3].set_ylabel('True Class Probability')
    axes2[i%3,i//3].legend()
    seaborn.lineplot(x=[th for _ in y_pred for th in data_retained],
                     y=auc, marker=markers[i//3], label=name, ax=axes[0,0], color=colors[i%3])

axes[0,0].set_ylabel('AUC')
axes[0,0].set_xlabel('Data Retained based on $\mathcal{H}$')
axes[0,0].set_ylim([0.7, 0.9])
axes[0,0].legend()
fig.savefig('models_uncertainty_curves.png', format='png')
fig2.savefig('true_class_probability_dx.png', format='png')
plt.show()


## Demonstration of the effectiveness of Concrete Dropout

h = [History.load('/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/Dropout/p_0.2/Train_DenseNet_Dx_SCZ_VIP_4_epoch_49.pkl'),
     History.load('/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/Dropout/p_0.5/Train_DenseNet_Dx_SCZ_VIP_4_epoch_49.pkl'),
     History.load('/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/Dropout/Concrete_Dropout/Train_DenseNet_Dx_SCZ_VIP_4_epoch_49.pkl'),
     History.load('/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/Train_DenseNet_Dx_SCZ_VIP_4_epoch_49.pkl')]
h_val = [History.load('/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/Dropout/p_0.2/Validation_DenseNet_Dx_SCZ_VIP_4_epoch_49.pkl'),
         History.load('/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/Dropout/p_0.5/Validation_DenseNet_Dx_SCZ_VIP_4_epoch_49.pkl'),
         History.load('/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/Dropout/Concrete_Dropout/Validation_DenseNet_Dx_SCZ_VIP_4_epoch_49.pkl'),
         History.load('/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/Validation_DenseNet_Dx_SCZ_VIP_4_epoch_49.pkl')]

plot_losses(h, h_val,
            patterns_to_del=['validation_', ' on validation set'],
            metrics=['roc_auc', 'balanced_accuracy'],
            experiment_names=['Dropout p=0.2', 'Dropout p=0.5', 'Concrete Dropout', 'Deterministic'],
            ylabels={'roc_auc': 'AUC', 'balanced_accuracy': 'Balanced Accuracy'},
            ylim={'roc_auc': [0, 1], 'balanced_accuracy': [0, 1]},
            figsize=(15,15),
            same_plot=True,
            saving_path='MCDropout_DenseNet_Dx.png')


## Feature re-using inside DenseNet: when does it occur ?
## Output: a dict {Block: {(layer_0, layer_1): SVCCA(layer_0, layer_1)}} for each block B of DenseNet and a pair of layers
#  inside B

stratif = {'train': {}, 'test': {'study': ['BSNIP'], 'diagnosis': ['control', 'schizophrenia']}}


## DenseNet121
# pretrained_path = "/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/" \
#                   "DenseNet_Dx_SCZ_VIP_4_epoch_49.pth"
# output_file = "/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/" \
#               "neurons_output_densenet121_fold4_epoch49.pkl"
# output_distances_file = "/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/DenseNet/Dx/" \
#                         "svcca_output_densenet121_fold4_epoch49.pkl"
#model = densenet121(num_classes=1, in_channels=1)
# blocks_config = [6, 12, 24, 16]

## tiny-DenseNet
pretrained_path = "/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/TinyDenseNet/Dx/" \
                  "TinyDenseNet_Exp9_Dx_SCZ_VIP_4_epoch_49.pth"
output_file = "/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/TinyDenseNet/Dx/" \
              "neurons_output_tiny_densenet_exp9_fold4_epoch49.pkl"
output_distances_file = "/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/TinyDenseNet/Dx/" \
                        "svcca_output_tiny_densenet_exp9_fold4_epoch49.pkl"
model = _densenet('exp9', 16, (6, 12, 16), 64, False, False, num_classes=1)
blocks_config = [6, 12, 16]


target_layers = [['features.denseblock{i}.denselayer{j}.conv1'.format(i=i,j=j)
                  for j in range(1,blocks_config[i-1]+1)] for i in range(1,len(blocks_config)+1)]

target_layers_flatten = [("block%i" % (i + 1), "layer%i" % (j + 1)) for i, b in enumerate(target_layers) for j, l in
                             enumerate(b)]
N = len(target_layers_flatten)

compute_outputs, compute_svcca = True, False

if compute_outputs:
    device='cuda'
    dx_mapping = LabelMapping(schizophrenia=1, control=0)
    input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128]), Normalize()]
    manager = DataManager(CONFIG['input_path'], CONFIG['metadata_path'],
                          batch_size=4,
                          number_of_folds=1,
                          labels=["diagnosis"],
                          labels_transforms=[dx_mapping],
                          custom_stratification=stratif,
                          input_transforms=input_transforms,
                          pin_memory=True,
                          drop_last=False)

    loaders = manager.get_dataloader(test=True)
    net = model.to(device)
    net.load_state_dict(torch.load(pretrained_path)['model'])
    net.eval()
    hooks = [[CCAHook(net, l, cca_distance="svcca", svd_device=device) for l in block] for block in target_layers]

    ## Computes and stores the outputs of each network for all the test set
    outputs = {'block{}'.format(i): {'layer{}'.format(j): [] for j in range(1,blocks_config[i-1]+1)} for i in range(1,len(blocks_config)+1)}
    labels = []
    pbar = tqdm(total=len(loaders.test), desc="Mini-Batch")
    for it, dataitem in enumerate(loaders.test):
        pbar.update()
        inputs = dataitem.inputs.to(device)
        labels.extend(dataitem.labels.detach().cpu().numpy())
        out = net(inputs)
        for i, block in enumerate(target_layers):
            for j, layer in enumerate(block):
                outputs["block%i"%(i+1)]["layer%i"%(j+1)].extend(hooks[i][j].get_hooked_value().cpu().detach().numpy())

    with open(output_file, 'wb') as f:
        pickle.dump(outputs, f)
else:
    outputs = get_pickle_obj(output_file)

if compute_svcca:
    device = 'cpu'
    ## Loads the outputs and computes the distances between all layers and store them
    distances_matrix = np.zeros((N, N))
    print('Transforming all npy arrays to torch tensors...', flush=True)
    output_tensors = {b: {l: torch.tensor(outputs[b][l], device=device) for l in outputs[b]}
                      for b in outputs}
    sizes = [16, 8, 8, 4]
    pbar = tqdm(total=N * (N + 1) / 2, desc="Nb couples done")
    for i in range(N):
        for j in range(i, N):
            pbar.update()
            (blocki, layeri), (blockj, layerj) = target_layers_flatten[i], target_layers_flatten[j]
            n_blocki, n_blockj = int(blocki[5:]), int(blockj[5:])
            # Computes the distances between the 2 representations
            distances_matrix[i, j] = 1 - CCAHook._conv3d(output_tensors[blocki][layeri],
                                                         output_tensors[blockj][layerj],
                                                         svcca_distance, sizes[n_blocki - 1],
                                                         sizes[n_blockj - 1], same_layer=False, accept_rate=0.5)['distance']

    with open(output_distances_file, 'wb') as f:
        pickle.dump({"target_layers_flatten": target_layers_flatten, "svcca_matrix": distances_matrix}, f)
else:
    svcca_results = get_pickle_obj(output_distances_file)
    distances = np.array(svcca_results['svcca_matrix'])
    distances = np.maximum(distances, distances.T)
    ticks = [sum(blocks_config[:i]) + config//2 for i,config in enumerate(blocks_config)]
    xticklabels = ['Block %i \n(%i layers)'%(i,l) for i,l in enumerate(blocks_config)]
    yticklabels = ['Block %i'%i for i,l in enumerate(blocks_config)]
    fig = plt.figure(figsize=(10,9))
    ax = seaborn.heatmap(distances, vmin=0.5, vmax=1, cmap='seismic', cbar_kws={'label': 'SVCCA'})
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(xticklabels))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(yticklabels))
    fig.savefig('svcca_matrix_densenet_dx.png', format='png')


## Data augmentation visualization
from pynet.augmentation import *
from nibabel import Nifti1Image
from nilearn.plotting import plot_anat

data = np.load(CONFIG['quasi_raw']['input_path'], mmap_mode='r')
original_img = data[0,0]
transformations = [
    original_img,
    flip(original_img, axis=2),
    add_blur(original_img, snr=[1000, 1000]),
    add_noise(original_img, snr=[10, 10]),
    Crop((97, 116, 97), "random", resize=True)(original_img),
    affine(original_img, rotation=[5, 5], translation=[10, 10], zoom=0),
    add_ghosting(original_img, intensity=[0.49, 0.5], n_ghosts=[4, 5], axis=0),
    add_motion(original_img, rotation=[40, 40], translation=[20, 20], n_transforms=3),
    add_spike(original_img, intensity=[0.20, 0.21], n_spikes=10),
    add_biasfield(original_img, coefficients=[0.7, 0.7]),
    add_swap(original_img, num_iterations=20)
]
tf_names = ['Original image', 'Flip', 'Gaussian Blur', 'Gaussian Noise',
            'Crop+Resize', 'Affine transformation', 'Ghosting Artefact', 'Motion Artefact', 'Spike Artefact',
            'Biasfield Artefact', 'Random Swap']
fig = plt.figure(constrained_layout=False, figsize=(20, 2/5*20))
nb_cols = (len(transformations)+1)//2
gs = fig.add_gridspec(5, nb_cols*5//2, height_ratios=[2.5, 2.5, 1, 2.5, 2.5])
for i, (name, img) in enumerate(zip(tf_names, transformations)):
    current_image = Nifti1Image((img-img.mean())/img.std(), np.eye(4))
    if i == 0:
        ax = fig.add_subplot(gs[1:4,i:i+3])
    else:
        ax = fig.add_subplot(gs[3*(i%2):3*(i%2)+2, 3+((i-1)//2)*2:3+((i-1)//2+1)*2])
    plot_anat(current_image, cut_coords=[70], display_mode='y', axes=ax, annotate=False,
              vmin=0, vmax=3)
    ax.set_title(name, fontweight='bold')
    ax.axis('off')
plt.savefig('data_augmentations_quasi-raw_example.png')


# Performance with DA

import operator
augmentations = ['Ghosting_Artefact',  'Spike_Artefact',  'Biasfield_Artefact', 'Motion_Artefact', 'Swap',
                 'Affine', 'Crop', 'Gaussian_Blur', 'Gaussian_Noise', 'Flip']
augmentation_names = ['Ghosting Artefact', 'Spike Artefact',  'Biasfield\nArtefact',  'Motion\nArtefact', 'Swap',
                      'Affine\nTransformation', 'Crop+Resize', 'Gaussian Blur', 'Gaussian Noise', 'Flip']

problems = ["Age", "Dx", "Sex"]
nb_epochs = [299, 99, 299]
metrics = ['mae', 'auc', 'auc']
comparison_metrics = [operator.le, operator.ge, operator.ge]
nb_folds = 5
y_limits = [(4,10), (0.7, 0.9), (0.8, 1)]
databases = ['HCP_IXI', 'SCZ_VIP', 'HCP_IXI']
preprocs = ['']
root = '/neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP'
file = 'Test_%s_%s_%s_%sfold{fold}_epoch{epoch}.pkl'
hyperparams = {}#'Crop': 'crop_115-138-115_'}#,
               #'Affine': 'rot-5_trans-10_'}#,
               #'Biasfield_Artefact': 'coeff_0.1_'}

def label_bar_plt(ax, rects, labels, **kwargs):
    """
    Attach a text label above each bar
    """
    for rect, label in zip(rects, labels):
        height = rect.get_y() + rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                label, ha='center', va='bottom', **kwargs)

fig, axes = plt.subplots(3, 1, squeeze=False, figsize=(18, 10))

for i, (pb, epochs, metric, db, comp, ylim) in enumerate(zip(problems, nb_epochs, metrics, databases,
                                                             comparison_metrics, y_limits)):
    for preproc in preprocs:
        if pb == "Dx" or pb == "Sex":
            get_metrics = get_binary_classification_metrics
        else:
            get_metrics = get_regression_metrics

        path_net = "ResNet/ResNet34" if pb == "Age" and preproc == "quasi_raw" else "DenseNet"
        net = "ResNet34" if pb == "Age" and preproc == "quasi_raw" else "DenseNet"
        baseline = get_metrics(os.path.join(root, preproc, path_net, 'N_500', pb, file%(net, pb, db, '')),
                                                     epochs_tested=nb_folds * [299], folds_tested=range(nb_folds), display=False)
        param = '_step_size_scheduler_10' if pb =="Age" and preproc == '' else ""
        results = [get_metrics(os.path.join(root, preproc, path_net, 'N_500', pb, 'Data_Augmentation', aug,
                                            file%(net, pb,db+param, hyperparams.get(aug) or '')),
                                      epochs_tested=nb_folds*[epochs], folds_tested=range(nb_folds), display=False)
                   for aug in augmentations]
        x = np.arange(len(results))
        mean = np.array([np.mean(res[metric]) for res in results])
        std = np.array([np.std(res[metric]) for res in results])
        mean_baseline = np.mean(baseline[metric])

        axes[i,0].axhline(mean_baseline, linestyle="-", color='gray', linewidth=.7, label='Baseline (w/o DA)')
        improvement_mask = comp(mean, mean_baseline)
        deterioration_mask = ~improvement_mask
        relative_imp = 100*(mean - mean_baseline)/mean_baseline

        if deterioration_mask.sum() > 0:
            neg_rects = axes[i,0].bar(x[deterioration_mask], np.abs(mean_baseline-mean[deterioration_mask]),
                                      width=0.4, color='red',
                                      bottom=np.min((mean[deterioration_mask], deterioration_mask.sum()*[mean_baseline]), axis=0),
                                      label='Degradation')
            axes[i,0].errorbar(x[deterioration_mask], mean[deterioration_mask],
                               yerr=std[deterioration_mask], capsize=3, fmt='none', ecolor='black')
            label_bar_plt(axes[i, 0], neg_rects, ['{:.2f}%'.format(p) for p in relative_imp[deterioration_mask]],
                          color='red')
        if improvement_mask.sum() > 0:
            pos_rects = axes[i,0].bar(x[improvement_mask], np.abs(mean[improvement_mask]-mean_baseline),
                                      width=0.4, color='green',
                                      bottom=np.min((mean[improvement_mask], improvement_mask.sum()*[mean_baseline]), axis=0),
                                      label='Improvement')
            axes[i,0].errorbar(x[improvement_mask], mean[improvement_mask],
                               yerr=std[improvement_mask], capsize=3, fmt='none', ecolor='black')
            label_bar_plt(axes[i, 0], pos_rects, ['{:.2f}%'.format(p) for p in relative_imp[improvement_mask]],
                          color='green')

        if i == len(problems) - 1:
            axes[i,0].set_xticks(x)
            axes[i,0].set_xticklabels(augmentation_names)
            # Group the augmentations according to the application field

        else:
            axes[i, 0].set_xticks([], [])
        axes[i,0].set_title('{pb} Prediction'.format(pb=pb.upper()), fontweight='bold')
        axes[i,0].set_ylabel(metric.upper())
        axes[i,0].set_ylim(ylim)
        axes[i,0].legend(loc='lower left')

fig.savefig('/home/benoit/Documents/Benchmark_IXI_HCP/Presentation/data_augmentation_performance_cat12.png')




