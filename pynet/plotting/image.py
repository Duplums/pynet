# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to display images.
"""

# Import
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage.transform import resize
from sklearn.decomposition import PCA
import torchvision
import random

def display_attention_maps(images, attention_maps, slicing_mode="middle_slicing", nb_samples=5):
    # images: numpy array of shape (samples, channels, dim)
    # attention_maps: numpy array of shape (samples, channels, dim)
    nb_samples = min(nb_samples, len(images))
    (N, C, H, W, D) = images.shape

    if slicing_mode == "middle_slicing":
        slices = [(H // 2, 0), (W // 2, 1), (D // 2, 2)]

    elif slicing_mode == "quart_slicing":
        slices = [(H // 4, 0), (H // 2, 0), (3*H // 4, 0)]
    else:
        raise NotImplementedError("Not yet implemented !")

    n_rows, n_cols = len(slices) * C, nb_samples
    fig = plt.figure(figsize=(15, 7), dpi=200)
    fig.clf()
    for i, n in enumerate(random.sample(range(N), nb_samples)):
        for j in range(C):
            for k, (indice, axis) in enumerate(slices):
                im = images[i, j]
                a_map = resize(attention_maps[i, j], (H, W, D))

                im = np.take(im, indice, axis=axis)
                attention_im = np.take(a_map, indice, axis=axis)

                plt.subplot(n_rows, n_cols, k * C * nb_samples + j * nb_samples + i + 1)
                plt.imshow(im, interpolation='bilinear', cmap='gray')
                plt.imshow(attention_im, interpolation='bilinear', cmap=cm.jet, alpha=0.2)
                plt.axis('off')
                plt.colorbar()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def age_discrimination(X, y, age_max_down, age_min_up):
    from scipy import ndimage
    X_1 = np.sum(X[y < age_max_down], axis=0)
    X_2 = np.sum(X[y > age_min_up], axis=0)

    X_diff = (X_2 - X_1)[0]
    (H, W, D) = X_diff.shape

    plt.subplot(1, 3, 1)
    plt.imshow(ndimage.rotate(X_diff[H//2,:,:], 90), cmap='coolwarm')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(ndimage.rotate(X_diff[:,W//2,:], 90), cmap='coolwarm')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(ndimage.rotate(X_diff[:,:,D//2], 90), cmap='coolwarm')
    plt.axis('off')
    plt.colorbar()
    fig = plt.gcf()
    fig.suptitle('Difference between [{}, {}]y ({} people) and [{}, {}]y ({} people)'.format(
        np.min(y), age_max_down, len(X[y < age_max_down]), age_min_up, np.max(y), len(X[y > age_min_up])))
    plt.show()



def plot_pca(X, labels=None, cmap=plt.cm.plasma, title=None):
    # Assume that X has dimension (n_samples, ...) and labels is a list of n_samples labels
    # associated to X
    pca = PCA(n_components=2)
    # Do the SVD
    pca.fit(X.reshape(len(X), -1))
    # Apply the reduction
    PC = pca.transform(X.reshape(len(X), -1))
    fig, ax = plt.subplots(figsize=(20, 30))
    # Color each point according to its label
    if labels is not None:
        labels = np.array(labels)
        label_mapping = {l: cmap(int(i * float(cmap.N - 1) / len(set(labels)))) for (i, l) in enumerate(set(labels))}
        for l in label_mapping:
            plt.scatter(PC[:,0][labels == l], PC[:,1][labels == l], c=[label_mapping[l]], label=l)
    else:
        plt.scatter(PC[:,0], PC[:,1])
    plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
    plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
    plt.legend()
    if title:
        plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_losses(train_history, val_history=None, patterns_to_del=None,
                metrics=None, experiment_names=None, titles=None, ylabels=None,
                saving_path=None, output_format="png", ylim=None):
    """
    :param train_history: History object or list of History objects
        a history from a training process including several metrics
    :param val_history: History object or list of History objects
        the validation history of the training process. The metric's names could be slightly different
    :param patterns_to_del: list or str
        patterns to del from the metrics names
    :param titles: dict
        a mapping between a metric and the corresponding plot title
    :param ylabels: dict
        a mapping between a metric and the corresponding y-axis title
    :param saving_path: str
        the path where the the plot will be saved
    :param output_format: str ('png', 'jpg', 'pdf'...)
        the output format of the plot that will be saved
    :return:
    """
    if isinstance(train_history, History):
        train_history = [train_history]
    if isinstance(val_history, History):
        val_history = [val_history]

    list_dict_training = [t.to_dict(patterns_to_del=patterns_to_del, drop_last=True) for t in train_history]
    if val_history is not None:
        list_dict_val = [t.to_dict(patterns_to_del=patterns_to_del, drop_last=True) for t in val_history]
        assert len(list_dict_training) == len(list_dict_val), "Unexpected number of validation exepriments"
    else:
        list_dict_val = None

    _metrics = set.intersection(*[set(t.keys()) for t in list_dict_training])
    if metrics is not None: # keep the order
        _metrics = [m for m in metrics if m in _metrics]

    fig, axes = plt.subplots(len(_metrics), 1, figsize=(10, 10), squeeze=False)
    experiment_names = experiment_names or ['Exp %i'%i for i in list_dict_training]

    for ax_indice, metric in enumerate(metrics):
        for (i, dict_train) in enumerate(list_dict_training):
            Y_train = dict_train[metric]
            Y_val = list_dict_val[i][metric] if (list_dict_val is not None and metric in list_dict_val[i]) else None
            X = list(range(len(Y_train[0])))
    
            p = axes[ax_indice,0].plot(X, np.quantile(Y_train, 0.5, axis=0), label=experiment_names[i]+' (training)')
            axes[ax_indice,0].fill_between(X, np.quantile(Y_train, 0.25, axis=0),
                                         np.quantile(Y_train, 0.75, axis=0), facecolor=p[0].get_color(),
                                         alpha=0.3)
            if Y_val is not None:
                p = axes[ax_indice,0].plot(X, np.quantile(Y_val, 0.5, axis=0), label=experiment_names[i]+' (val)')
                axes[ax_indice,0].fill_between(X, np.quantile(Y_val, 0.25, axis=0),
                                             np.quantile(Y_val, 0.75, axis=0), facecolor=p[0].get_color(), alpha=0.3)
    
            axes[ax_indice,0].set_xlabel("Epochs")
            axes[ax_indice,0].set_ylabel((ylabels or dict()).get(metric) or metric)
            axes[ax_indice,0].set_title("\n\n"+((titles or dict()).get(metric) or metric))
            axes[ax_indice,0].legend(loc='upper left')
            if ylim is not None:
                if isinstance(ylim, list):
                    axes[ax_indice,0].set_ylim(ylim)
                elif isinstance(ylim, dict):
                    try:
                        axes[ax_indice,0].set_ylim(ylim[metric])
                    except KeyError:
                        pass
            axes[ax_indice,0].grid()

    plt.tight_layout()
    plt.show()

    if saving_path:
        plt.savefig(saving_path, format=output_format)

if __name__ == '__main__':
    from pynet.history import History
    h_resnet = History.load('/neurospin/psy_sbox/bd261576/checkpoints/scz_prediction/tmp/Train_PsyNet_ResNet_Pretrained_4_epoch_99.pkl')
    h_resnet_val = History.load('/neurospin/psy_sbox/bd261576/checkpoints/scz_prediction/tmp/Validation_PsyNet_ResNet_Pretrained_4_epoch_99.pkl')

    h = History.load('/neurospin/psy_sbox/bd261576/checkpoints/scz_prediction/tmp/Train_PsyNet_4_epoch_99.pkl')
    h_val = History.load('/neurospin/psy_sbox/bd261576/checkpoints/scz_prediction/tmp/Validation_PsyNet_4_epoch_99.pkl')

    plot_losses(h, h_val,
                patterns_to_del=['validation_', ' on validation set'],
                metrics=['gm_loss', 'likelihood_u'],
                titles={'gm_loss': 'Gaussian Mixture Loss ($\\alpha_m=0.5$)',
                        'likelihood_u': 'Negative log-likelihood of $z_u$'},
                ylabels={'gm_loss': '$L_{GM}$',
                         'likelihood_u': "$-\log(L_{lkd}^u)$"},
                experiment_names=['$E_s=E_u$'],
                ylim={"gm_loss": [0, 1000], 'likelihood_u':[0, 1000]},
                saving_path='/home/bd261576/Documents/BenchMark_IXI_HCP/dx_prediction_GMLoss_PsyNet.png')


def plot_2_set_losses(x, loss_1, loss_2):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# We assume a binary classification where Y_true has shape (n_samples,) and Y_pred has shape (n_samples, 2)
# or (n_samples,)
def roc_curve_plot(Y_pred, Y_true, title=None):
    from sklearn import metrics
    n_samples = len(Y_true)
    assert n_samples == len(Y_pred)
    assert len(Y_pred.shape) <= 2
    if len(Y_pred.shape) == 2:
        Y_pred = Y_pred[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(Y_true, Y_pred)
    auc = metrics.roc_auc_score(Y_true, Y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    if title:
        plt.title('ROC curve of {}\nAUC={}'.format(title, auc))
    else:
        plt.title('ROC curve\nAUC={}'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()



def linear_reg_plots(Y_pred, Y_true, labels=None, cmap=plt.cm.plasma):
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr

    reg = LinearRegression().fit(Y_true, Y_pred)
    coefs, intercepts = reg.coef_, reg.intercept_
    (r, pval) = pearsonr(Y_pred.flatten(), Y_true.flatten())
    MAE = np.mean(np.abs(Y_pred - Y_true))
    RMSE = np.sqrt(np.mean(np.abs(Y_pred-Y_true)**2))

    if labels is not None:
        label_mapping = {l: cmap(int(i*float(cmap.N-1)/len(set(labels)))) for (i,l) in enumerate(set(labels))}
        for l in label_mapping:
            plt.scatter(Y_true[labels==l], Y_pred[labels==l], c=[label_mapping[l]], label=l)
    else:
        plt.scatter(Y_true, Y_pred)

    plt.plot(Y_true, Y_true, color='red', label='Perfect case')
    plt.plot(Y_true, [coefs[0]*y[0]+intercepts[0] for y in Y_true], color='green', label='Linear Regression')
    plt.legend()
    plt.xlabel('True age')
    plt.ylabel('Predicted age')
    plt.title('Linear regression: $R^2={R2:.2f}$, $r={r:.2f}$ (p-value {pval:1.2e} for $H_0$=not correlate),\n'
              'MAE={mae:.2f}, RMSE={rmse:.2f}'.format(R2=reg.score(Y_true, Y_pred), r=r, pval=pval, mae=MAE, rmse=RMSE))
    plt.show()


def plot_anat_array(data, **kwargs):
    import nilearn.plotting as pt
    import nibabel
    nii_data = nibabel.Nifti1Image(np.array(data, dtype=np.float32), np.eye(4, dtype=np.float32))
    v = pt.plot_img(nii_data, **kwargs)


def plot_3d_data(data, nb_samples=3, channel=None, channel_names=None,
                 n_slices_per_dim=3, random=True, cmap=None,
                 saving_path=None, saving_format='png'):
    """ Plot several samples of 3d data on 3 axis: Sagital (SAG), Coronal (COR) and Axial (AXI)

    Currently supports 3D dataset of the form (samples, channels, dim).

    Parameters
    ----------
    data: array (samples, channels, dim)
        the data to be displayed.
    nb_samples: int, default 5
        the number of samples to be displayed.
    channel: int, default None
        will select slices with data using the provided channel. If None, selects all the channels
    channel_names: list of str, default None
        the channel names to be displayed.
    n_slices_per_dim: int, default 3
        selects the nb of slices to show per input dimension
    random: bool, default True
        select randomly 'nb_samples' data, otherwise the 'nb_samples' firsts.
    cmap: plt.cmap object
    saving_path: str, default None
        if not None, the path to the saved image
    """
    # Check input parameters
    if data.ndim != 5 :
        raise ValueError("Unsupported data dimension.")

    (total_samples, nb_channels, *ndims) = data.shape
    if channel is not None:
        channels = [channel]
    else:
        channels = list(range(nb_channels))
    
    if channel_names is not None:
        assert len(channel_names) == len(channels)
    else:
        channel_names = ["ch %i"%i for i in channels]

    if random:
        indices = np.random.randint(0, total_samples, nb_samples)
    else:
        indices = np.range(nb_samples)

    # Get the slice index in each direction (Sagittal, Coronal, Axial)
    slices = [[int(ndims[dim] * ((s + 1) / (n_slices_per_dim + 1))) for dim in range(3)]
              for s in range(n_slices_per_dim)]
    # Now, introduce the ":" slice on each remaining dimension
    # For instance, if the first dim has the slice index [[32,..], [64,..], [96,..]] then creates
    # [[32, :, :], [64, :, :], [96, :, :],...]
    all_slices = np.zeros((3*n_slices_per_dim, 3), dtype=np.object)
    for s in range(3*n_slices_per_dim):
        for slicer in range(3):
            if slicer==(s//n_slices_per_dim):
                all_slices[s][slicer] = slices[s%n_slices_per_dim][slicer]
            else:
                all_slices[s][slicer] = slice(None)

    slices_name = ['SAG', 'COR', 'AXI']
    slices_name = ['%s \n slice %i'%(slices_name[i], slices[c][i]) for i in range(3) for c in range(n_slices_per_dim)]

    (n_rows, n_cols) = (len(channels)*len(indices), 3*n_slices_per_dim+1) # (+1 for channel name)
    empty_space_ratio = 1/10 # % of the image size

    fig_width = 2 * n_slices_per_dim
    fig_height = fig_width * ((n_rows+(nb_samples-1)*empty_space_ratio) * ndims[0])/(n_cols * ndims[1])
    height_ratios = np.kron(np.ones(nb_samples), list(np.ones(len(channels)))+[empty_space_ratio])[:-1]
    fig, axes = plt.subplots(n_rows+(nb_samples-1), n_cols, figsize=(fig_width, fig_height), dpi=200, clear=True,
                             gridspec_kw=dict(wspace=0.0, hspace=0.0, height_ratios=height_ratios))

    for cnt1, ind in enumerate(indices):
        for cnt2, ch in enumerate(channels):
            # <nb_channels> rows per sample
            # This is the index of the current line
            current_index = np.ravel_multi_index((cnt1, cnt2), (len(indices), len(channels))) + cnt1
            for col in range(n_cols-1):
                image = data[ind, ch, all_slices[col][0], all_slices[col][1], all_slices[col][2]]
                cmap = cmap or "gray"
                if cnt2 == 0 and cnt1>0: # erase the empty axis ticks and frame
                    axes[current_index-1, col+1].remove()
                axes[current_index, col+1].axis("off")
                if cnt1+cnt2 == 0:
                    axes[current_index, col+1].set_title(slices_name[col], loc='center', fontsize=6)
                axes[current_index, col+1].imshow(image, cmap=cmap)
            # Add the legend for all channels
            title_ax = axes[current_index, 0]
            title_ax.axis("off")
            title_ax.text(0.5, 0.5, '%s (sample %i)'%(channel_names[ch], ind), horizontalalignment='center',
                          verticalalignment='center', transform=title_ax.transAxes, fontsize=4)
        if cnt1 > 0:
            axes[cnt1*(len(channels)+1)-1, 0].remove()
    plt.show()
    if saving_path is not None:
        plt.savefig(saving_path, dpi=200, format=saving_format)


def plot_data(data, slice_axis=2, nb_samples=5, channel=0, labels=None,
              random=True, rgb=False, cmap=None):
    """ Plot an image associated data.

    Currently support 2D or 3D dataset of the form (samples, channels, dim).

    Parameters
    ----------
    data: array (samples, channels, dim)
        the data to be displayed.
    slice_axis: int, default 2
        the slice axis for 3D data.
    nb_samples: int, default 5
        the number of samples to be displayed.
    channel: int, default 0
        will select slices with data using the provided channel.
    labels: list of str, default None
        the data labels to be displayed.
    random: bool, default True
        select randomly 'nb_samples' data, otherwise the 'nb_samples' firsts.
    rgb: bool, default False
        if set expect three RGB channels.
    """
    # Check input parameters
    if data.ndim not in range(4, 6):
        raise ValueError("Unsupported data dimension.")
    nb_channels = data.shape[1]
    if rgb:
        if nb_channels != 3:
            raise ValueError("With RGB mode activated expect exactly 3 "
                             "channels.")
        else:
            nb_channels = 1

    # Reorganize 3D data
    if data.ndim == 5:
        indices = [0, 1, 2]
        assert slice_axis in indices
        indices.remove(slice_axis)
        indices = [slice_axis + 1, 0, indices[0] + 1, indices[1] + 1]
        slices = [img.transpose(indices) for img in data]
        data = np.concatenate(slices, axis=0)

    valid_indices = [
        idx for idx in range(len(data)) if data[idx, channel].max() > 0]

    # Plot data on grid
    # plt.figure()
    # _data = torchvision.utils.make_grid(torch.from_numpy(data))
    # _data = _data.numpy()
    # plt.imshow(np.transpose(_data, (1, 2, 0)))
    if random:
        indices = np.random.randint(0, len(valid_indices), nb_samples)
    else:
        if len(valid_indices) < nb_samples:
            nb_samples = len(valid_indices)
        indices = range(nb_samples)
    plt.figure(figsize=(15, 7), dpi=200)
    for cnt1, ind in enumerate(indices):
        ind = valid_indices[ind]
        for cnt2 in range(nb_channels):
            if rgb:
                im = data[ind].transpose(1, 2, 0)
                cmap = cmap or None
            else:
                im = data[ind, cnt2]
                cmap = cmap or "gray"
            plt.subplot(nb_channels, nb_samples, nb_samples * cnt2 + cnt1 + 1)
            plt.axis("off")
            if cnt2 == 0 and labels is None:
                plt.title("Image " + str(ind))
            elif cnt2 == 0:
                plt.title(labels[ind])
            plt.imshow(im, cmap=cmap)


def plot_segmentation_data(data, mask, slice_axis=2, nb_samples=5):
    """ Display 'nb_samples' images and segmentation masks stored in data and
    mask.

    Currently support 2D or 3D dataset of the form (samples, channels, dim).

    Parameters
    ----------
    data: array (samples, channels, dim)
        the data to be displayed.
    mask: array (samples, channels, dim)
        the mask data to be overlayed.
    slice_axis: int, default 2
        the slice axis for 3D data.
    nb_samples: int, default 5
        the number of samples to be displayed.
    """
    # Check input parameters
    if data.ndim not in range(4, 6):
        raise ValueError("Unsupported data dimension.")

    # Reorganize 3D data
    if data.ndim == 5:
        indices = [0, 1, 2]
        assert slice_axis in indices
        indices.remove(slice_axis)
        indices = [slice_axis + 1, 0, indices[0] + 1, indices[1] + 1]
        slices = [img.transpose(indices) for img in data]
        data = np.concatenate(slices, axis=0)
        slices = [img.transpose(indices) for img in mask]
        mask = np.concatenate(slices, axis=0)
    mask = np.argmax(mask, axis=1)
    valid_indices = [idx for idx in range(len(mask)) if mask[idx].max() > 0]
    print(mask.shape, len(valid_indices))

    # Plot data on grid
    indices = np.random.randint(0, len(valid_indices), nb_samples)
    plt.figure(figsize=(15, 7), dpi=200)
    for cnt, ind in enumerate(indices):
        ind = valid_indices[ind]
        im = data[ind, 0]
        plt.subplot(2, nb_samples, cnt + 1)
        plt.axis("off")
        # plt.title("Image " + str(ind))
        plt.imshow(im, cmap="gray")
        mask_im = mask[ind]
        plt.subplot(2, nb_samples, cnt + 1 + nb_samples)
        plt.axis("off")
        plt.imshow(mask_im, cmap="jet")
        plt.imshow(im, cmap="gray", alpha=0.4)


def rescale_intensity(arr, in_range, out_range):
    """ Return arr after stretching or shrinking its intensity levels.

    Parameters
    ----------
    arr: array
        input array.
    in_range, out_range: 2-tuple
        min and max intensity values of input and output arr.

    Returns
    -------
    out: array
        array after rescaling its intensity.
    """
    imin, imax = in_range
    omin, omax = out_range
    out = np.clip(arr, imin, imax)
    out = (out - imin) / float(imax - imin)
    return out * (omax - omin) + omin
