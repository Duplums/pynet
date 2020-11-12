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
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.manifold import TSNE
from skimage.transform import resize
from sklearn.decomposition import PCA
import nilearn
from pynet.history import History
from nilearn.image import new_img_like
from nilearn.plotting import plot_anat, plot_stat_map
from nibabel import Nifti1Image
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

def plot_data_reduced(X, labels=None, reduction='pca', cmap=plt.cm.plasma, ax=None, title=None):
    # Assume that X has dimension (n_samples, ...) and labels is a list of n_samples labels
    # associated to X
    assert reduction in ['pca', 't_sne'], "Reduction method not implemented yet"
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 30))

    if reduction == 'pca':
        pca = PCA(n_components=2)
        # Do the SVD
        pca.fit(X.reshape(len(X), -1))
        # Apply the reduction
        PC = pca.transform(X.reshape(len(X), -1))
    else:
        PC = TSNE(n_components=2).fit_transform(X.reshape(len(X), -1))
    # Color each point according to its label
    if labels is not None:
        labels = np.array(labels)
        label_mapping = {l: cmap(int(i * float(cmap.N - 1) / len(set(labels)))) for (i, l) in enumerate(set(labels))}
        for l in label_mapping:
            ax.scatter(PC[:,0][labels == l], PC[:,1][labels == l], c=[label_mapping[l]], label=l)
    else:
        ax.scatter(PC[:,0], PC[:,1], alpha=0.8)
    if reduction == 'pca':
        ax.set_xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
        ax.set_ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
    ax.legend()
    if title:
        ax.set_title(title)

def plot_losses(train_history, val_history=None, patterns_to_del=None,
                metrics=None, experiment_names=None, titles=None, ylabels=None,
                saving_path=None, output_format="png", ylim=None, same_plot=False, **kwargs):
    """
    :param train_history: History object or list of History objects
        a history from a training process including several metrics
    :param val_history: History object or list of History objects
        the validation history of the training process. The metric's names could be slightly different
    :param patterns_to_del: list or str
        patterns to del from the metrics names
    :param experiment_names: list of str
        The name of the experiments corresponding to the histories
    :param titles: dict
        a mapping between a metric and the corresponding plot title
    :param ylabels: dict
        a mapping between a metric and the corresponding y-axis title
    :param same_plot: bool
        if True, plots all the metrics in the same figure
    :param saving_path: str
        the path where the the plot will be saved
    :param output_format: str ('png', 'jpg', 'pdf'...)
        the output format of the plot that will be saved
    :param **kwargs
        arguments given to plt.subplots(...)
    :return:
    """
    if isinstance(train_history, History):
        train_history = [train_history]
    if isinstance(val_history, History):
        val_history = [val_history]

    list_dict_training = [t.to_dict(patterns_to_del=patterns_to_del, drop_last=True) for t in train_history]
    if val_history is not None:
        list_dict_val = [t.to_dict(patterns_to_del=patterns_to_del, drop_last=True) for t in val_history]
        assert len(list_dict_training) == len(list_dict_val), "Unexpected number of validation experiments"
    else:
        list_dict_val = None

    _metrics = set.intersection(*[set(t.keys()) for t in list_dict_training])
    if metrics is not None: # keep the order
        _metrics = [m for m in metrics if m in _metrics]
    else:
        metrics = _metrics

    n_rows = int(np.floor(np.sqrt(len(list_dict_training))))
    n_cols = int(np.ceil(np.sqrt(len(list_dict_training))))
    if n_rows * n_cols < len(list_dict_training):
        n_rows = n_cols


    if same_plot:
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, **kwargs)

    for metric in metrics:
        if not same_plot:
            fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, **kwargs)
        for (i, dict_train) in enumerate(list_dict_training):
            (row, col) = (i%n_rows, i//n_rows)
            if metric not in dict_train:
                continue
            Y_train = dict_train[metric]
            Y_val = list_dict_val[i][metric] if (list_dict_val is not None and metric in list_dict_val[i]) else None
            X = list(range(len(Y_train[0])))

            if same_plot:
                exp_name = (ylabels.get(metric) or '') + ' (training)' if ylabels is not None else 'Training'
            else:
                exp_name = experiment_names[i] +' (training)' if experiment_names is not None else 'Training'
            p = axes[row, col].plot(X, np.quantile(Y_train, 0.5, axis=0), label=exp_name)
            axes[row, col].fill_between(X, np.quantile(Y_train, 0.25, axis=0),
                                         np.quantile(Y_train, 0.75, axis=0), facecolor=p[0].get_color(),
                                         alpha=0.3)
            if Y_val is not None:
                if same_plot:
                    exp_name = (ylabels.get(metric) or '') + ' (validation)' if ylabels is not None else 'Validation'
                else:
                    exp_name = experiment_names[i] + ' (val)' if experiment_names is not None else 'Validation'
                p = axes[row, col].plot(X, np.quantile(Y_val, 0.5, axis=0), label=exp_name)
                axes[row, col].fill_between(X, np.quantile(Y_val, 0.25, axis=0),
                                             np.quantile(Y_val, 0.75, axis=0), facecolor=p[0].get_color(), alpha=0.3)

            axes[row, col].set_xlabel("Epochs")
            if not same_plot:
                axes[row, col].set_ylabel((ylabels or dict()).get(metric) or metric)
            axes[row, col].legend(loc='lower right')
            if experiment_names is not None:
                axes[row, col].set_title(experiment_names[i], fontsize=12)
            axes[row, col].grid()
            if ylim is not None:
                if isinstance(ylim, list):
                    axes[row, col].set_ylim(ylim)
                elif isinstance(ylim, dict):
                    try:
                        axes[row, col].set_ylim(ylim[metric])
                    except KeyError:
                        pass
            axes[row, col].grid()
        if titles is not None and metric in titles:
            fig.suptitle(titles[metric], fontsize=14)
        plt.tight_layout()
        if saving_path and not same_plot:
            plt.savefig(saving_path+'_'+metric, format=output_format)
    if saving_path and same_plot:
        plt.savefig(saving_path, format=output_format)

    return fig, axes

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


def linear_reg_plots(Y_pred, Y_true, labels=None, cmap=plt.cm.plasma, axes=None, title=''):
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr

    reg = LinearRegression().fit(Y_true, Y_pred)
    coefs, intercepts = reg.coef_, reg.intercept_
    (r, pval) = pearsonr(Y_pred.flatten(), Y_true.flatten())
    MAE = np.mean(np.abs(Y_pred - Y_true))
    RMSE = np.sqrt(np.mean(np.abs(Y_pred-Y_true)**2))
    if axes is None:
        fig, axes = plt.subplots()
    if labels is not None:
        label_mapping = {l: cmap(int(i*float(cmap.N-1)/len(set(labels)))) for (i,l) in enumerate(set(labels))}
        for l in label_mapping:
            axes.scatter(Y_true[labels==l], Y_pred[labels==l], c=[label_mapping[l]], label=l)
    else:
        axes.scatter(Y_true, Y_pred)

    axes.plot(Y_true, Y_true, color='red', label='Perfect case')
    axes.plot(Y_true, [coefs[0]*y[0]+intercepts[0] for y in Y_true], color='green',
              label='Linear Regression\n($R^2={R2:.2f}$, $r={r:.2f}$, p-value={pval:1.2e})'.
              format(R2=reg.score(Y_true, Y_pred), r=r, pval=pval))
    axes.legend(loc='upper left')
    axes.set_xlabel('True age (years)')
    axes.set_ylabel('Predicted age')
    axes.set_title(title+'\nMAE={mae:.2f}, RMSE={rmse:.2f}'.format(mae=MAE, rmse=RMSE))
    plt.show()


def plot_3d_data(data, nb_samples=3, channel=None, channel_names=None,
                 n_slices_per_dim=4, random=True, title=None,
                 ref_nii_img=None, saving_path=None, saving_format='png', **kwargs):
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
    ref_nii_img: image, default None [can be str as well]
        Reference image to be used by nilearn to plot the slices
    cmap: plt.cmap object
    saving_path: str, default None
        if not None, the path to the saved image
    kwargs: dict
        it will be passed to the nilearn function <plot_anat> directly
    """
    # Check input parameters
    if data.ndim != 5 :
        raise ValueError("Unsupported data dimension.")

    (total_samples, nb_channels, *ndims) = data.shape
    nb_samples = min(nb_samples, total_samples)
    if channel is not None:
        channels = [channel]
    else:
        channels = list(range(nb_channels))

    if channel_names is not None:
        assert len(channel_names) == len(channels)

    if random:
        indices = np.random.randint(0, total_samples, nb_samples)
    else:
        indices = np.arange(nb_samples)

    cuts = ['x', 'y', 'z']
    cuts_name = ['SAG', 'COR', 'AXI']
    colorbar = kwargs.pop('colorbar', False)

    (n_rows, n_cols) = (len(channels)*len(indices), 3*n_slices_per_dim+1) # (+1 for channel name)
    empty_space_ratio = 1/10 # % of the image size
    cbar_width_space = 0.2 # % of the image width dedicated to color bar
    right_cbar = 1 - cbar_width_space / n_slices_per_dim
    image_width = 4

    # The scale to preserve all the time (careful with subplots_adjust)
    fig_scale = ((n_rows+(nb_samples-1)*empty_space_ratio) * ndims[0])/(n_cols * ndims[1])
    fig_width = image_width * n_slices_per_dim
    fig_height = fig_width * fig_scale

    height_ratios = np.kron(np.ones(nb_samples), list(np.ones(len(channels)))+[empty_space_ratio])[:-1]

    fig, axes = plt.subplots(n_rows+(nb_samples-1), n_cols, figsize=(fig_width, fig_height), dpi=200, clear=True,
                             gridspec_kw=dict(wspace=0.0, hspace=0.0, height_ratios=height_ratios), squeeze=False)

    for cnt1, ind in enumerate(indices):
        for cnt2, ch in enumerate(channels):
            # <nb_channels> rows per sample
            # This is the index of the current line
            current_index = np.ravel_multi_index((cnt1, cnt2), (len(indices), len(channels))) + cnt1

            if ref_nii_img is not None:
                current_image = new_img_like(ref_nii_img, data=data[ind, ch])
            else:
                current_image = Nifti1Image(data[ind, ch], np.eye(4))
            # Get the best cut slices by dimension. We assume all the channels have the same best cut slices.
            if cnt2 == 0 and cnt1 == 0:
                best_cut_slices = [nilearn.plotting.find_cut_slices(current_image, cuts[i], n_slices_per_dim)
                                   for i in range(3)]
            for col in range(n_cols-1):
                plot_func = plot_stat_map if 'bg_img' in kwargs else plot_anat

                display = plot_func(current_image, display_mode=cuts[col//n_slices_per_dim],
                                    cut_coords=[best_cut_slices[col//n_slices_per_dim][col%n_slices_per_dim]],
                                    axes=axes[current_index, col+1],
                                    annotate=False, colorbar=False, **kwargs)
                display.annotate(size=5)

                if cnt2 == 0 and cnt1>0: # erase the empty axis ticks and frame
                    axes[current_index-1, col+1].remove()
                axes[current_index, col+1].axis("off")
                if cnt1+cnt2 == 0:
                    axes[current_index, col+1].set_title(cuts_name[col//n_slices_per_dim], loc='center', fontsize=6)
            # Add the legend for all channels
            title_ax = axes[current_index, 0]
            title_ax.axis("off")
            if channel_names is not None:
                if nb_samples > 1:
                    title_ax.text(0.5, 0.5, '%s (sample %i)'%(channel_names[ch], ind), horizontalalignment='center',
                                  verticalalignment='center', transform=title_ax.transAxes, fontsize=6, wrap=True)
                else:
                    title_ax.text(0.5, 0.5, '%s'%channel_names[ch], horizontalalignment='center',
                                  verticalalignment='center', transform=title_ax.transAxes, fontsize=6, wrap=True)
            elif nb_samples > 1:
                title_ax.text(0.5, 0.5, 'Sample %i' % ind, horizontalalignment='center',
                              verticalalignment='center', transform=title_ax.transAxes, fontsize=6, wrap=True)
        if cnt1 > 0:
            axes[cnt1*(len(channels)+1)-1, 0].remove()
    if colorbar:
        if 'vmax' not in kwargs:
            raise ValueError('vmax not set, there is no unique color bar for all the pictures. Please specify a value.')
        vmax = kwargs.get('vmax')
        cmap = kwargs.get('cmap')
        plt.subplots_adjust(left=0, right=right_cbar, bottom=0, top=right_cbar*fig_scale, wspace=0, hspace=0)
        cax = fig.add_axes([(3*right_cbar+1)/4, 0, (1-right_cbar)/2, right_cbar*fig_scale*(nb_channels/(nb_channels+1))])
        fig.colorbar(cm.ScalarMappable(colors.Normalize(vmin=-vmax, vmax=vmax), cmap=cmap), cax=cax)

    if title is not None:
        fig.suptitle(title, y=0.99, fontsize=10)

    if saving_path is not None:
        plt.savefig(saving_path, dpi=200, format=saving_format)

    plt.show()

# Global parameters
logger = logging.getLogger("pynet")


def plot_data(data, slice_axis=2, nb_samples=5, channel=0, labels=None,
              random=True, rgb=False, cmap=None, title=None):
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
    title: str, default None
        the figure title.
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
    fig = plt.figure(figsize=(15, 7), dpi=200)
    fig.title = title
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
    logger.debug(mask.shape, len(valid_indices))

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
