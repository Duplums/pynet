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

def linear_reg_plots(Y_pred, Y_true):
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr

    reg = LinearRegression().fit(Y_pred, Y_true)
    coefs, intercepts = reg.coef_, reg.intercept_
    (r, pval) = pearsonr(Y_pred.flatten(), Y_true.flatten())
    MAE = np.mean(np.abs(Y_pred - Y_true))
    RMSE = np.sqrt(np.mean(np.abs(Y_pred-Y_true)**2))

    plt.scatter(Y_pred, Y_true)
    plt.plot(Y_pred, Y_pred, color='red', label='Perfect case')
    plt.plot(Y_pred, [coefs[0]*y[0]+intercepts[0] for y in Y_pred], color='green', label='Linear Regression')
    plt.legend()
    plt.xlabel('Predicted age')
    plt.ylabel('True age')
    plt.title('Linear regression: $R^2={R2:.2f}$, $r={r:.2f}$ (p-value {pval:.2f} for H0=correlate),\n'
              'MAE={mae:.2f}, RMSE={rmse:.2f}'.format(R2=reg.score(Y_pred, Y_true), r=r, pval=pval, mae=MAE, rmse=RMSE))
    plt.show()


def plot_data(data, slice_axis=2, nb_samples=5, channel=0, labels=None,
              random=True, rgb=False):
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
                cmap = None
            else:
                im = data[ind, cnt2]
                cmap = "gray"
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
