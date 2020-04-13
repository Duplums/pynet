# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define common metrics.
"""

# Third party import
import torch
import numpy as np
import torch.nn.functional as func
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

def get_confusion_matrix(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return confusion_matrix(y, y_pred.detach().cpu().numpy())

def roc_auc(y_pred, y):
    return roc_auc_score(y, y_pred[:,1].detach().cpu().numpy())

def balanced_accuracy(y_pred, y):
    if len(y_pred.shape) == 1:
        y_pred = (y_pred > 0)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.data.max(dim=1)[1] # get the indices of the maximum
    return balanced_accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

# Apply the 3D Sobel filter to an input pytorch tensor
class Sobel3D:
    def __init__(self, padding=0, norm=False, device='cpu'):
        h = [1, 2, 1]
        h_d = [1, 0, -1]
        G_z = [[[h_d[k] * h[i] * h[j] for k in range(3)] for j in range(3)] for i in range(3)]
        G_y = [[[h_d[j] * h[i] * h[k] for k in range(3)] for j in range(3)] for i in range(3)]
        G_x = [[[h_d[i] * h[j] * h[k] for k in range(3)] for j in range(3)] for i in range(3)]
        self.G = torch.tensor([[G_x], [G_y], [G_z]], dtype=torch.float, device=device)
        self.padding = padding
        self.norm = norm

    def __call__(self, x):
        # x: 3d tensor (B, C, T, H, W)
        x_filtered =  nn.functional.conv3d(x, self.G, padding=self.padding)
        if self.norm:
            x_filtered = torch.sqrt(torch.sum(x_filtered ** 2, dim=1)).unsqueeze(1)
        return x_filtered

def accuracy(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1] # get the indices of the maximum
    accuracy = y_pred.eq(y).sum().cpu().numpy() / y.size()[0]
    return accuracy

# True Positive Rate = TP/P (also called Recall)
def sensitivity(y_pred, y, positive_label=1):
    y_pred = y_pred.data.max(dim=1)[1]
    TP = (y_pred.eq(y) & y.eq(positive_label)).sum().cpu().numpy()
    P = y.eq(positive_label).sum().cpu().numpy()
    if P == 0:
        return 0.0
    return float(TP/P)

# True Negative Rate = TN/N
def specificity(y_pred, y, negative_label=0):
    y_pred = y_pred.data.max(dim=1)[1]
    TN = (y_pred.eq(y) & y.eq(negative_label)).sum().cpu().numpy()
    N = y.eq(negative_label).sum().cpu().numpy()
    if N == 0:
        return 0.0
    return float(TN/N)


def RMSE(y_pred, y):
    rmse = torch.sqrt(torch.mean((y_pred - y)**2)).detach().cpu().numpy()
    return float(rmse)

def _dice(y_pred, y):
    """ Binary dice indice adapted to pytorch tensors.
    """
    flat_y_pred = torch.flatten(y_pred)
    flat_y = torch.flatten(y)
    intersection = (flat_y_pred * flat_y).sum()
    return (2. * intersection + 1.) / (flat_y_pred.sum() + flat_y.sum() + 1.)


def multiclass_dice(y_pred, y):
    """ Extension of the dice to a n classes problem.
    """
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    y_pred = func.softmax(y_pred, dim=1)
    dice = 0.
    n_classes = y.shape[1]
    for cnt in range(n_classes):
        dice += _dice(y_pred[:, cnt], y[:, cnt])
    return dice / n_classes


METRICS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "multiclass_dice": multiclass_dice,
    "RMSE": RMSE,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "confusion_matrix": get_confusion_matrix,
    "roc_auc": roc_auc
    # cf. scikit doc: " The binary case expects a shape (n_samples,), and the scores
    # must be the scores of the class with the greater label."
}
