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
from sklearn.metrics import roc_auc_score


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
    "multiclass_dice": multiclass_dice,
    "RMSE": RMSE,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "roc_auc": lambda y_pred, y: roc_auc_score(y, y_pred[:,1].detach().cpu().numpy())
    # cf. scikit doc: " The binary case expects a shape (n_samples,), and the scores
    # must be the scores of the class with the greater label."
}
