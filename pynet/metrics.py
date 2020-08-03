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
import logging
import torch
import numpy as np
import torch.nn.functional as func
import sklearn.metrics as sk_metrics
from pynet.utils import Metrics
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

# Global parameters
logger = logging.getLogger("pynet")

def get_confusion_matrix(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return confusion_matrix(y, y_pred.detach().cpu().numpy())

@Metrics.register
def accuracy(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    return sk_metrics.accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

def roc_auc(y_pred, y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
        return roc_auc_score(y, y_pred[:,1].detach().cpu().numpy())
    elif len(y_pred.shape) < 2:
        return roc_auc_score(y, y_pred.detach().cpu().numpy())
    else:
        raise ValueError('Invalid shape for y_pred: {}'.format(y_pred.shape))

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


@Metrics.register
def multiclass_dice(y_pred, y):
    """ Extension of the dice to a n classes problem.
    """
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    y_pred = func.softmax(y_pred, dim=1)
    dice = 0.
    n_classes = y_pred.shape[1]
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
@Metrics.register
def pearson_correlation(y_pred, y):
    """ Pearson correlation.
    """
    mean_ypred = torch.mean(y_pred)
    mean_y = torch.mean(y)
    ypredm = y_pred.sub(mean_ypred)
    ym = y.sub(mean_y)
    r_num = ypredm.dot(ym)
    r_den = torch.norm(ypredm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


class BinaryClassificationMetrics(object):
    """ Computes and stores the average and current value.
    """
    def __init__(self, score, thr=0.5, with_logit=True):
        self.thr = 0.5
        self.score = score
        self.with_logit = with_logit

    def __call__(self, y_pred, y):
        if self.with_logit:
            y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y = y.view(-1)
        logger.debug("  prediction: {0}".format(
            y_pred.detach().numpy().tolist()))
        pred = (y_pred >= self.thr).type(torch.int32)
        truth = (y >= self.thr).type(torch.int32)
        logger.debug("  class prediction: {0}".format(
            pred.detach().numpy().tolist()))
        logger.debug("  truth: {0}".format(truth.detach().numpy().tolist()))
        metrics = {}
        tp = pred.mul(truth).sum(0).float()
        tn = (1 - pred).mul(1 - truth).sum(0).float()
        fp = pred.mul(1 - truth).sum(0).float()
        fn = (1 - pred).mul(truth).sum(0).float()
        acc = (tp + tn).sum() / (tp + tn + fp + fn).sum()
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
        metrics = {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "accuracy": acc,
            "precision": pre,
            "recall": rec
        }
        return metrics[self.score]


class SKMetrics(object):
    """ Wraping arounf scikit-learn metrics.
    """
    def __init__(self, name, thr=0.5, with_logit=True, **kwargs):
        self.name = name
        self.thr = thr
        self.kwargs = kwargs
        self.with_logit = with_logit
        if name in ("false_discovery_rate", "false_negative_rate",
                    "false_positive_rate", "negative_predictive_value",
                    "positive_predictive_value", "true_negative_rate",
                    "true_positive_rate", "accuracy"):
            self.metric = getattr(sk_metrics, "confusion_matrix")
        else:
            self.metric = getattr(sk_metrics, name)

    def __call__(self, y_pred, y):
        if self.with_logit:
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.view(-1, 1)
            y = y.view(-1, 1)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()
        if self.name not in ("roc_auc_score", "average_precision_score",
                             "log_loss", "brier_score_loss"):
            y_pred = (y_pred > self.thr).astype(int)
        metric = self.metric(y, y_pred, **self.kwargs)
        if self.name in ("false_discovery_rate", "false_negative_rate",
                         "false_positive_rate", "negative_predictive_value",
                         "positive_predictive_value", "true_negative_rate",
                         "true_positive_rate", "accuracy"):
            tn, fp, fn, tp = metric.ravel()
        if self.name == "false_discovery_rate":
            metric = fp / (tp + fp)
        elif self.name == "false_negative_rate":
            metric = fn / (tp + fn)
        elif self.name == "false_positive_rate":
            metric = fp / (fp + tn)
        elif self.name == "negative_predictive_value":
            metric = tn / (tn + fn)
        elif self.name == "positive_predictive_value":
            metric = tp / (tp + fp)
        elif self.name == "true_negative_rate":
            metric = tn / (tn + fp)
        elif self.name == "true_positive_rate":
            metric = tp / (tp + fn)
        elif self.name == "accuracy":
            metric = (tp + tn) / (tp + fp + fn + tn)
        return metric


for name in ("accuracy", "true_positive", "true_negative", "false_positive",
             "false_negative", "precision", "recall"):
    Metrics.register(
        BinaryClassificationMetrics(name), name="binary_{0}".format(name))

for name in ("accuracy", "average_precision_score", "cohen_kappa_score",
             "roc_auc_score", "log_loss", "matthews_corrcoef",
             "precision_score", "false_discovery_rate", "false_negative_rate",
             "false_positive_rate", "negative_predictive_value",
             "positive_predictive_value", "true_negative_rate",
             "true_positive_rate"):
    Metrics.register(
        SKMetrics(name), name="sk_{0}".format(name))

Metrics.register(SKMetrics("fbeta_score", beta=1), name="f1_score")
Metrics.register(SKMetrics("fbeta_score", beta=2), name="f2_score")
