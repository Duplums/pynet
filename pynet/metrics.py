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


class NCross_Entropy:

    def __init__(self, l_nb_classes, **cross_entropy_kwargs):
        self.l_nb_classes = l_nb_classes
        self.losses = []
        self.last_computed_losses = []
        for _ in self.l_nb_classes:
            loss = nn.CrossEntropyLoss(**cross_entropy_kwargs)
            self.losses.append(loss)
            self.last_computed_losses.append(0)


    def __call__(self, inputs, targets):
        out = 0
        start = 0
        for i, nb_classes in enumerate(self.l_nb_classes):
            loss = self.losses[i](inputs[:, start:start+nb_classes], targets[:, start:start+nb_classes])
            self.last_computed_losses[i] = float(loss)
            start += nb_classes
            out += loss

        return out

    def log_errors(self, history, epoch, it):
        for i, l in enumerate(self.last_computed_losses):
            history.log((epoch, it), **{"cross_entropy_%i" % i: l})

class MultiTaskLoss:

    def __init__(self,l_losses, weights=None, net=None, reg=None, lambda_reg=1e-3, l_metrics=None):

        self.losses = l_losses
        self.metrics = l_metrics
        self.reg = reg
        self.lambda_reg = lambda_reg
        self.net = net
        self.weights = weights or [1 for _ in l_losses]
        self.last_computed_losses = [0 for _ in l_losses]

    def __call__(self, inputs, targets):
        out = 0
        for i, loss in enumerate(self.losses):
            loss_ = loss(inputs[:, i], targets[:, i])
            self.last_computed_losses[i] = float(loss_)
            out += self.weights[i] * loss_
        if self.reg == "l2":
            l2_reg = 0
            for param in self.net.get_reg_params():
                l2_reg += torch.norm(param)
            out += self.lambda_reg * l2_reg

        return out

    def log_errors(self, history, epoch, it):
        for i, loss in enumerate(self.losses):
            history.log((epoch, it), **{"%s component %i" % (type(loss).__name__, i): self.last_computed_losses[i]})

    def log_metrics(self, inputs, targets, history, epoch, it):
        for i, metric in enumerate(self.metrics):
            history.log((epoch, it), **{"%s component %i" % (type(metric).__name__, i): metric(inputs, targets)})


class L12Loss:

    def __init__(self, reduction='mean', alpha=1, beta=0.5):
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        self.alpha = alpha
        self.beta = beta


    def __call__(self, inputs, targets):

        return self.alpha * self.l1_loss(inputs, targets) + self.beta * self.l2_loss(inputs, targets)




def accuracy(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    accuracy = y_pred.eq(y).sum().cpu().numpy() / y.size()[0]
    return accuracy

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
    y_pred = func.softmax(y_pred, dim=1)
    dice = 0.
    n_classes = y.shape[1]
    for cnt in range(n_classes):
        dice += _dice(y_pred[:, cnt], y[:, cnt])
    return dice / n_classes


def dice_loss(y_pred, y):
    """ Loss based on the dice: scales between [0, 1], optimized when
    minimized.
    """
    return 1 - multiclass_dice(y_pred, y)


METRICS = {
    "accuracy": accuracy,
    "multiclass_dice": multiclass_dice,
    "RMSE": RMSE
}
