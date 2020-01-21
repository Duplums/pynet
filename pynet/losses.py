# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Third party import
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import numpy as np

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


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, dim="3d"):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = SSIM.create_window(window_size, self.channel, dim)
        self.dim = dim

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def create_window(window_size, channel, dim):
        _1D_window = SSIM.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        if dim == "2d":
            window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        else:
            _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).\
                float().unsqueeze(0).unsqueeze(0)
            window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
        return window

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True, dim="3d"):
        if dim == "3d":
            Conv = nn.functional.conv3d
        else:
            Conv = nn.functional.conv2d

        mu1 = Conv(img1, window, padding=window_size // 2, groups=channel)
        mu2 = Conv(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = Conv(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = Conv(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = Conv(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)



    def forward(self, img1, img2):
        if self.dim == "3d":
            (_, channel, _, _, _) = img1.size()
        else:
            (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = SSIM.create_window(self.window_size, channel, self.dim)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return SSIM._ssim(img1, img2, window, self.window_size, channel, self.size_average, self.dim)



class SemiSupervisedLoss:

    def __init__(self):
        self.loss_un = nn.MSELoss()
        self.loss_age = nn.L1Loss()
        self.loss_sex = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.last_computed_losses = dict()

    def sex_accuracy(self, logit, y_true):
        return float((logit > 0.5).eq(y_true).sum().cpu().numpy()/y_true.size()[0])


    def __call__(self, inputs, targets):
        # [in, ["age", "sex"]]
        self.last_computed_losses = {
            'reconstruction': self.loss_un(inputs[0], targets[0]),
            'age': self.loss_age(inputs[1][:,0], targets[1][:,0]),
            'sex': self.loss_sex(inputs[1][:,1], targets[1][:,1]),
            'accuracy': self.sex_accuracy(self.sigmoid(inputs[1][:,1]), targets[1][:,1])
        }

        return self.last_computed_losses['reconstruction'] * 10 + 0.1*self.last_computed_losses['age'] + \
               self.last_computed_losses['sex']

    def get_aux_losses(self):
        return self.last_computed_losses

class MultiTaskLoss:

    def __init__(self,l_losses, weights=None, net=None, reg=None, lambda_reg=1e-3, l_metrics=None):

        self.losses = l_losses
        self.metrics = l_metrics
        self.reg = reg
        self.lambda_reg = lambda_reg
        self.net = net
        self.weights = weights or [1 for _ in l_losses]
        self.last_computed_losses = {"%s component %i" % (type(l).__name__, i): 0 for i,l in enumerate(l_losses)}

    def __call__(self, inputs, targets):
        out = 0
        for i, loss in enumerate(self.losses):
            loss_ = loss(inputs[i], targets[i])
            name_loss = "%s component %i" % (type(loss).__name__, i)
            self.last_computed_losses[name_loss] = float(loss_)
            out += self.weights[i] * loss_
        if self.reg == "l2":
            l2_reg = 0
            for param in self.net.get_reg_params():
                l2_reg += torch.norm(param)
            out += self.lambda_reg * l2_reg

        return out

    def get_aux_losses(self):
        return self.last_computed_losses

class RMSELoss:

    def __call__(self, inputs, targets):
        return torch.sqrt(torch.mean((inputs - targets)**2))


class L12Loss:

    def __init__(self, reduction='mean', alpha=1, beta=0.5):
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        self.alpha = alpha
        self.beta = beta


    def __call__(self, inputs, targets):
        return self.alpha * self.l1_loss(inputs, targets) + self.beta * self.l2_loss(inputs, targets)


def dice_loss_1(logits, true, eps=1e-7):
    """ Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    true: a tensor of shape [B, 1, H, W].
    logits: a tensor of shape [B, C, H, W]. Corresponds to
    the raw output or logits of the model.
    eps: added to the denominator for numerical stability.
    dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def dice_loss_2(output, target, weights=1):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    """
    output = func.softmax(output, dim=1)
    target = torch.argmax(target, dim=1).type(torch.LongTensor)
    encoded_target = output.data.clone().zero_()
    encoded_target.scatter_(1, target.unsqueeze(1), 1)
    encoded_target = Variable(encoded_target)

    assert output.size() == encoded_target.size(), "Input sizes must be equal."
    assert output.dim() == 4, "Input must be a 4D Tensor."

    num = (output * encoded_target).sum(dim=3).sum(dim=2)
    den1 = output.pow(2).sum(dim=3).sum(dim=2)
    den2 = encoded_target.pow(2).sum(dim=3).sum(dim=2)

    dice = (2 * num / (den1 + den2)) * weights
    return dice.sum() / dice.size(0)


class MultiDiceLoss(object):
    """ Define a multy classes dice loss.

    Note that PyTorch optimizers minimize a loss. In this case, we would like
    to maximize the dice loss so we return the negated dice loss.
    """
    def __init__(self, weight=None, ignore_index=None, nb_batch=None):
        """ Class instanciation.

        Parameters
        ----------
        weight: FloatTensor (C), default None
             a manual rescaling weight given to each class.
        ignore_index: int, default None
            specifies a target value that is ignored and does not contribute
            to the input gradient.
        nb_batch: int, default None
            the number of mini batch to rescale loss between 0 and 1.
        """
        self.weight = weight or 1
        self.ignore_index = ignore_index
        self.nb_batch = nb_batch or 1

    def __call__(self, output, target):
        """ Compute the loss.

        Note that this criterion is performing nn.Softmax() on the model
        outputs.

        Parameters
        ----------
        output: Variable (NxCxHxW)
            unnormalized scores for each class (the model output) where C is
            the number of classes.
        target: LongTensor (NxCxHxW)
            the class indices.
        """
        eps = 1  # 0.0001
        n_classes = output.size(1) * self.nb_batch

        output = func.softmax(output, dim=1)
        target = torch.argmax(target, dim=1).type(torch.LongTensor)
        # output = output.exp()

        encoded_target = output.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1) + eps
        denominator = output + encoded_target
        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = self.weight * (1 - (numerator / denominator))
        print(loss_per_channel)

        return loss_per_channel.sum() / n_classes


class SoftDiceLoss(_Loss):
    """ Soft Dice Loss.
    """
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = (torch.sum(torch.mul(y_pred, y_pred)) +
                 torch.sum(torch.mul(y_true, y_true)) + eps)
        dice = 2 * intersection / union
        dice_loss = 1 - dice
        return dice_loss


class CustomKLLoss(_Loss):
    """ KL Loss.
    """
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return (torch.mean(torch.mul(mean, mean)) +
                torch.mean(torch.mul(std, std)) -
                torch.mean(torch.log(torch.mul(std, std))) - 1)


class CombinedLoss(_Loss):
    """ Combined Loss.

    Diceloss + k1 * L2loss + k2 * KLloss
    Since the output of the segmentation decoder has N channels (prediction
    for each tumor subregion), we simply add the N dice loss functions.
    A hyper-parameter weight of k1=0.1, k2=0.1 was found empirically in the
    paper.
    """
    def __init__(self, num_classes, k1=0.1, k2=0.1):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def forward(self, outputs, y_true):
        y_pred, y_mid = outputs
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred = y_pred[:, :self.num_classes]
        seg_truth = y_true[:, :self.num_classes]
        vae_pred = y_pred[:, self.num_classes:]
        vae_truth = y_true[:, self.num_classes:]
        dice_loss = None
        for idx in range(self.num_classes):
            if dice_loss is None:
                dice_loss = self.dice_loss(
                    seg_pred[:, idx], seg_truth[:, idx])
            else:
                dice_loss += self.dice_loss(
                    seg_pred[:, idx], seg_truth[:, idx])
        l2_loss = self.l2_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        print("dice_loss:%.4f, L2_loss:%.4f, KL_div:%.4f, combined_loss:"
              "%.4f" % (dice_loss, l2_loss, kl_div, combined_loss))
        return combined_loss
