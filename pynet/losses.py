# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Third party import
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

# Global parameters
logger = logging.getLogger("pynet")

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


class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations, arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)

class SupervisedNTXenLoss(nn.Module):

    def __init__(self, supervised_loss, alpha=0.1, **kwargs):
        super().__init__()
        self.ntXenloss = NTXenLoss(**kwargs)
        self.sup = supervised_loss
        self.sup_val = 0.0
        self.alpha = alpha

    def forward(self, z_i, z_j, labels):
        sup = (self.sup(z_i[:, -1], labels) + self.sup(z_j[:, -1], labels))/2.0
        self.sup_val = sup.detach().cpu().numpy()

        if self.ntXenloss.return_logits:
            unsup, sim_zij, true_pairs = self.ntXenloss(z_i[:, :-1], z_j[:, :-1])
            return (unsup + self.alpha * sup), sim_zij, true_pairs
        else:
            unsup = self.ntXenloss(z_i[:, :-1], z_j[:, :-1])
            return unsup + self.alpha * sup

    def get_aux_losses(self):
        return {'MAE': self.sup_val}

    def __str__(self):
        return "{}(alpha={})".format(type(self).__name__, self.alpha)

class GeneralizedSupervisedNTXenLoss(nn.Module):
    def __init__(self, kernel='rbf', temperature=0.1, return_logits=False, sigma=1.0):
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)

        :param temperature:
        :param return_logits:
        """

        # sigma = prior over the label's range
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        if self.kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        else:
            assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

    def forward(self, z_i, z_j, labels):
        N = len(z_i)
        assert N == len(labels), "Unexpected labels length: %i"%len(labels)
        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        all_labels = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy() # [2N, *]
        weights = self.kernel(all_labels, all_labels) # [2N, 2N]
        weights = weights * (1 - np.eye(2*N)) # puts 0 on the diagonal
        weights /= weights.sum(axis=1)
        # if 'rbf' kernel and sigma->0, we retrieve the classical NTXenLoss (without labels)
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1), torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0) # [2N, 2N]
        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        loss = -1./N * (torch.from_numpy(weights).to(z_i.device) * log_sim_Z).sum()

        correct_pairs = torch.arange(N, device=z_i.device).long()

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss

    def __str__(self):
        return "{}(temp={}, kernel={}, sigma={})".format(type(self).__name__, self.temperature,
                                                         self.kernel.__name__, self.sigma)


class AgeSexSupervisedNTXenLoss(GeneralizedSupervisedNTXenLoss):
    """
    We assume the inputs labels y have shape [N, 2] where the fisrt component is the age
    and the second component is the sex. The 'rbf' kernel is used for age and the kernel for sex is defined as:
    k(s1, s2) = 1_{s1=s2}. The total kernel is defined as:

                k([a1, s1], [a2, s2]) = rbf(a1, a2)k(s1, s2)

    """
    def _kernel(self, y1, y2):
        """
        :param y1: np.array of shape [N, 2] (1st comp: age, 2nd comp: sex)
        :param y2: np.array of shape [N, 2] (1st comp: age, 2nd comp: sex)
        :return: np.array K of shape [N, N] where K[i,j] = rbf(y1[i,0],y2[j,0])*1_{y1[i,1]==y2[j,1]}
        """
        rbf = rbf_kernel(y1[:, 0].reshape(-1, 1), y2[:, 0].reshape(-1, 1), gamma=1. / (2 * self.sigma ** 2))
        S = np.kron(y1[:,1], np.ones((len(y1), 1))) == np.kron(y2[:,1], np.ones((len(y2), 1))).T

        return rbf * S

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel = self._kernel

class SBRLoss(nn.Module):
    """
    Refer to paper:
    Yunho Jeon, Yongseok Choi, Jaesun Park, Subin Yi, Dong-Yeon Cho, and Jiwon Kim
    Sample-based Regularization: A Transfer Learning Strategy Toward Better Generalization, arXiv 2020
    """

    def __init__(self, model, sup_loss, feature_extractor, num_classes, distance="euclidean", beta=1e-5, device='cuda'):
        super().__init__()
        assert distance in ["euclidean", "cosinus"], "Unknown distance: %s"%distance

        self.model = model
        self.output_features = None
        self.num_classes = num_classes
        self.distance = distance
        self.supervised_loss = sup_loss
        self.beta = beta
        self.device=device

        for name, layer in self.model.named_modules():
            if feature_extractor == name:
                layer.register_forward_hook(self.set_output_features)

    def set_output_features(self, module, input, output):
        self.output_features = output

    def forward(self, outputs, targets):
        assert len(self.output_features) == len(targets), "Inconsistent number of samples in the batch " \
                                                         "(%i samples from output features, %i samples from labels)"%\
                                                         (len(self.output_features), len(targets))
        sup_loss = self.supervised_loss(outputs, targets)

        b = targets.size(0)
        z = self.output_features.view(b, -1)
        if self.distance == "cosinus":
            z = func.normalize(z, p=2, dim=-1)  # dim [b, f]
            distmat = 1 - z @ z.T # dim [b, b]
        elif self.distance == "euclidean":
            distmat = z.pow(2).sum(dim=1, keepdim=True).expand(b, b).clone()
            distmat += distmat.T - 2 *  z @ z.T # dim [b, b]

        targets = targets.unsqueeze(1).unsqueeze(1).expand(b, b, self.num_classes)
        classes = torch.arange(self.num_classes).expand(b, b, self.num_classes).to(self.device)
        mask = targets.eq(classes) & targets.transpose(0,1).eq(classes.transpose(0,1)) # dim [b, b, num_classes]

        reg_loss = (distmat.unsqueeze(-1).expand(b, b, self.num_classes) * mask).sum() / (2.0 * b)

        return sup_loss + self.beta * reg_loss


class LGMLoss(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    """
    def __init__(self, num_classes, feat_dim, alpha, device='cuda'):
        # alpha in [0, 1], % degree of margin associated to d_k
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.device = device
        self.alpha = alpha

        # Defines (mu, sigma) that will be optimized during the training
        self.centers = nn.Parameter(torch.zeros((num_classes, feat_dim), device=self.device, requires_grad=True))
        self.log_covs = nn.Parameter(torch.zeros((num_classes, feat_dim), device=self.device, requires_grad=True))

    def forward(self, feat, label):
        assert feat.device == label.device == self.centers.device
        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)
        covs = torch.exp(log_covs) # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1) # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1) #eq.(13) get (d_k) for all k in [1,K]


        y_onehot = torch.zeros((batch_size, self.num_classes), device=self.device)
        y_onehot.scatter_(1, torch.unsqueeze(label.long(), dim=-1), self.alpha) # set alpha to the true position
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot) # get (1+alpha) * d_k if k is the true label, d_k otherwise

        slog_covs = torch.sum(log_covs, dim=-1)
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5*(tslog_covs + margin_dist) #eq.(17)
        logits = -0.5 * (tslog_covs + dist)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5*torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        likelihood = (1.0/batch_size) * (cdist + reg)

        return logits, margin_logits, likelihood

    @staticmethod
    def compute_likelihood(center, log_cov, feat):
        # center: [n_features]; log_cov: [n_features]; feat: [batch_size, n_features]
        batch_size = feat.shape[0]
        if batch_size == 0:
            return torch.tensor(0.0)

        covs = torch.exp(log_cov)  # [n_features]
        tcovs = covs.repeat(batch_size, 1)  # [batch_size, n_features]
        diff = feat - torch.unsqueeze(center, dim=0) # [batch_size, n_features]
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = 0.5 * torch.sum(diff) # scalar

        reg = 0.5 * torch.sum(log_cov) # scalar
        likelihood = (1.0 / batch_size) * (dist + reg) # scalar

        return likelihood


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, dim="3d", device="cuda"):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.device = device
        self.size_average = size_average
        self.channel = 1
        self.window = SSIM.create_window(window_size, self.channel, dim, device)
        self.dim = dim

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def create_window(window_size, channel, dim, device):
        _1D_window = SSIM.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        if dim == "2d":
            window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        else:
            _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).\
                float().unsqueeze(0).unsqueeze(0)
            window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
        return window.to(device)

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
            window = SSIM.create_window(self.window_size, channel, self.dim, self.device)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return SSIM._ssim(img1, img2, window, self.window_size, channel, self.size_average, self.dim)


class ConcreteDropoutLoss:
    def __init__(self, model, criterion, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        self.model = model
        self.criterion = criterion
        self._set_dropout_regularizers(weight_regularizer=weight_regularizer,
                                       dropout_regularizer=dropout_regularizer)

    def __call__(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        reg = self._get_regularization_loss()

        return loss + reg

    def _set_dropout_regularizers(self, **kwargs):
        def _set_dropout_state_in_module(module):
            if module.__class__.__name__.endswith('ConcreteDropout'):
                for (prop, val) in kwargs.items():
                    setattr(module, prop, val)
        self.model.apply(_set_dropout_state_in_module)


    def _get_regularization_loss(self):
        regularization_loss = 0.0

        def get_module_regularization_loss(module):
            nonlocal regularization_loss
            if module.__class__.__name__.endswith('ConcreteDropout'):
                regularization_loss = regularization_loss + module.regularisation()
        self.model.apply(get_module_regularization_loss)
        return regularization_loss




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
            loss_ = loss(inputs[:,i], targets[:,i])
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
from pynet.utils import Losses


@Losses.register
class FocalLoss(object):
    """ Define a Focal Loss.

    Loss(pt) = −αt mt (1−pt)γ log(pt)

    where pt is the model's estimated probability for each class.

    When an example is misclassified and pt is small, the modulating factor
    is near 1 and the loss is unaffected. As pt goes to 1, the factor goes to
    0 and the loss for well-classified examples is down-weighted.
    The focusing parameter γ smoothly adjusts the rate at which easy examples
    are down-weighted. When γ= 0, the loss is equivalent to cross entropy, and
    as γ isincreased the effect of the modulating factor is likewise increased.
    For instance, with γ= 2, an example classified with pt= 0.9 would have
    100×lower loss compared with cross entropy and with pt≈0.968 it would have
    1000×lower loss.
    Then we use an α-balanced variant of the focal loss for addressing class
    imbalance with a weighting factor α ∈ [0,1]. In practice α may be set by
    inverse class frequency.

    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, n_classes, gamma=2, alpha=None, reduction="mean",
                 with_logit=True):
        """ Class instanciation.

        Parameters
        ----------
        n_classes: int
            the number of classes.
        gamma: float, default 2
            the focusing parameter >=0.
        alpha: float or list of float, default None
            if set use alpha-balanced variant of the focal loss.
        reduction: str, default 'mean'
            specifies the reduction to apply to the output: 'none' - no
            reduction will be applied, 'mean' - the sum of the output
            will be divided by the number of elements in the output, 'sum'
            - the output will be summed.
        with_logit: bool, default True
            apply the softmax logit function to the result.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.with_logit = with_logit
        self.eps = 1e-9
        alpha = alpha or 1
        if not isinstance(alpha, list):
            alpha = [alpha] * n_classes
        if len(alpha) != n_classes:
            raise ValueError("Invalid alphas size.")
        logger.debug("  alpha: {0}".format(alpha))
        self.alpha = torch.FloatTensor(alpha).view(-1, 1)
        # self.alpha = self.alpha / self.alpha.sum()
        self.debug("alpha", self.alpha)

    def __call__(self, output, target):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤target[i]≤C−1.
        """
        logger.debug("Focal loss...")
        self.debug("output", output)
        self.debug("target", target)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        dim = output.dim()
        logger.debug("  n_batches: {0}".format(n_batch))
        logger.debug("  n_classes: {0}".format(n_classes))
        logger.debug("  dim: {0}".format(dim))
        if self.with_logit:
            output = func.softmax(output, dim=1)
        logit = output + self.eps
        self.debug("logit", logit)

        # Reshape data
        # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
        if dim > 2:
            logit = logit.view(n_batch, n_classes, -1)
            self.debug("logit", logit)
            logit = logit.permute(0, 2, 1).contiguous()
            self.debug("logit", logit)
            logit = logit.view(-1, n_classes)
            self.debug("logit", logit)
        target = torch.squeeze(target, dim=1)
        target = target.view(-1, 1)
        self.debug("target", target)

        # Create the labels one hot encoded tensor
        idx = target.data
        one_hot = torch.zeros(target.size(0), n_classes,
                              device=device, dtype=output.dtype)
        target_one_hot = one_hot.scatter_(1, idx, 1.) + self.eps

        # Compute the focal loss
        if self.alpha.device != device:
            self.alpha = self.alpha.to(device)
        pt = torch.sum(target_one_hot * logit, dim=1)
        self.debug("pt", pt)
        logpt = torch.log(pt)
        weight = torch.pow(1 - pt, self.gamma)
        self.debug("weight", weight)
        alpha = self.alpha[idx]
        alpha = torch.squeeze(alpha)
        self.debug("alpha", alpha)
        loss = -1 * alpha * weight * logpt
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss) / self.alpha[target].mean()
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def _forward_without_resizing(self, output, target):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤target[i]≤C−1.
        """
        logger.debug("Focal loss...")
        self.debug("output", output)
        self.debug("target", target)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        dim = output.dim()
        logger.debug("  n_batches: {0}".format(n_batch))
        logger.debug("  n_classes: {0}".format(n_classes))
        logger.debug("  dim: {0}".format(dim))
        if self.with_logit:
            output = func.softmax(output, dim=1)
        logit = output + self.eps
        self.debug("logit", logit)

        # Create the labels one hot encoded tensor
        one_hot = torch.zeros(n_batch, n_classes, *target.shape[1:],
                              device=device, dtype=output.dtype)
        target_one_hot = one_hot.scatter_(
            1, target.unsqueeze(1), 1.) + self.eps

        # Compute the focal loss
        if self.alpha.device != device:
            self.alpha = self.alpha.to(device)
        weight = torch.pow(1 - logit, self.gamma)
        self.debug("weight", weight)
        shape = [1, n_classes] + [1] * len(target.shape[1:])
        alpha = self.alpha.view(*shape)
        alpha = alpha.expand_as(weight)
        self.debug("alpha", alpha)
        focal = -1 * alpha * weight * torch.log(logit)
        self.debug("focal", focal)
        loss = torch.sum(target_one_hot * focal, dim=1)
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss) / self.alpha[target].mean()
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Losses.register
class MaskLoss(object):
    """ Define a Masked Loss.

    Loss(pt) = −αt mt log(pt)

    where pt is the model's estimated probability for each class.
    """
    def __init__(self, n_classes, beta=0.2, alpha=None, reduction="mean",
                 with_logit=True):
        """ Class instanciation.

        Parameters
        ----------
        n_classes: int
            the number of classes.
        beta: float, default 0.2
            the minimum value in the mask.
        alpha: float or list of float, default None
            if set use alpha-balanced variant of the focal loss.
        reduction: str, default 'mean'
            specifies the reduction to apply to the output: 'none' - no
            reduction will be applied, 'mean' - the sum of the output
            will be divided by the number of elements in the output, 'sum'
            - the output will be summed.
        with_logit: bool, default True
            apply the log softmax logit function to the result.
        """
        self.beta = beta
        self.alpha = alpha
        self.reduction = reduction
        self.with_logit = with_logit
        self.eps = 1e-9
        alpha = alpha or 1
        if not isinstance(alpha, list):
            alpha = [alpha] * n_classes
        if len(alpha) != n_classes:
            raise ValueError("Invalid alphas size.")
        logger.debug("  alpha: {0}".format(alpha))
        self.alpha = torch.FloatTensor(alpha)
        self.debug("alpha", self.alpha)

    def __call__(self, output, target, mask):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤ target[i] ≤C−1.
        mask: Tensor (N,*)
            the binary mask used to mask the loss.
        """
        logger.debug("Maked loss...")
        self.debug("output", output)
        self.debug("target", target)
        self.debug("mask", mask)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")
        if mask is not None and output.shape[0] != mask.shape[0]:
            raise ValueError("Expected pred & mask same batch size.")
        if mask is not None and output.shape[2:] != mask.shape[1:]:
            raise ValueError("Expected pred & mask same data size.")
        if mask is not None and output.device != mask.device:
            raise ValueError("Pred & mask must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        logger.debug("  n_batches: {0}".format(n_batch))
        logger.debug("  n_classes: {0}".format(n_classes))

        if self.alpha.device != device:
            self.alpha = self.alpha.to(device)
        if self.with_logit:
            output = func.log_softmax(output, dim=1)
        logit = output + self.eps
        self.debug("logit", logit)

        # Compute the focal loss
        mask[mask <= self.beta] = self.beta
        loss = func.nll_loss(logit, target, weight=self.alpha,
                             reduction="none")
        loss = loss * mask
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss) / self.alpha[target].mean()
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Losses.register
class SoftDiceLoss(object):
    """ Define a multi class Dice Loss.

    Dice = (2 intersec Y) / (X + Y)

    Note that PyTorch optimizers minimize a loss. In this case, we would like
    to maximize the dice loss so we return 1 - Dice.
    """
    def __init__(self, with_logit=True, reduction="mean"):
        """ Class instanciation.

        Parameters
        ----------
        with_logit: bool, default True
            apply the softmax logit function to the result.
        reduction: str, default 'mean'
            specifies the reduction to apply to the output: 'none' - no
            reduction will be applied, 'mean' - the sum of the output
            will be divided by the number of elements in the output, 'sum'
            - the output will be summed.
        """
        self.with_logit = with_logit
        self.reduction = reduction
        self.smooth = 1e-6
        self.eps = 1e-6

    def __call__(self, output, target):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤targets[i]≤C−1.
        """
        logger.debug("Dice loss...")
        self.debug("output", output)
        self.debug("target", target)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        if self.with_logit:
            prob = func.softmax(output, dim=1)
        else:
            prob = output
        self.debug("logit", prob)

        # Create the labels one hot encoded tensor
        prob = prob.view(n_batch, -1)
        dims = list(range(len(target.shape)))
        dims.insert(1, len(target.shape))
        dims = tuple(dims)
        logger.debug("permute {0}".format(dims))
        target_one_hot = func.one_hot(target, num_classes=n_classes)
        self.debug("target_one_hot", target_one_hot)
        target_one_hot = target_one_hot.permute(dims)
        target_one_hot = target_one_hot.contiguous().view(n_batch, -1)
        if target_one_hot.device != device:
            target_one_hot = target_one_hot.to(device)
        self.debug("target_one_hot", target_one_hot)

        # Compute the dice score
        intersection = prob * target_one_hot
        self.debug("intersection", intersection)
        dice_score = (2 * intersection.sum(dim=1) + self.smooth) / (
            target_one_hot.sum(dim=1) + prob.sum(dim=1) + self.smooth)
        loss = 1. - dice_score
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def _forward_without_resizing(self, output, target):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤targets[i]≤C−1.
        """
        logger.debug("Dice loss...")
        self.debug("output", output)
        self.debug("target", target)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        if self.with_logit:
            prob = func.softmax(output, dim=1)
        else:
            prob = output
        self.debug("logit", prob)

        # Create the labels one hot encoded tensor
        one_hot = torch.zeros(n_batch, n_classes, *target.shape[1:],
                              device=device, dtype=output.dtype)
        target_one_hot = one_hot.scatter_(1, target.unsqueeze(1), 1.)
        self.debug("one hot", target_one_hot)

        # Compute the dice score
        dims = tuple(range(1, len(target.shape) + 1))
        intersection = torch.sum(prob * target_one_hot, dims)
        self.debug("intersection", intersection)
        cardinality = torch.sum(prob + target_one_hot, dims)
        self.debug("cardinality", cardinality)
        dice_score = 2. * intersection / (cardinality + self.eps)
        loss = 1. - dice_score
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Losses.register
class CustomKLLoss(object):
    """ KL Loss.
    """
    def __call__(self, mean, std):
        return (torch.mean(torch.mul(mean, mean)) +
                torch.mean(torch.mul(std, std)) -
                torch.mean(torch.log(torch.mul(std, std))) - 1)


@Losses.register
class NvNetCombinedLoss(object):
    """ Combined Loss.

    Cross Entropy loss + k1 * L2 loss + k2 * KL loss
    Since the output of the segmentation decoder has N channels (prediction
    for each tumor subregion), we simply add the N dice loss functions.
    A hyper-parameter weight of k1=0.1, k2=0.1 was found empirically in the
    paper.
    """
    def __init__(self, num_classes, k1=0.1, k2=0.1):
        super(NvNetCombinedLoss, self).__init__()
        self.layer_outputs = None
        self.num_classes = num_classes
        self.k1 = k1
        self.k2 = k2
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.l2_loss = nn.MSELoss(reduction="mean")
        self.kl_loss = CustomKLLoss()

    def __call__(self, output, target):
        logger.debug("NvNet Combined Loss...")
        self.debug("output", output)
        self.debug("target", target)
        if self.layer_outputs is not None:
            y_mid = self.layer_outputs
            self.debug("y_mid", y_mid)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape != target.shape:
            raise ValueError("Expected pred & true of same size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")
        if self.layer_outputs is not None and y_mid.shape[-1] != 256:
            raise ValueError("128 means & stds expected.")

        device = output.device
        if self.layer_outputs is not None:
            est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
            self.debug("est_mean", est_mean)
            self.debug("est_std", est_std)
            vae_pred = output[:, self.num_classes:]
            vae_truth = target[:, self.num_classes:]
            self.debug("vae_pred", vae_pred)
            self.debug("vae_truth", vae_truth)
        seg_pred = output[:, :self.num_classes]
        seg_truth = target[:, :self.num_classes]
        self.debug("seg_pred", seg_pred)
        self.debug("seg_truth", seg_truth)
        seg_truth = torch.argmax(seg_truth, dim=1).type(torch.LongTensor)
        if seg_truth.device != device:
            seg_truth = seg_truth.to(device)
        self.debug("seg_truth", seg_truth)

        ce_loss = self.ce_loss(seg_pred, seg_truth)
        if self.layer_outputs is not None:
            l2_loss = self.l2_loss(vae_pred, vae_truth)
            kl_div = self.kl_loss(est_mean, est_std)
            combined_loss = ce_loss + self.k1 * l2_loss + self.k2 * kl_div
        else:
            l2_loss, kl_div = (None, None)
            combined_loss = ce_loss
        logger.debug(
            "ce_loss: {0}, L2_loss: {1}, KL_div: {2}, combined_loss: "
            "{3}".format(ce_loss, l2_loss, kl_div, combined_loss))
        return combined_loss

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Losses.register
class MSELoss(object):
    """ Calculate the Mean Square Error loss between I and J.
    """
    def __init__(self, concat=False):
        """ Init class.

        Parameters
        ----------
        concat: bool, default False
            if set asssume that the target image J is a concatenation of the
            moving and fixed.
        """
        super(MSELoss, self).__init__()
        self.concat = concat

    def __call__(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute MSE loss...")
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        self.debug("I", arr_i)
        self.debug("J", arr_j)
        loss = torch.mean((arr_i - arr_j) ** 2)
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Losses.register
class PCCLoss(object):
    """ Calculate the Pearson correlation coefficient between I and J.
    """
    def __init__(self, concat=False):
        """ Init class.

        Parameters
        ----------
        concat: bool, default False
            if set asssume that the target image J is a concatenation of the
            moving and fixed.
        """
        super(PCCLoss, self).__init__()
        self.concat = concat

    def __call__(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute PCC loss...")
        nb_channels = arr_j.shape[1]
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        logger.debug("  channels: {0}".format(nb_channels))
        self.debug("I", arr_i)
        self.debug("J", arr_j)
        centered_arr_i = arr_i - torch.mean(arr_i)
        centered_arr_j = arr_j - torch.mean(arr_j)
        pearson_loss = torch.sum(
            centered_arr_i * centered_arr_j) / (
                torch.sqrt(torch.sum(centered_arr_i ** 2) + 1e-6) *
                torch.sqrt(torch.sum(centered_arr_j ** 2) + 1e-6))
        loss = 1. - pearson_loss
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Losses.register
class NCCLoss(object):
    """ Calculate the normalize cross correlation between I and J.
    """
    def __init__(self, concat=False, win=None):
        """ Init class.

        Parameters
        ----------
        concat: bool, default False
            if set asssume that the target image J is a concatenation of the
            moving and fixed.
        win: list of in, default None
            the window size to compute the correlation, default 9.
        """
        super(NCCLoss, self).__init__()
        self.concat = concat
        self.win = win

    def __call__(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute NCC loss...")
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        ndims = len(list(arr_i.size())) - 2
        if ndims not in [1, 2, 3]:
            raise ValueError("Volumes should be 1 to 3 dimensions, not "
                             "{0}.".format(ndims))
        if self.win is None:
            self.win = [9] * ndims
        device = arr_i.device
        sum_filt = torch.ones([1, 1, *self.win]).to(device)
        pad_no = math.floor(self.win[0] / 2)
        stride = tuple([1] * ndims)
        padding = tuple([pad_no] * ndims)
        logger.debug("  ndims: {0}".format(ndims))
        logger.debug("  stride: {0}".format(stride))
        logger.debug("  padding: {0}".format(padding))
        logger.debug("  filt: {0} - {1}".format(
            sum_filt.shape, sum_filt.get_device()))
        logger.debug("  win: {0}".format(self.win))
        logger.debug("  I: {0} - {1} - {2}".format(
            arr_i.shape, arr_i.get_device(), arr_i.dtype))
        logger.debug("  J: {0} - {1} - {2}".format(
            arr_j.shape, arr_j.get_device(), arr_j.dtype))

        var_arr_i, var_arr_j, cross = self._compute_local_sums(
            arr_i, arr_j, sum_filt, stride, padding)
        cc = cross * cross / (var_arr_i * var_arr_j + 1e-5)
        loss = -1 * torch.mean(cc)
        logger.debug("  loss: {0}".format(loss))
        logger.info("Done.")

        return loss

    def _compute_local_sums(self, arr_i, arr_j, filt, stride, padding):
        conv_fn = getattr(func, "conv{0}d".format(len(self.win)))
        logger.debug("  conv: {0}".format(conv_fn))

        arr_i2 = arr_i * arr_i
        arr_j2 = arr_j * arr_j
        arr_ij = arr_i * arr_j

        sum_arr_i = conv_fn(arr_i, filt, stride=stride, padding=padding)
        sum_arr_j = conv_fn(arr_j, filt, stride=stride, padding=padding)
        sum_arr_i2 = conv_fn(arr_i2, filt, stride=stride, padding=padding)
        sum_arr_j2 = conv_fn(arr_j2, filt, stride=stride, padding=padding)
        sum_arr_ij = conv_fn(arr_ij, filt, stride=stride, padding=padding)

        win_size = np.prod(self.win)
        logger.debug("  win size: {0}".format(win_size))
        u_arr_i = sum_arr_i / win_size
        u_arr_j = sum_arr_j / win_size

        cross = (sum_arr_ij - u_arr_j * sum_arr_i - u_arr_i * sum_arr_j +
                 u_arr_i * u_arr_j * win_size)
        var_arr_i = (sum_arr_i2 - 2 * u_arr_i * sum_arr_i + u_arr_i *
                     u_arr_i * win_size)
        var_arr_j = (sum_arr_j2 - 2 * u_arr_j * sum_arr_j + u_arr_j *
                     u_arr_j * win_size)

        return var_arr_i, var_arr_j, cross


@Losses.register
class RCNetLoss(object):
    """ RCNet Loss function.

    This loss needs intermediate layers outputs.
    Use a callback function to set the 'layer_outputs' class parameter before
    each evaluation of the loss function.
    If you use an interface this parameter is updated automatically?

    PCCLoss
    """
    def __init__(self):
        self.similarity_loss = PCCLoss(concat=True)
        self.layer_outputs = None

    def __call__(self, moving, fixed):
        logger.debug("Compute RCNet loss...")
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        stem_results = self.layer_outputs["stem_results"]
        for stem_result in stem_results:
            params = stem_result["stem_params"]
            if params["raw_weight"] > 0:
                stem_result["raw_loss"] = self.similarity_loss(
                    stem_result["warped"], fixed) * params["raw_weight"]
        loss = sum([
            stem_result["raw_loss"] * stem_result["stem_params"]["weight"]
            for stem_result in stem_results if "raw_loss" in stem_result])
        self.layer_outputs = None
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss


@Losses.register
class VMILoss(object):
    """ Variational Mutual information loss function.

    Reference: http://bayesiandeeplearning.org/2018/papers/136.pdf -
               https://discuss.pytorch.org/t/help-with-histogram-and-loss-
               backward/44052/5
    """
    def get_positive_expectation(self, p_samples, average=True):
        log_2 = math.log(2.)
        Ep = log_2 - F.softplus(-p_samples)
        # Note JSD will be shifted
        if average:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples, average=True):
        log_2 = math.log(2.)
        Eq = F.softplus(-q_samples) + q_samples - log_2
        # Note JSD will be shifted
        if average:
            return Eq.mean()
        else:
            return Eq

    def __call__(self, lmap, gmap):
        """ The fenchel_dual_loss from the DIM code
        Reshape tensors dims to (N, Channels, chunks).

        Parameters
        ----------
        lmap: Tensor
            the moving data.
        gmap: Tensor
            the fixed data.
        """
        lmap = lmap.reshape(2, 128, -1)
        gmap = gmap.squeeze()

        N, units, n_locals = lmap.size()
        n_multis = gmap.size(2)

        # First we make the input tensors the right shape.
        l = lmap.view(N, units, n_locals)
        l = lmap.permute(0, 2, 1)
        l = lmap.reshape(-1, units)

        m = gmap.view(N, units, n_multis)
        m = gmap.permute(0, 2, 1)
        m = gmap.reshape(-1, units)

        u = torch.mm(m, l.t())
        u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

        mask = torch.eye(N).to(l.device)
        n_mask = 1 - mask

        E_pos = get_positive_expectation(u, average=False).mean(2).mean(2)
        E_neg = get_negative_expectation(u, average=False).mean(2).mean(2)

        E_pos = (E_pos * mask).sum() / mask.sum()
        E_neg = (E_neg * n_mask).sum() / n_mask.sum()
        loss = E_neg - E_pos

        return loss
