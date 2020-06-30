import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from pynet.metrics import balanced_accuracy, sensitivity, get_confusion_matrix
from pynet.models.resnet import resnet34
from pynet.utils import tensor2im
from pynet.losses import SSIM, LGMLoss


class PsyNet(nn.Module):
    def __init__(self, alpha_wgan, num_classes=2, lr=0.0002, device='cuda',
                 path_to_file=None):
        super().__init__()
        self.std_optim=False
        self.device = device
        self.latent_dim = alpha_wgan.latent_dim
        self.num_classes = num_classes

        # Defines the hyperparameters
        self.lambda_gm = 0.1 # regularization for the GM loss (associated to the likelihood)
        self.E = alpha_wgan.E # Universal encoder

        # Defines the loss
        self.loss = LGMLoss(self.num_classes, self.latent_dim, 0.5)
        self.cross_entropy = nn.CrossEntropyLoss()
        # Instantiates the optimizers

        self.E_optimizer = optim.Adam(self.E.parameters(), lr=lr)
        self.gm_optimizer = optim.Adam(self.loss.parameters(), lr=lr)
        self.scheduler_E = torch.optim.lr_scheduler.StepLR(self.E_optimizer, step_size=1, gamma=0.9)
        self.scheduler_GM = torch.optim.lr_scheduler.StepLR(self.gm_optimizer, step_size=1, gamma=0.9)

        self.last_losses = dict()
        self.data_encoded = list()
        self.metadata = list()
        self.path_to_file = path_to_file

    def set_requires_grad(self, **kwargs):
        for (net, requires_grad) in kwargs.items():
            for p in getattr(self, net).parameters():
                p.requires_grad = requires_grad


    def __call__(self, gen_load, pbar=None, **kwargs):

        stop_it = False
        total_loss = 0  # does not make sense for this model (no unique loss but 3 different losses)
        values = list()  # the mean is computed after a complete epoch
        y, y_true, X = [], [], []
        while not stop_it:
            try:
                next_dataitem = gen_load.__next__()[0]
                real_images = next_dataitem.inputs.to(self.device)
                labels = next_dataitem.labels.to(self.device).long()
                if pbar is not None:
                    pbar.update()
            except StopIteration:
                break

            x_enc = self.E(real_images)
            logits, mlogits, likelihood = self.loss(x_enc.squeeze(), labels)
            classif_loss =  self.cross_entropy(mlogits, labels)
            gm_loss = classif_loss + self.lambda_gm * likelihood
            
            if not self.training:
                y.extend(mlogits.detach().cpu().numpy())
                y_true.extend(labels.detach().cpu().numpy())
                X.extend(real_images.detach().cpu().numpy())

            if self.training:
                gm_loss.backward()
                self.E_optimizer.step()
                self.gm_optimizer.step()

            values.append(dict(likelihood=likelihood.detach().cpu().numpy(),
                               gm_loss=gm_loss.detach().cpu().numpy(),
                               cross_entropy=classif_loss.detach().cpu().numpy()))
            total_loss += gm_loss.detach().cpu().numpy()

        if self.training:
            self.scheduler_E.step()
            self.scheduler_GM.step()

        values_to_return = dict()
        for l in values.copy():
            for (k, v) in l.items():
                values_to_return.setdefault(k, []).append(v)
        values = {k: np.mean(v) for (k, v) in values_to_return.items()}

        if self.training:
            return total_loss, values
        else:
            return total_loss, values, y, y_true, X



