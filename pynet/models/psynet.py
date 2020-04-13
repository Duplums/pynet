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
    def __init__(self, latent_dim=1000, num_classes=2, lr=0.0002, device='cuda',
                 path_to_file=None):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Defines the hyperparameters
        self.lambda_gm = 0.1 # regularization for the GM loss (associated to the likelihood)
        self.lambda_gm_u = 0.1 # regularization term associated to the general latent space
        self.lambda_reg = 0.1

        # Defines the nb of generators, disc, code disc iterations per global iteration
        self.gm_iter = 1
        # self.g_iter = 1
        # self.d_iter = 1
        # self.cd_iter = 1

        # Instantiates the Encoders E_u and E_s

        self.E_s = resnet34(in_channels=1, num_classes=self.latent_dim) # Specific encoder
        self.E_u = Discriminator(out_class=self.latent_dim, is_dis=False) # Common encoder
        #self.E_s = Discriminator(out_class=self.latent_dim, is_dis=False) # Specific encoder

        self.GM = LGMLoss(self.num_classes, self.latent_dim, 0.5)
        #self.GM_s = LGMLoss(self.num_classes, self.latent_dim, 0.5)
        
        # Instantiates the optimizers

        self.eu_optimizer = optim.Adam(self.E_u.parameters(), lr=lr)
        self.es_optimizer = optim.Adam(self.E_s.parameters(), lr=lr)
        self.gm_optimizer = optim.Adam(self.GM.parameters(), lr=lr)


        # Instantiates the losses

        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()
        self.SSIM = SSIM(window_size=8, device=self.device)


        self.last_losses = dict()
        self.data_encoded = list()
        self.metadata = list()
        self.path_to_file = path_to_file

    def set_requires_grad(self, **kwargs):
        for (net, requires_grad) in kwargs.items():
            for p in getattr(self, net).parameters():
                p.requires_grad = requires_grad

    def __call__(self, gen_load):

        ###############################################
        # Train Classifier
        ###############################################
        self.set_requires_grad(E_u=True, E_s=True)
        outputs = []
        for iters in range(self.gm_iter):
            self.E_s.zero_grad()
            self.E_u.zero_grad()
            try:
                dataitem = gen_load.__next__()[0]
                self.real_images = dataitem.inputs.to(self.device)
                targets = dataitem.labels.to(self.device)
            except StopIteration:
                if not self.training and self.data_encoded is not None and self.path_to_file is not None:
                    np.save(self.path_to_file, np.array(self.data_encoded))
                    np.save(os.path.join(os.path.dirname(self.path_to_file), "labels_encoded.npy"),
                            np.array(self.metadata))
                return True, outputs
            batch_size = self.real_images.size(0)

            z_u = self.E_u(self.real_images).view(batch_size, -1)
            z_s = self.E_s(self.real_images).view(batch_size, -1)
            z = z_u + z_s

            logits, mlogits, likelihood = self.GM(z, targets)
            gm_loss = self.cross_entropy(mlogits, targets) + self.lambda_gm * likelihood
            b_acc = balanced_accuracy(mlogits, targets)
            sensitivity_val = sensitivity(mlogits, targets)
            outputs.extend(mlogits)

            likelihood_u = LGMLoss.compute_likelihood(self.GM.centers[0,:], self.GM.log_covs[0,:], z_u)
            mask_pos = (targets == 1)

            #log_diff = torch.log(torch.clamp(1-torch.exp(self.GM.log_covs[0,:]-self.GM.log_covs[1,:]), min=1e-8,max=1)) + self.GM.log_covs[1,:]
            #likelihood_s = LGMLoss.compute_likelihood(self.GM.centers[1,:]-self.GM.centers[0,:], log_diff, z_s[mask_pos])

            total_loss = gm_loss #+ self.lambda_gm * likelihood_u + self.lambda_gm * likelihood_s
            print(total_loss, flush=True)
            if self.training:
                total_loss.backward()
                self.gm_optimizer.step()
                self.eu_optimizer.step()
                self.es_optimizer.step()
            else:
                self.data_encoded.extend(z_s.detach().cpu().numpy())
                self.metadata.extend(targets.detach().cpu().numpy())
            self.last_losses.update(total_loss=total_loss.detach().cpu().numpy(),
                                    likelihood_u=likelihood_u.detach().cpu().numpy(),
                                    #L2_reg=L2_reg_cov_s.detach().cpu().numpy(),
                                    #gm_s_loss=gm_s_loss.detach().cpu().numpy(),
                                    #gm_loss=gm_loss.detach().cpu().numpy(),
                                    balanced_accuracy=b_acc,
                                    sensitivity=sensitivity_val)
        return False, torch.stack(outputs)


    def get_aux_losses(self):
        return self.last_losses

    def get_current_visuals(self):
        return None
        # return tensor2im(torch.cat([self.real_images[:1],
        #                             self.x_hat[:1],
        #                             self.x_rand[:1]], dim=0))


    @staticmethod
    def calc_gradient_penalty(model, x, x_gen, w=10):
        """WGAN-GP gradient penalty"""
        assert x.size() == x_gen.size(), "real and sampled sizes do not match"
        alpha_size = tuple((len(x), *(1,) * (x.dim() - 1)))
        alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
        alpha = alpha_t(*alpha_size).uniform_()
        x_hat = x.data * alpha + x_gen.data * (1 - alpha)
        x_hat.requires_grad = True

        def eps_norm(x, _eps=1e-16):
            x = x.view(len(x), -1)
            return (x * x + _eps).sum(-1).sqrt()

        def bi_penalty(x):
            return (x - 1) ** 2

        grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

        penalty = w * bi_penalty(eps_norm(grad_xhat)).mean()
        return penalty


# ***********************************************
# Encoder and Discriminator have the same architecture
# ***********************************************
class Discriminator(nn.Module):
    def __init__(self, channel=512, out_class=1, is_dis=True, with_pred=False, out_pred=1):
        super(Discriminator, self).__init__()
        self.is_dis = is_dis
        self.channel = channel
        self.with_pred = with_pred
        n_class = out_class

        self.conv1 = nn.Conv3d(1, channel // 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel // 16, channel // 8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel // 8)
        self.conv3 = nn.Conv3d(channel // 8, channel // 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel // 4)
        self.conv4 = nn.Conv3d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel // 2)
        self.conv5 = nn.Conv3d(channel // 2, channel, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(channel)
        self.conv6 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)

    def forward(self, x, hs=None, return_hidden_out=False, _return_activations=False):
        if hs is None: hs = 0
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)
        h6 = self.conv6(h5)
        output = h6 + hs
        if return_hidden_out:
            return output, h6
        return output

