from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from pynet.losses import SSIM
from pynet.utils import tensor2im
from torch.nn import Parameter
import numpy as np
from pynet.metrics import Sobel3D


class SchizNet(nn.Module):
    def __init__(self, in_channels, input_size, batch_size, lambda_rec=1):
        super().__init__()

        latent_dim = 128
        out_channels=256
        enc_dims = (out_channels,) + tuple(np.array(input_size)[1:]//16)

        self.encoder = EncoderSchiz(in_channels, out_channels=out_channels, input_size=input_size, latent_dim=latent_dim)
        # self.decoder = DecoderSchiz(input_channels=out_channels, output_channels=in_channels,
        #                             in_dim=enc_dims, latent_dim=latent_dim)

        self.classifier = Disc(2, np.product(enc_dims), 64) # (mu, sigma) for each latent dim
        # self.ones = torch.ones(batch_size//2, device='cuda')
        self.b_size = batch_size
        self.lambda_rec = lambda_rec
        self.disc_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        # x1, x2, target1, target2 = x[:self.b_size//2], x[self.b_size//2:], target[:self.b_size//2], target[self.b_size//2:]
        # self.x1 = x1
        # self.x2 = x2
        # self.x1_enc = self.encoder(self.x1)
        # self.x2_enc = self.encoder(self.x2)
        # self.x1_dec = self.decoder(self.x_enc1)
        # self.x2_dec = self.decoder(self.x2_enc)
        # self.loss_rec = self.rec_loss(self.x1_dec, self.x1) + self.rec_loss(self.x2_dec, self.x2)
        # embedded_target = 2 * (target1 == target2) - 1
        #
        # self.loss_embedded = self.cosine_embedding(torch.flatten(self.x1_enc[:,:50], 1),
        #                                            torch.flatten(self.x2_enc[:,:50], 1),
        #                                            embedded_target) + \
        #                      self.cosine_embedding(torch.flatten(self.x1_enc[:,50:], 1),
        #                                            torch.flatten(self.x2_enc[:,50:], 1),
        #                                            self.ones)
        self.x = x
        self.x_enc, self.sup1, self.sup2 = self.encoder(self.x)
        # self.x_dec = self.decoder(self.x_enc)
        self.classif = self.classifier(self.x_enc)

        return self.classif

    def get_loss(self, out, target):

        # self.age_loss = self.mse_loss(self.classif, target)
        # self.sex_loss = self.bce_loss(self.classif[:,1], target[:,1])
        self.dx_loss = self.cross_entropy(self.classif, target.long()) + \
                       0.1 * self.cross_entropy(self.sup1, target.long()) + \
                       0.1 * self.cross_entropy(self.sup2, target.long())

        # self.rec = self.mse_loss(self.x_dec, self.x)
        # self.KL = -0.5 * torch.mean(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())

        return self.dx_loss #+ 0.1 * self.KL

    def get_aux_losses(self):
        return {
                # 'loss_rec': np.mean(self.rec.detach().cpu().numpy()),
                # 'KL': np.mean(self.KL.detach().cpu().numpy())
                # 'loss_age': np.mean(self.age_loss.detach().cpu().numpy()),
                # 'loss_sex': np.mean(self.sex_loss.detach().cpu().numpy())
                'dx_loss': np.mean(self.dx_loss.detach().cpu().numpy()),
                }

    # def get_current_visuals(self):
    #     return tensor2im(torch.cat([self.x_filtered, self.x_dec]))

class Disc(nn.Module):
    def __init__(self, nb_classes, latent_dim, dim=512):
        super(Disc, self).__init__()
        self.size = latent_dim
        self.dim = dim

        self.classify = nn.Sequential(
            nn.Linear(self.size, dim), # Add the age info
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, nb_classes),
        )

    def forward(self, x):
        x = x.view(-1, self.size)
        x = self.classify(x)
        return x.squeeze(dim=1)


class EncoderSchiz(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, latent_dim, norm=nn.InstanceNorm3d):
        super().__init__()
        self.out_channels = out_channels
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []
        self.layer6 = []

        self.layer1.append(nn.Conv3d(in_channels, 32, 4, 2, 1)) # //2
        self.layer1.append(norm(32))
        self.layer1.append(nn.LeakyReLU(0.2, inplace=True))
        self.l1 = nn.Sequential(*self.layer1)

        self.layer2.append(nn.Conv3d(32, 64, 4, 2, 1)) # //2
        self.layer1.append(norm(64))
        self.layer2.append(nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(*self.layer2)

        self.layer3.append(nn.Conv3d(64, 128, 4, 2, 1)) #// 2
        self.layer3.append(norm(128))
        self.layer3.append(nn.LeakyReLU(0.2, inplace=True))
        self.l3 = nn.Sequential(*self.layer3)

        self.layer4.append(nn.Conv3d(128, 256, 4, 2, 1)) #//2
        self.layer4.append(norm(256))
        self.layer4.append(nn.LeakyReLU(0.2, inplace=True))
        self.l4 = nn.Sequential(*self.layer4)

        # self.layer5.append(nn.Conv3d(256, self.out_channels, 4, 2, 1))
        # self.layer5.append(norm(self.out_channels))
        # self.layer5.append(nn.LeakyReLU(0.2, inplace=True))
        # self.l5 = nn.Sequential(*self.layer5)
        #
        # self.layer6.append(norm(self.out_channels))
        # self.layer6.append(nn.Conv3d(self.out_channels, self.out_channels, 4, 2, 1))
        # self.layer6.append(nn.LeakyReLU(0.2, inplace=True))
        # self.l6 = nn.Sequential(*self.layer6)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        # self.fc1 = nn.Linear(out_size, latent_dim) ## mu of p(z | x) assuming p(z) ~ N(0, I)
        # self.fc2 = nn.Linear(out_size, latent_dim) ## sigma of p(z | x)
        self.fc1 = nn.Linear(self.out_channels * np.product(np.array(input_size)[1:]//16), 2)
        self.fc2 = nn.Linear(128 * np.product(np.array(input_size)[1:]//8), 2)
        # self.freeze_weights([self.l1, self.l2, self.l3, self.l4])

    def freeze_weights(self, layers):
        for l in layers:
            for p in l.parameters():
                p.require_grad=False

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        l3_supervised = self.fc2(out.view(len(x), -1))
        out = self.l4(out)
        l4_supervised = self.fc1(out.view(len(x), -1))
        # out = self.l5(out)
        # out = self.l6(out)
        # out = self.max_pool(out)
        # mu = self.fc1(out.view(x.size(0), -1))
        # logvar = self.fc2(out.view(x.size(0), -1))
        # z = self.reparameterize(mu, logvar)

        return out, l3_supervised, l4_supervised

    # Generate a random multivariate gaussian vector
    def reparameterize(self, mu, logvar):
        return torch.randn(*mu.size(), device='cuda') * torch.exp(0.5*logvar) + mu


class DecoderSchiz(nn.Module):
    def __init__(self, input_channels, output_channels, in_dim, latent_dim, norm=nn.InstanceNorm3d):
        super().__init__()
        self.input_channels = input_channels
        self.in_dim = in_dim
        # self.fc = nn.Linear(latent_dim, np.product(in_dim))
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []
        self.layer6 = []

        # self.layer1.append(nn.ConvTranspose3d(self.input_channels, 512, 4, 2, 1))
        # self.layer1.append(norm(512))
        # self.layer1.append(nn.ReLU(inplace=True))
        # self.l1 = nn.Sequential(*self.layer1)
        #
        self.layer2.append(nn.ConvTranspose3d(self.input_channels, 256, 4, 2, 1))
        self.layer2.append(norm(256))
        self.layer2.append(nn.ReLU(inplace=True))
        self.l2 = nn.Sequential(*self.layer2)
        #
        self.layer3.append(nn.ConvTranspose3d(256, 128, 4, 2, 1))
        self.layer3.append(norm(128))
        self.layer3.append(nn.ReLU(inplace=True))
        self.l3 = nn.Sequential(*self.layer3)

        self.layer4.append(nn.ConvTranspose3d(128, 64, 4, 2, 1))
        self.layer4.append(norm(64))
        self.layer4.append(nn.LeakyReLU(inplace=True))
        self.l4 = nn.Sequential(*self.layer4)

        self.layer5.append(nn.ConvTranspose3d(64, 32, 4, 2, 1))
        self.layer5.append(norm(32))
        self.layer5.append(nn.LeakyReLU(inplace=True))
        self.l5 = nn.Sequential(*self.layer5)

        self.layer6.append(nn.ConvTranspose3d(32, output_channels, 4, 2, 1))
        # self.layer6.append(nn.Tanh())
        self.l6 = nn.Sequential(*self.layer6)

        # self.freeze_weights([self.l3, self.l4, self.l5, self.l6])

    def freeze_weights(self, layers):
        for l in layers:
            for p in l.parameters():
                p.require_grad=False

    def forward(self, net):
        # out = self.fc(net)
        # out = self.l1(net)
        out = self.l2(net)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out) #net.view(net.size(0), *self.in_dim)
        return out



class DIDD(nn.Module):
    def __init__(self, in_channels, sep, input_size, device='cuda'):
        super().__init__()
        # Final output size after encoding
        self.size = np.max([np.array(input_size) // 64, [1 for _ in range(len(input_size))]], axis=0)
        self.sep = sep
        self.out_channels = 512
        self.in_channels = in_channels
        self.device=device
        self.E_common = Encoder(self.in_channels, self.out_channels -  2 * sep)
        self.E_dx = Encoder(self.in_channels, sep)
        self.E_ctl = Encoder(self.in_channels, sep)
        self.decoder = Decoder(self.in_channels, self.out_channels)
        self.discriminator = Disc(self.out_channels - 2 * sep, self.size)
        self.l1 = nn.MSELoss()
        self.bce = nn.BCELoss()

    def get_ae_params(self):
        return list(self.E_common.parameters()) + list(self.E_dx.parameters()) + list(self.E_ctl.parameters()) +\
                list(self.decoder.parameters())

    def get_disc_params(self):
        return self.discriminator.parameters()

    def forward(self, data_ctl, data_dx, with_disc=False):

        self.ctl = data_ctl
        self.dx = data_dx
        self.ctl_common = self.E_common(data_ctl)
        self.dx_common = self.E_common(data_dx)
        self.ctl_label = torch.zeros(len(data_ctl), device=self.device)
        self.dx_label = torch.ones(len(data_ctl), device=self.device)

        if not with_disc:
            self.ctl_spec = self.E_ctl(data_ctl)
            self.ctl_enc_dx = self.E_dx(data_ctl)
            self.dx_spec = self.E_dx(data_ctl)
            self.dx_enc_ctl = self.E_ctl(data_dx)

            assert self.dx_spec.shape == self.ctl_spec.shape

            self.zero_encoding = torch.zeros(self.dx_spec.shape, device=self.device)

            ctl_encoded = torch.cat([self.ctl_common, self.ctl_spec, self.zero_encoding], dim=1)
            dx_encoded = torch.cat([self.dx_common, self.zero_encoding, self.dx_spec], dim=1)

            self.ctl_decoded = self.decoder(ctl_encoded)
            self.dx_decoded = self.decoder(dx_encoded)

            return torch.cat([self.ctl_decoded, self.dx_decoded], dim=0)


    def zeros_rec_adv_loss(self, *args, lambda_rec=1, lambda_zeros=1, lambda_adv=1, **kwargs):
        self.loss_rec = self.l1(self.ctl_decoded, self.ctl) + self.l1(self.dx_decoded, self.dx)
        self.loss_zeros = self.l1(self.ctl_enc_dx, self.zero_encoding) + self.l1(self.dx_enc_ctl, self.zero_encoding)
        self.loss_adv = self.bce(self.discriminator(self.ctl_common), self.dx_label) + \
                        self.bce(self.discriminator(self.dx_common), self.dx_label)

        return lambda_rec * self.loss_rec + lambda_zeros * self.loss_zeros + lambda_adv * self.loss_adv

    def disc_loss(self, *args, **kwargs):
        self.loss_disc = self.bce(self.discriminator(self.ctl_common), self.ctl_label) + \
                         self.bce(self.discriminator(self.dx_common), self.dx_label)
        return self.loss_disc

    def get_aux_losses(self):
        return {'loss_rec': np.mean(self.loss_rec.detach().cpu().numpy()),
                'loss_zeros': np.mean(self.loss_zeros.detach().cpu().numpy()),
                'loss_adv': np.mean(self.loss_adv.detach().cpu().numpy()),
                'loss_disc': np.mean(self.loss_disc.detach().cpu().numpy())
                }

    def get_current_visuals(self):
        return tensor2im(torch.cat([self.ctl_decoded, self.dx_decoded], dim=0))



class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []
        self.layer6 = []

        self.layer1.append(SpectralNorm(nn.Conv3d(in_channels, 32, 4, 2, 1)))
        self.layer1.append(nn.BatchNorm3d(32))
        self.layer1.append(nn.LeakyReLU(0.2, inplace=True))
        self.l1 = nn.Sequential(*self.layer1)

        self.layer2.append(SpectralNorm(nn.Conv3d(32, 64, 4, 2, 1)))
        self.layer1.append(nn.BatchNorm3d(64))
        self.layer2.append(nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(*self.layer2)

        self.layer3.append(SpectralNorm(nn.Conv3d(64, 128, 4, 2, 1)))
        self.layer3.append(nn.BatchNorm3d(128))
        self.layer3.append(nn.LeakyReLU(0.2, inplace=True))
        self.l3 = nn.Sequential(*self.layer3)

        self.layer4.append(SpectralNorm(nn.Conv3d(128, 256, 4, 2, 1)))
        self.layer4.append(nn.BatchNorm3d(256))
        self.layer4.append(nn.LeakyReLU(0.2, inplace=True))
        self.l4 = nn.Sequential(*self.layer4)

        self.layer5.append(SpectralNorm(nn.Conv3d(256, self.out_channels, 4, 2, 1)))
        self.layer5.append(nn.BatchNorm3d(self.out_channels))
        self.layer5.append(nn.LeakyReLU(0.2, inplace=True))
        self.l5 = nn.Sequential(*self.layer5)

        self.layer6.append(nn.BatchNorm3d(self.out_channels))
        self.layer6.append(SpectralNorm(nn.Conv3d(self.out_channels, self.out_channels, 4, 2, 1)))
        self.layer6.append(nn.LeakyReLU(0.2, inplace=True))
        self.l6 = nn.Sequential(*self.layer6)

    def forward(self, net):
        out = self.l1(net)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)

        return out


class Decoder(nn.Module):
    def __init__(self, output_channels, input_channels):
        super(Decoder, self).__init__()
        self.input_channels = input_channels

        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []
        self.layer6 = []

        self.layer1.append(SpectralNorm(nn.ConvTranspose3d(self.input_channels, 512, 4, 2, 1)))
        self.layer1.append(nn.BatchNorm3d(512))
        self.layer1.append(nn.ReLU(inplace=True))
        self.l1 = nn.Sequential(*self.layer1)

        self.layer2.append(SpectralNorm(nn.ConvTranspose3d(512, 256, 4, 2, 1)))
        self.layer2.append(nn.BatchNorm3d(256))
        self.layer2.append(nn.ReLU(inplace=True))
        self.l2 = nn.Sequential(*self.layer2)

        self.layer3.append(SpectralNorm(nn.ConvTranspose3d(256, 128, 4, 2, 1)))
        self.layer3.append(nn.BatchNorm3d(128))
        self.layer3.append(nn.ReLU(inplace=True))
        self.l3 = nn.Sequential(*self.layer3)

        self.layer4.append(SpectralNorm(nn.ConvTranspose3d(128, 64, 4, 2, 1)))
        self.layer4.append(nn.BatchNorm3d(64))
        self.layer4.append(nn.ReLU(inplace=True))
        self.l4 = nn.Sequential(*self.layer4)

        self.layer5.append(SpectralNorm(nn.ConvTranspose3d(64, 32, 4, 2, 1)))
        self.layer5.append(nn.BatchNorm3d(32))
        self.layer5.append(nn.ReLU(inplace=True))
        self.l5 = nn.Sequential(*self.layer5)

        self.layer6.append(nn.ConvTranspose3d(32, output_channels, 4, 2, 1))
        #self.layer6.append(nn.Tanh())
        self.l6 = nn.Sequential(*self.layer6)

    def forward(self, net):
        out = self.l1(net)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        return out



def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        # if not self._made_params():
        #     self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        # self._update_u_v()
        return self.module.forward(*args)
