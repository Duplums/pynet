import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from pynet.utils import tensor2im
from pynet.losses import SSIM


class Alpha_WGAN(nn.Module):
    def __init__(self, latent_dim=1000, lr=0.0002, device='cuda', path_to_file="data_encoded.npy"):
        super().__init__()
        self.device = device
        self.path_to_file = path_to_file
        self.latent_dim = latent_dim
        # Defines the nb of generators iterations per discriminator/Code discriminator iteration.
        self.g_iter = 2
        self.d_iter = 1
        self.cd_iter = 1

        # Instantiates the Generator, Discriminator, Code Discrimnator (z_r vs z_hat), Encoder

        self.G = Generator(noise=latent_dim)
        self.CD = Code_Discriminator(code_size=latent_dim, num_units=4096)
        self.D = Discriminator(is_dis=True)
        self.E = Discriminator(out_class=latent_dim, is_dis=False)

        # Instantiates the optimizers

        self.g_optimizer = optim.Adam(self.G.parameters(), lr=lr)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=lr)
        self.e_optimizer = optim.Adam(self.E.parameters(), lr=lr)
        self.cd_optimizer = optim.Adam(self.CD.parameters(), lr=lr)

        # Instantiates the losses

        self.criterion_bce = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()
        self.SSIM = SSIM(window_size=8, device=self.device)

        self.last_losses = dict()
        self.data_encoded = list()
        self.metadata = list()


    def set_requires_grad(self, **kwargs):
        for (net, requires_grad) in kwargs.items():
            for p in getattr(self, net).parameters():
                p.requires_grad = requires_grad

    def __call__(self, gen_load):
        if not self.training:
            ## Just encodes the data and saves it
            try:
                next_dataitem = gen_load.__next__()[0]
            except StopIteration:
                if self.data_encoded is not None:
                    np.save(self.path_to_file, np.array(self.data_encoded))
                    np.save(os.path.join(os.path.dirname(self.path_to_file), "labels_alpha_wgan.npy"),
                            np.array(self.metadata))
                return True, []
            real_images = next_dataitem.inputs.to(self.device)
            labels = next_dataitem.labels
            z_hat = self.E(real_images).view(real_images.size(0), -1)
            self.data_encoded.extend(z_hat.detach().cpu().numpy())
            if labels is not None:
                self.metadata.extend(labels.detach().cpu().numpy())
            return False, []

        ###############################################
        # Train Encoder - Generator
        ###############################################
        self.set_requires_grad(D=False, CD=False, E=True, G=True)
        for iters in range(self.g_iter):
            self.G.zero_grad()
            self.E.zero_grad()
            try:
                self.real_images = gen_load.__next__()[0].inputs.to(self.device)
            except StopIteration:
                return True,  []
            _batch_size = self.real_images.size(0)
            z_rand = torch.randn((_batch_size, self.latent_dim), device=self.device)
            z_hat = self.E(self.real_images).view(_batch_size, -1)
            self.x_hat = self.G(z_hat)
            self.x_rand = self.G(z_rand)
            c_loss = -self.CD(z_hat).mean()

            d_real_loss = self.D(self.x_hat).mean()
            d_fake_loss = self.D(self.x_rand).mean()
            d_loss = -d_fake_loss - d_real_loss
            l1_loss = 10 * self.criterion_l1(self.x_hat, self.real_images)
            ssim_loss = - self.SSIM(self.x_hat, self.real_images)
            loss1 = l1_loss + c_loss + d_loss #+ ssim_loss

            if iters < self.g_iter - 1:
                loss1.backward()
            else:
                loss1.backward(retain_graph=True)
            self.e_optimizer.step()
            self.g_optimizer.step()
            self.g_optimizer.step()

        ###############################################
        # Train D
        ###############################################
        self.set_requires_grad(D=True, CD=False, E=False, G=False)

        for iters in range(self.d_iter):
            self.d_optimizer.zero_grad()
            try:
                real_images = gen_load.__next__()[0].inputs.to(self.device)
            except StopIteration:
                return True, []
            _batch_size = real_images.size(0)
            z_rand = torch.randn((_batch_size, self.latent_dim), device=self.device)
            z_hat = self.E(real_images).view(_batch_size, -1)
            x_hat = self.G(z_hat)
            x_rand = self.G(z_rand)
            x_loss2 = -2 * self.D(real_images).mean() + self.D(x_hat).mean() + self.D(x_rand).mean()
            gradient_penalty_r = Alpha_WGAN.calc_gradient_penalty(self.D, real_images.data, x_rand.data)
            gradient_penalty_h = Alpha_WGAN.calc_gradient_penalty(self.D, real_images.data, x_hat.data)
            loss2 = x_loss2 + gradient_penalty_r + gradient_penalty_h
            loss2.backward(retain_graph=True)
            self.d_optimizer.step()

        ###############################################
        # Train CD
        ###############################################
        self.set_requires_grad(D=False, CD=True, E=False, G=False)

        for iters in range(self.cd_iter):
            self.cd_optimizer.zero_grad()
            z_rand = torch.randn((_batch_size, self.latent_dim), device=self.device)
            gradient_penalty_cd = Alpha_WGAN.calc_gradient_penalty(self.CD, z_hat.data, z_rand.data)

            loss3 = -self.CD(z_rand).mean() - c_loss + gradient_penalty_cd

            loss3.backward(retain_graph=True)
            self.cd_optimizer.step()
        
        self.last_losses.update(loss_enc_dec=loss1.detach().cpu().numpy(),
                                loss_disc=loss2.detach().cpu().numpy(),
                                loss_code_disc=loss3.detach().cpu().numpy(),
                                loss_l1=l1_loss.detach().cpu().numpy(),
                                loss_ssim=ssim_loss.detach().cpu().numpy())
        print("")
        print(self.last_losses, flush=True)
        return False, []

    def compute_MS_SSIM(self, test_gen=None, N=100, batch_size=8, save_img=False, saving_dir=None):
        assert batch_size % 2 == 0, "batch_size must be an even number"

        from tqdm import tqdm
        # If test_gen is None, computes the MS-SSIM for N random samples,
        # otherwise, computes the MS-SSIM for the true test samples and the reconstructed ones
        ssim = SSIM(window_size=8)

        x_to_save = []
        if test_gen is None:
            ms_ssim = 0
            pbar = tqdm(total=2*(N//batch_size), desc="SSIM-Generated samples")
            for k in range(2):
                for n in range(N//batch_size):
                    pbar.update()
                    z_rand = torch.randn((batch_size, self.latent_dim), device=self.device)
                    x_rand = self.G(z_rand)
                    if save_img and len(x_to_save) < 50:
                        x_to_save.extend(x_rand.detach().cpu().numpy())
                    ms_ssim += float(ssim(x_rand[:batch_size//2], x_rand[batch_size//2:]))
            if save_img:
                np.save(os.path.join(saving_dir or '', 'rand_gen_alpha_wgan.npy'), np.array(x_to_save))
            return ms_ssim/max((2*N//batch_size), 1)
        else:
            inter_ssim = 0
            real_ssim = 0
            it = 0
            x_to_save = [[], []]
            pbar = tqdm(total=len(test_gen), desc="SSIM-Test samples")
            for dataitem in test_gen:
                pbar.update()
                x_true = dataitem.inputs.to(self.device)
                x_rec = self.G(self.E(x_true).view(len(x_true), -1))
                if save_img and len(x_to_save[0]) < 50:
                    x_to_save[0].extend(x_rec.detach().cpu().numpy())
                    x_to_save[1].extend(x_true.detach().cpu().numpy())
                inter_ssim += float(ssim(x_true, x_rec))
                real_ssim += float(ssim(x_true[:batch_size//2], x_true[batch_size//2:]))
                it += 1
            if save_img:
                np.save(os.path.join(saving_dir or '', 'rec_img_alpha_wgan.npy'), np.array(x_to_save[0]))
                np.save(os.path.join(saving_dir or '', 'true_img_alpha_wgan.npy'), np.array(x_to_save[1]))
            return inter_ssim/max(it, 1), real_ssim/max(it, 1)


    def get_aux_losses(self):
        return self.last_losses

    def get_current_visuals(self):
        return tensor2im(torch.cat([self.real_images[:1],
                                    self.x_hat[:1],
                                    self.x_rand[:1]], dim=0))

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
        if self.with_pred:
            self.conv4b = nn.Conv3d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1)
            self.bn4b = nn.BatchNorm3d(channel // 2)
            self.conv5b = nn.Conv3d(channel // 2, channel, kernel_size=4, stride=2, padding=1)
            self.bn5b = nn.BatchNorm3d(channel)
            self.conv6b = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)

        self.conv4 = nn.Conv3d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel // 2)
        self.conv5 = nn.Conv3d(channel // 2, channel, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(channel)
        self.conv6 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)

    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)
        h6 = self.conv6(h5)
        output = h6

        if self.with_pred:
            h4_b = F.leaky_relu(self.bn4b(self.conv4b(h3)), negative_slope=0.2)
            h5_b = F.leaky_relu(self.bn5b(self.conv5b(h4_b)), negative_slope=0.2)
            h6_b = self.conv6(h5_b)
            output = h6, h6_b

        return output


class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100, num_units=750):
        super(Code_Discriminator, self).__init__()
        n_class = 1
        self.l1 = nn.Sequential(nn.Linear(code_size, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2, inplace=True))
        self.l3 = nn.Linear(num_units, 1)

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = h3

        return output


class Generator(nn.Module):
    def __init__(self, noise: int = 100, channel: int = 32):
        super(Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.noise = noise
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c * 16, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(_c * 16)

        self.tp_conv2 = nn.Conv3d(_c * 16, _c * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c * 8)

        self.tp_conv3 = nn.Conv3d(_c * 8, _c * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(_c * 4)

        self.tp_conv4 = nn.Conv3d(_c * 4, _c * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(_c * 2)

        self.tp_conv5 = nn.Conv3d(_c * 2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm3d(_c)

        self.tp_conv6 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, noise):
        noise = noise.view(-1, self.noise, 1, 1, 1)
        h = self.tp_conv1(noise)
        h = self.relu(self.bn1(h))

        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv5(h)
        h = self.relu(self.bn5(h))

        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv6(h)

        h = F.tanh(h)

        return h