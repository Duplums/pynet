import numpy as np
import torch
import pickle
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from pynet.utils import tensor2im
from pynet.losses import SSIM
from pynet.metrics import Sobel3D

class Alpha_WGAN_Predictors(nn.Module):
    def __init__(self, latent_dim=1000):
        super().__init__()

        # Instantiates the Encoder and Predictor
        self.PD = nn.Linear(latent_dim, 1) # useful for predictors on the latent space
        self.E = Discriminator(out_class=latent_dim, is_dis=False)

    def forward(self, x):
        z = self.E(x)
        pred = self.PD(z.squeeze())
        return pred


class Alpha_WGAN(nn.Module):
    def __init__(self, latent_dim=1000, lr=0.0002, use_kl=False, device='cuda', path_to_file=None):
        super().__init__()
        self.std_optim=False
        self.device = device
        self.path_to_file = path_to_file
        self.latent_dim = latent_dim
        # Defines the nb of generators iterations per discriminator/Code discriminator iteration.
        self.g_iter = 2
        self.d_iter = 1
        self.cd_iter = 1
        self.use_kl = use_kl

        # Hyperparameters
        self.lambda_VAE = 1
        self.lambda_prop = 0
        self.lambda_l1 = 10
        self.lambda_d = 1

        # Instantiates the Generator, Discriminator, Code Discrimnator (z_r vs z_hat), Encoder

        self.G = Generator(noise=latent_dim)
        self.CD = Code_Discriminator(code_size=latent_dim, num_units=128) # not used if use_kl == True
        self.PD = nn.Linear(latent_dim, 1) # useful for predictors on the latent space
        self.D = Discriminator(is_dis=True)
        self.E = Discriminator(out_class=latent_dim, is_dis=False)

        # Instantiates the optimizers

        self.g_optimizer = optim.Adam(self.G.parameters(), lr=lr, weight_decay=1e-5)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=lr, weight_decay=1e-5)
        self.e_optimizer = optim.Adam(self.E.parameters(), lr=lr, weight_decay=1e-5)
        self.cd_optimizer = optim.Adam(self.CD.parameters(), lr=lr, weight_decay=1e-5)
        self.pd_optimizer = optim.Adam(self.PD.parameters(), lr=lr, weight_decay=1e-5)
        
        optimizers = ['g_optimizer', 'd_optimizer', 'e_optimizer', 'cd_optimizer', 'pd_optimizer']
        self.schedulers = {optim : torch.optim.lr_scheduler.StepLR(getattr(self, optim), 1, gamma=0.9) for optim in optimizers}

        self.logvar = torch.zeros((1, self.latent_dim), device=self.device)

        # Instantiates the losses

        self.criterion_bce = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()
        self.SSIM = SSIM(window_size=8, device=self.device)

        self.data_encoded = dict(data_enc=[], y_true=[], y_pred=[])

    def set_requires_grad(self, **kwargs):
        for (net, requires_grad) in kwargs.items():
            for p in getattr(self, net).parameters():
                p.requires_grad = requires_grad

    def __call__(self, gen_load, pbar=None, visualizer=None, **kwargs):
        stop_it = False
        total_loss = 0 # does not make sense for this model (no unique loss but 3 different losses)
        values = list() # the mean is computed after a complete epoch

        while not stop_it:
            if not self.training:
                try:
                    next_dataitem = gen_load.__next__()
                    if pbar is not None:
                        pbar.update()
                except StopIteration:
                    stop_it=True
                    break

                ###############################################
                # Test (or Validation) of the Model
                ###############################################
                real_images = next_dataitem.inputs.to(self.device)
                labels = next_dataitem.labels.to(self.device)
                if labels is not None:
                    self.data_encoded["y_true"].extend(labels.detach().cpu().numpy())

                x_enc = self.E(real_images).view(len(real_images), -1)
                z_hat = self.reparameterize(x_enc, self.logvar.repeat(len(x_enc), 1))
                z_rand = torch.randn((len(real_images), self.latent_dim), device=self.device)
                x_hat = self.G(z_hat)
                x_rand = self.G(z_rand)
                prop = self.PD(x_enc)

                d_real_loss = self.D(x_hat).mean()
                d_fake_loss = self.D(x_rand).mean()
                d_loss = -d_fake_loss - d_real_loss
                l1_loss = self.criterion_l1(x_hat, real_images)
                ssim_loss = - self.SSIM(x_hat, real_images)
                inv_idx = torch.arange(x_hat.size(0)-1, -1, -1).long()
                ssim_loss_intra = - self.SSIM(x_hat[inv_idx], x_hat)
                if self.use_kl:
                    c_loss = 0.5 * torch.mean(torch.sum(x_enc ** 2, 1))  ## Simplified version of KL when logvar == 0
                elif self.use_kl == False:
                    c_loss = -self.CD(z_hat).mean()
                elif self.use_kl == None:
                    c_loss = 0.5 * torch.mean(torch.sum(x_enc ** 2, 1)) - self.CD(z_hat).mean()
                prop_loss = self.criterion_l1(prop, labels)
                loss1 = self.lambda_l1 * l1_loss + self.lambda_VAE*c_loss + self.lambda_d * d_loss + \
                        self.lambda_prop*prop_loss

                # No GP for the test (since no grad is computed)
                x_loss2 = -2 * self.D(real_images).mean() + self.D(x_hat).mean() + self.D(x_rand).mean()
                if self.use_kl:
                    loss3 = c_loss
                else:
                    loss3 = -self.CD(z_rand).mean() - c_loss

                ## Save all the losses
                values.append(dict(loss_enc_dec=loss1.detach().cpu().numpy(),
                                   loss_disc=x_loss2.detach().cpu().numpy(),
                                   loss_code_disc=loss3.detach().cpu().numpy(),
                                   loss_l1=l1_loss.detach().cpu().numpy(),
                                   loss_ssim=ssim_loss.detach().cpu().numpy(),
                                   loss_intra_ssim=ssim_loss_intra.detach().cpu().numpy(),
                                   loss_prop=prop_loss.detach().cpu().numpy()))

                ## Save the latent vectors
                self.data_encoded["data_enc"].extend(x_enc.detach().cpu().numpy())
                self.data_encoded["y_pred"].extend(prop.detach().cpu().numpy())
                continue
            else:
                current_values = dict()
                ###############################################
                # Train Encoder - Generator
                ###############################################
                self.set_requires_grad(D=False, CD=False, E=True, G=True, PD=True)
                for iters in range(self.g_iter):
                    self.g_optimizer.zero_grad()
                    self.e_optimizer.zero_grad()
                    self.pd_optimizer.zero_grad()
                    try:
                        next_loading = gen_load.__next__()
                        real_images = next_loading.inputs.to(self.device)
                        targets = next_loading.labels.to(self.device)
                        if pbar is not None:
                            pbar.update()
                    except StopIteration:
                        stop_it = True
                        break
                    _batch_size = real_images.size(0)
                    z_rand = torch.randn((_batch_size, self.latent_dim), device=self.device)
                    x_enc = self.E(real_images).view(_batch_size, -1)
                    z_hat = self.reparameterize(x_enc, self.logvar.repeat(len(x_enc), 1))

                    x_hat = self.G(z_hat)
                    x_rand = self.G(z_rand)
                    if self.use_kl:
                        c_loss = 0.5 * torch.mean(torch.sum(x_enc ** 2, 1)) ## Simplified version of KL when logvar == 0
                    elif self.use_kl == False:
                        c_loss = -self.CD(z_hat).mean()
                    elif self.use_kl == None:
                        c_loss = 0.5 * torch.mean(torch.sum(x_enc ** 2, 1)) - self.CD(z_hat).mean()

                    prop_loss = self.criterion_l1(self.PD(x_enc), targets)

                    d_real_loss = self.D(x_hat).mean()
                    d_fake_loss = self.D(x_rand).mean()
                    d_loss = -d_fake_loss - d_real_loss
                    l1_loss = self.criterion_l1(x_hat, real_images)
                    ssim_loss = - self.SSIM(x_hat, real_images)

                    loss1 = self.lambda_l1* l1_loss + self.lambda_VAE*c_loss + self.lambda_d*d_loss + \
                            self.lambda_prop*prop_loss

                    # Save these to plot them
                    self.x_hat = x_hat.detach().cpu().numpy()
                    self.x_rand = x_rand.detach().cpu().numpy()
                    self.real_images = real_images.detach().cpu().numpy()

                    loss1.backward(retain_graph=True)

                    self.pd_optimizer.step()
                    self.e_optimizer.step()
                    self.g_optimizer.step()

                    current_values['loss_enc_dec'] = loss1.detach().cpu().numpy()
                    current_values['loss_l1'] = l1_loss.detach().cpu().numpy()
                    current_values['loss_ssim'] = ssim_loss.detach().cpu().numpy()
                    current_values['loss_prop'] = prop_loss.detach().cpu().numpy()

                ###############################################
                # Train D
                ###############################################
                self.set_requires_grad(D=True, CD=False, E=False, G=False, PD=False)

                for iters in range(self.d_iter):
                    self.d_optimizer.zero_grad()

                    try:
                        real_images = gen_load.__next__().inputs.to(self.device)
                        if pbar is not None:
                            pbar.update()
                    except StopIteration:
                        stop_it = True
                        break
                    _batch_size = real_images.size(0)
                    z_rand = torch.randn((_batch_size, self.latent_dim), device=self.device)

                    x_enc = self.E(real_images).view(_batch_size, -1)
                    z_hat = self.reparameterize(x_enc, self.logvar.repeat(len(x_enc), 1))

                    x_hat = self.G(z_hat)
                    x_rand = self.G(z_rand)
                    x_loss2 = -2 * self.D(real_images).mean() + self.D(x_hat).mean() + self.D(x_rand).mean()
                    gradient_penalty_r = Alpha_WGAN.calc_gradient_penalty(self.D, real_images.data, x_rand.data)
                    gradient_penalty_h = Alpha_WGAN.calc_gradient_penalty(self.D, real_images.data, x_hat.data)
                    loss2 = x_loss2 + (gradient_penalty_r + gradient_penalty_h)
                    loss2.backward(retain_graph=True)
                    self.d_optimizer.step()

                    current_values['loss_disc'] = loss2.detach().cpu().numpy()
                    current_values['gradient_penalty_d'] = (gradient_penalty_r + gradient_penalty_h).detach().cpu().numpy()

                ###############################################
                # Train CD
                ###############################################
                if not stop_it:
                    if self.use_kl==False or self.use_kl==None:
                        self.set_requires_grad(D=False, CD=True, E=False, G=False, PD=False)

                        for iters in range(self.cd_iter):
                            self.cd_optimizer.zero_grad()
                            z_rand = torch.randn((_batch_size, self.latent_dim), device=self.device)
                            c_loss = -self.CD(z_hat).mean()
                            gradient_penalty_cd = Alpha_WGAN.calc_gradient_penalty(self.CD, z_hat.data, z_rand.data)

                            loss3 = -self.CD(z_rand).mean() - self.lambda_VAE * c_loss + gradient_penalty_cd
                            loss3.backward(retain_graph=True)
                            self.cd_optimizer.step()
                    else:
                        loss3 = 0.5 * torch.mean(torch.sum(x_enc ** 2, 1))

                    current_values['loss_VAE'] = loss3.detach().cpu().numpy()

                    if len(values) % 10 == 0 and visualizer is not None:
                        visualizer.refresh_current_metrics()
                        visuals = self.get_current_visuals()
                        visualizer.display_images(visuals, ncols=3, middle_slices=True)

                    values.append(current_values)

                    if self.g_iter > 0 and self.d_iter > 0:
                        print('\nloss_enc_dec: {}, loss_disc: {}'.format(float(loss1), float(loss2)), flush=True)
        
        for scheduler in self.schedulers.values():
            scheduler.step()

        if self.data_encoded is not None and self.path_to_file is not None and not self.training:
            with open(self.path_to_file, 'wb') as f:
                pickle.dump(self.data_encoded, f)

        values_to_return = dict()
        for l in values.copy():
            for (k, v) in l.items():
                values_to_return.setdefault(k, []).append(v)
        values = {k: np.mean(v) for (k, v) in values_to_return.items()}

        if self.training:
            return total_loss, values
        else:
            return total_loss, values, [], [], []

    def reparameterize(self, mu, logvar):
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

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
            rec_ssim = 0
            it = 0
            x_to_save = [[], [], []]
            pbar = tqdm(total=len(test_gen), desc="SSIM-Test samples")
            for dataitem in test_gen:
                pbar.update()
                x_true = dataitem.inputs.to(self.device)
                x_enc = self.E(x_true).view(len(x_true), -1)
                z_hat = self.reparameterize(x_enc, torch.zeros_like(x_enc))
                x_rec = self.G(z_hat)
                if save_img and len(x_to_save[0]) < 50:
                    x_to_save[0].extend(x_rec.detach().cpu().numpy())
                    x_to_save[1].extend(x_true.detach().cpu().numpy())
                    x_to_save[2].extend(x_enc.detach().cpu().numpy())
                inter_ssim += float(ssim(x_true, x_rec))
                real_ssim += float(ssim(x_true[:batch_size//2], x_true[batch_size//2:]))
                rec_ssim += float(ssim(x_rec[:batch_size//2], x_rec[batch_size//2:]))
                it += 1
            if save_img:
                np.save(os.path.join(saving_dir or '', 'rec_img_alpha_wgan.npy'), np.array(x_to_save[0]))
                np.save(os.path.join(saving_dir or '', 'true_img_alpha_wgan.npy'), np.array(x_to_save[1]))
                np.save(os.path.join(saving_dir or '', 'enc_img_alpha_wgan.npy'), np.array(x_to_save[2]))
            return inter_ssim/max(it, 1), real_ssim/max(it, 1), rec_ssim/max(it, 1)


    def get_current_visuals(self):
        return tensor2im(np.concatenate([self.real_images[:1],
                                         self.x_hat[:1],
                                         self.x_rand[:1]], axis=0))

    @staticmethod
    def calc_gradient_penalty(model, x, x_gen, w=10):
        """WGAN-GP gradient penalty"""
        assert x.size() == x_gen.size(), "real and sampled sizes do not match"
        alpha_size = tuple((len(x), *(1,) * (x.dim() - 1)))
        alpha = torch.rand(*alpha_size, device='cuda' if x.is_cuda else 'cpu')
        x_hat = x.data * alpha + x_gen.data * (1 - alpha)
        x_hat.requires_grad = True

        def eps_norm(x, _eps=1e-16):
            x = x.view(len(x), -1)
            return (x * x + _eps).sum(-1).sqrt()

        def bi_penalty(x):
            return (x) ** 2

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
        self.l1 = nn.Sequential(nn.Linear(code_size, num_units),
                                nn.LeakyReLU(0.2))
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.LeakyReLU(0.2))
        self.l3 = nn.Linear(num_units, 1)

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = h3

        return output.squeeze()


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