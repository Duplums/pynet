import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class SAE(nn.Module):

    def __init__(self, patch_size, nb_hidden_units):
        super().__init__()

        self.encoder = nn.Linear(np.product(patch_size), nb_hidden_units)
        self.decoder = nn.Linear(nb_hidden_units, np.product(patch_size))

    def forward(self, x):

        shape_x = x.shape
        x = torch.flatten(x, start_dim=1)
        encoded = torch.sigmoid(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded)).view(shape_x)

        return encoded, decoded


class SAE_Loss:
    def __init__(self, rho, n_hidden, lambda_=0.1, device='cpu'):
        self.rho = torch.FloatTensor([rho for _ in range(n_hidden)]).unsqueeze(0).to(device)
        self.mse = nn.MSELoss()
        self.lambda_ = lambda_

    def kl_divergence(p, q):
        '''
        args:
            2 tensors `p` and `q`
        returns:
            kl divergence between the softmax of `p` and `q` with the hypothesis that p and q follows Bernouilli
            distributions
        '''
        p = F.softmax(p)
        q = F.softmax(q)
        epsilon = 1e-8
        s1 = torch.sum(p * torch.log((p+epsilon) / (q+epsilon)))
        s2 = torch.sum((1 - p) * torch.log((1 - p + epsilon) / (1 - q + epsilon)))

        return s1 + s2

    def __call__(self, input, target):
        encoded, decoded = input
        rho_hat = torch.sum(encoded, dim=0, keepdim=True)
        sparsity_penalty = SAE_Loss.kl_divergence(rho_hat, self.rho)
        mse = self.mse(decoded, target)

        return mse + self.lambda_ * sparsity_penalty



class PayanNet(nn.Module):

    def __init__(self, num_out, input_size, patch_size, pretrained_weights_path):
        super().__init__()

        (c, h, w, d) = input_size
        model = torch.load(pretrained_weights_path)["model"]
        self.kernel_weights, self.bias = model['encoder']['weights'], model['encoder']['bias']

        nb_bases = len(self.kernel_weights)
        self.patch_size = patch_size
        assert self.bias.shape == (nb_bases,)
        # Reshape the kernel weights to match the conv kernel size
        self.kernel_weights = self.kernel_weights.view((nb_bases, c) + patch_size)

        self.kernel_weights.requires_grad = False
        self.bias.requires_grad = False
        self.hidden = nn.Linear((h-patch_size[0]+1)//patch_size[0]*(w-patch_size[1]+1)//patch_size[1]*
                                (d-patch_size[2]+1)//patch_size[2]*nb_bases, 800)
        self.regressor = nn.Linear(800, num_out)

    def forward(self, x):
        x = F.conv3d(x, self.kernel_weights, bias=self.bias)
        x = torch.sigmoid(x) # we retrieve the encoded input of the SAE
        x = F.max_pool3d(x, self.patch_size)
        x = torch.sigmoid(self.hidden(x))
        x = self.regressor(x)

        return x

class PayanLikeNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        # ResNet-like 1st layer
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=3)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1)
        self.batchnorm2 = nn.BatchNorm3d(128)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1)
        self.batchnorm3 = nn.BatchNorm3d(256)
        self.adaptive_maxpool = nn.AdaptiveMaxPool3d(5)
        self.dropout = nn.Dropout(0.5)
        self.hidden_layer1 = nn.Linear(256 * 5**3, 256)
        self.hidden_layer2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)

        x = self.adaptive_maxpool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.hidden_layer1(x))
        x = self.dropout(x)
        x = self.hidden_layer2(x)

        return x.squeeze()







