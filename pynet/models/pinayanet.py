import torch
import torch.nn as nn
import numpy as np

class SemiSupervisedAE(nn.Module):
    # Network describred in "Using deep autoencoders to identify abnormal brain structural patterns in neuropsychiatric
    # disorders: A large-scale multi-sample study", 2019, Pinaya et al.
    def __init__(self, input_size, nb_regressors):
        super().__init__()

        self.input_size = tuple(input_size)
        self.h1 = nn.Linear(np.prod(input_size), 100, bias=True)
        self.z = nn.Linear(100, 75, bias=True)
        self.h1_r = nn.Linear(75, 100, bias=True)
        self.reconstruction = nn.Linear(100, np.prod(input_size), bias=True)
        self.dropout = nn.Dropout(0.5)
        self.selu = nn.SELU(inplace=True)

        self.fc = nn.Linear(100, nb_regressors, bias=True)


    def forward(self, x):
        x = x + 1e-4 * torch.randn_like(x)
        x = torch.flatten(x, 1)
        x = self.h1(x)
        x = self.selu(x)

        x_reg = self.fc(x)

        x_enc = self.z(x)
        x_enc = self.selu(x_enc)

        x_dec = self.h1_r(x_enc)
        x_dec = self.selu(x_dec)

        x_rec = self.reconstruction(x_dec)
        x_rec = torch.reshape(x_rec, (-1,)+self.input_size)

        out = [x_rec, x_reg]

        return out


