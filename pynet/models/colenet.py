import torch.nn as nn
import numpy as np
import torch

class ColeNet(nn.Module):

    def __init__(self, num_classes, input_size):
        super().__init__()
        # input_size == (C, H, W, D)
        self.down = []
        self.input_size = input_size
        self.name = "ColeNet"

        channels = [8, 16, 32, 64, 128]
        for i, c in enumerate(channels):
            if i == 0:
                self.down.append(ConvBlock(self.input_size[0], c))
            else:
                self.down.append(ConvBlock(channels[i-1], c))

        self.down = nn.ModuleList(self.down)
        self.classifier = Classifier(channels[-1] * np.prod(np.array(self.input_size[1:])//2**len(channels)), num_classes)
        # Kernel initializer
        # Weight initialization
        self.weight_initializer()

    def weight_initializer(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def get_reg_params(self):
        return self.classifier.parameters()

    def forward(self, x):
        for m in self.down:
            x = m(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.squeeze(x)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.pooling = nn.MaxPool3d(2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x

class Classifier(nn.Module):

    def __init__(self, num_input_features, num_classes):
        super().__init__()
        self.input_features = num_input_features
        self.num_classes = num_classes

        self.fc = nn.Linear(num_input_features, num_classes)


    def forward(self, x):
        x = self.fc(x)
        return x

