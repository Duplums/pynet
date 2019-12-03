import torch.nn as nn
import numpy as np
import torch

class ConvNet3D(nn.Module):

    def __init__(self, num_classes, channels, fc, input_size,
                down_mode="maxpool", batchnorm=False):
        super().__init__()
        # input_size == (C, H, W, D)
        self.down = []
        self.input_size = input_size
        self.depth = len(channels)
        self.channels = channels
        self.name = "ConvNet_%s" % '_'.join([str(c) for c in channels])

        for i, c in enumerate(channels):
            if i == 0:
                self.down.append(Conv(self.input_size[0], c, down_mode, batchnorm))
            else:
                self.down.append(Conv(channels[i-1], c, down_mode, batchnorm))

        self.down = nn.ModuleList(self.down)
        self.avg_pooling = nn.AdaptiveAvgPool3d(1) # = nb of channels

        self.classifier = Classifier(channels[-1], num_classes, fc)
        # Kernel initializer
        # Weight initialization
        self.weight_initializer()

    def weight_initializer(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def visualize_data(self, visualizer):
        # if self.classifier.last_features is not None:
        #     visualizer.t_SNE(self.classifier.last_features, len(self.classifier.last_features)* [0])
        pass


    def get_reg_params(self):
        return self.classifier.parameters()

    def forward(self, x):
        for m in self.down:
            x = m(x)
        x = self.avg_pooling(x) # flatten the input but keep the 1st dim (batch size)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, down_mode="maxpool", batchnorm=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if batchnorm:
            self.batchnorm = nn.BatchNorm3d(out_channels)
        if down_mode == "maxpool":
            self.pooling = nn.MaxPool3d(2)
        elif self.down_mode == "conv":
            self.pooling = nn.Conv3d(out_channels, out_channels, 2, stride=2)

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)
        if hasattr(self, 'batchnorm'):
            x = self.batchnorm(x)
        x = self.pooling(x)
        return x

class Classifier(nn.Module):

    def __init__(self, input_dims, num_classes, fc):
        super().__init__()
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.last_features = None

        self.fcs = []

        for i, n in enumerate(fc):
            if i == 0:
                self.fcs.append(nn.Linear(np.prod(self.input_dims), n))
            else:
                self.fcs.append(nn.Linear(fc[i-1], n))
            self.fcs.append(nn.Dropout(0.5, inplace=True))
            self.fcs.append(nn.ReLU(inplace=True))

        self.fcs = nn.ModuleList(self.fcs)
        self.final_fc = nn.Linear(fc[-1], self.num_classes)

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        self.last_features = x.cpu().detach().numpy()
        x = self.final_fc(x)
        return x

