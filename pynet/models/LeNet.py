import torch
import torch.nn as nn

class LeNetLike(nn.Module):

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
