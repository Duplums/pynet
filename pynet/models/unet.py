# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
The U-Net is a convolutional encoder-decoder neural network.
"""

# Imports
import ast
import collections
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as func
from pynet.utils import tensor2im

class UNet(nn.Module):
    """ UNet.

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:

    - padding is used in 3x3x3 convolutions to prevent loss
      of border pixels
    - merging outputs does not require cropping due to (1)
    - residual connections can be used by specifying
      UNet(merge_mode='add')
    - if non-parametric upsampling is used in the decoder
      pathway (specified by upmode='upsample'), then an
      additional 1x1x1 3d convolution occurs after upsampling
      to reduce channel dimensionality by a factor of 2.
      This channel halving happens with the convolution in
      the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=1, depth=5, 
                 start_filts=16, up_mode="transpose", down_mode="maxpool",
                 merge_mode="concat", batchnorm=False, dim="3d",
                 skip_connections=False, mode="seg", input_size=None,
                 nb_regressors=None, freeze_encoder=False):
        """ Init class.

        Parameters
        ----------
        num_classes: int
            the number of features in the output segmentation map.
        in_channels: int, default 1
            number of channels in the input tensor.
        depth: int, default 5
            number of layers in the U-Net.
        start_filts: int, default 16
            number of convolutional filters for the first conv.
        up_mode: string, default 'transpose'
            type of upconvolution. Choices: 'transpose' for transpose
            convolution, 'upsample' for nearest neighbour upsampling
        down_mode: string, default 'maxpool'
            Choices: 'maxpool' for maxpool, 'conv' for convolutions with stride=2
        merge_mode: str, default 'concat', can be 'add' or None
            the skip connections merging strategy.
        skip_connections: bool, whether we add skip connections between conv layers or not
        batchnorm: bool, default False
            normalize the inputs of the activation function.
        mode: 'str', default 'seg'
            Whether the network is turned in 'segmentation' mode ("seg") or 'classification' mode ("classif") or both
            ("seg_classif")
            The input_size is required for classification
        input_size: tuple (optional) it is required for classification only. It should be a tuple (C, H, W, D) (for 3d)
                    or (C, H, W) (for 2d)
        dim: str, default '3d'
            '3d' or '2d' input data.
        """
        # Inheritance
        super(UNet, self).__init__()

        # Check inputs
        if dim in ("2d", "3d"):
            self.dim = dim
        else:
            raise ValueError(
                "'{}' is not a valid mode for merging up and down paths. Only "
                "'3d' and '2d' are allowed.".format(dim))
        if mode in ("seg", "classif", "seg_classif"):
            self.mode = mode
        else:
            raise ValueError("'{}' is not a valid mode. Should be in 'seg' "
                             "or 'classif' mode.".format(mode))
        if up_mode in ("transpose", "upsample"):
            self.up_mode = up_mode
        else:
            raise ValueError(
                "'{}' is not a valid mode for upsampling. Only 'transpose' "
                "and 'upsample' are allowed.".format(up_mode))
        if merge_mode in ("concat", "add", None):
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                "'{}' is not a valid mode for merging up and down paths. Only "
                "'concat' and 'add' are allowed.".format(up_mode))
        if down_mode in ("maxpool", "conv"):
            self.down_mode = down_mode
        else:
            raise ValueError(
                "'{}' is not a valid mode for down sampling. Only 'maxpool' "
                "and 'conv' are allowed".format(down_mode)
            )
        if self.up_mode == "upsample" and self.merge_mode == "add":
            raise ValueError(
                "up_mode 'upsample' is incompatible with merge_mode 'add' at "
                "the moment because it doesn't make sense to use nearest "
                "neighbour to reduce depth channels (by half).")

        # Declare class parameters
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.input_size = input_size
        self.nb_regressors = nb_regressors
        self.depth = depth
        self.down = []
        self.up = [] # Useful in seg mode
        self.classifier = None # Useful in classif mode
        self.freeze_encoder = freeze_encoder
        self.name = "UNet_D%i_%s" % (self.depth, self.mode)

        # Create the encoder pathway
        for cnt in range(depth):
            in_channels = self.in_channels if cnt == 0 else out_channels
            out_channels = self.start_filts * (2**cnt)
            down_sampling = False if cnt == 0 else True
            self.down.append(
                Down(in_channels, out_channels, self.dim, down_mode=self.down_mode,
                     pooling=down_sampling, batchnorm=batchnorm,
                     skip_connections=skip_connections))

        # Freeze all the layers if necessary
        if self.freeze_encoder:
            for down_m in self.down:
                for param in down_m.parameters():
                    param.requires_grad = False

        if self.mode == "seg" or self.mode == "seg_classif":
            # Create the decoder pathway
            # - careful! decoding only requires depth-1 blocks
            for cnt in range(depth - 1):
                in_channels = out_channels
                out_channels = in_channels // 2
                self.up.append(
                    Up(in_channels, out_channels, up_mode=up_mode, dim=self.dim,
                       merge_mode=merge_mode, batchnorm=batchnorm,
                       skip_connections=skip_connections))

        if self.mode == "classif" or self.mode == "seg_classif":
            final_dims = np.array(self.input_size[1:]) // 2 ** (self.depth - 1)
            final_dims=np.insert(final_dims, 0, self.start_filts * 2**(self.depth-1)) # insert the channels

            self.classifier = Classifier(self.nb_regressors, input_size=final_dims,
                                         dim=self.dim, batchnorm=batchnorm)

        # Add the list of modules to current module
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

        # Get ouptut segmentation
        if self.mode == "seg" or self.mode == "seg_classif":
            self.conv_final = Conv1x1x1(out_channels, self.num_classes, self.dim)

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

    def forward(self, x):
        encoder_outs = []
        for module in self.down:
            x = module(x)
            encoder_outs.append(x)
        x_enc = x
        if self.mode == "seg" or self.mode == "seg_classif":
            encoder_outs = encoder_outs[:-1][::-1]
            for cnt, module in enumerate(self.up):
                x_up = encoder_outs[cnt]
                x = module(x, x_up)
            # No softmax is used. This means you need to use
            # nn.CrossEntropyLoss in your training script,
            # as this module includes a softmax already.
            x_seg = self.conv_final(x)
        if self.mode == "classif" or self.mode == "seg_classif":
            # No softmax used
            x_classif = self.classifier(x_enc)

        if self.mode == "seg":
            return x_seg
        if self.mode == "classif":
            return x_classif

        return [x_seg, x_classif]


class Classifier(nn.Module):
    def __init__(self, nb_regressors, input_size, dim, batchnorm=True):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.num_classes = nb_regressors
        self.batchnorm = batchnorm
        self.dim = dim
        self.fc1 = nn.Linear(128*5*6*5, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, self.num_classes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((5, 6, 5))

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return x.squeeze()


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size=3, stride=1,
                 padding=1, bias=True, batchnorm=True):
        super(DoubleConv, self).__init__()
        self.batchnorm = batchnorm
        self.conv1 = eval(
                    "nn.Conv{0}(in_channels, out_channels, kernel_size, "
                    "stride=stride, padding=padding, bias=bias)".format(dim))
        self.conv2 = eval(
                    "nn.Conv{0}(out_channels, out_channels, kernel_size, "
                    "stride=stride, padding=padding, bias=bias)".format(dim))
        self.relu = nn.ReLU(inplace=True)
        if batchnorm:
            self.norm1 = eval(
                    "nn.BatchNorm{0}(out_channels)".format(dim))
            self.norm2 = eval(
                    "nn.BatchNorm{0}(out_channels)".format(dim))

    def forward(self, x):
        x = self.conv1(x)
        if self.batchnorm:
            x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.batchnorm:
            x = self.norm2(x)
        x = self.relu(x)

        return x


def UpConv(in_channels, out_channels, dim, mode="transpose"):
    if mode == "transpose":
        return eval(
            "nn.ConvTranspose{0}(in_channels, out_channels, kernel_size=2, "
            "stride=2)".format(dim)) 
    elif mode == "upsample":
        # out_channels is always going to be the same as in_channels
        return nn.Sequential(collections.OrderedDict([
            ("up", nn.Upsample(mode="nearest", scale_factor=2)),
            ("conv1x", Conv1x1x1(in_channels, out_channels, dim))]))
    # else:
    #     return eval(
    #         "nn.MaxUnpool{0}(2)".format(dim)
    #     )

def Conv1x1x1(in_channels, out_channels, dim, groups=1):
    return eval(
        "nn.Conv{0}(in_channels, out_channels, kernel_size=1, groups=groups, "
        "stride=1)".format(dim))


class Down(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation and optionally a BatchNorm follows each convolution.
    """
    def __init__(self, in_channels, out_channels, dim, pooling=True,
                 down_mode='maxpool', batchnorm=True, skip_connections=False):
        super(Down, self).__init__()

        self.pooling = pooling
        self.down_mode = down_mode
        self.skip_connections = skip_connections
        if self.down_mode == "maxpool":
            self.maxpool = eval("nn.MaxPool{0}(2)".format(dim))
            self.doubleconv = DoubleConv(in_channels, out_channels, dim, batchnorm=batchnorm)
        else:
            self.downconv = eval("nn.Conv{0}(in_channels, out_channels, kernel_size=2, stride=2)".format(dim))
            if self.pooling:
                self.doubleconv = DoubleConv(out_channels, out_channels, dim, batchnorm=batchnorm)
            else:
                self.doubleconv = DoubleConv(in_channels, out_channels, dim, batchnorm=batchnorm)

    def forward(self, x):
        if self.down_mode == "maxpool":
            if self.pooling:
                x = self.maxpool(x)
            x = self.doubleconv(x)
        else:
            if self.pooling:
                x_down = self.downconv(x)
                x = self.doubleconv(x_down)
                if self.skip_connections:
                    x = x + x_down
            else:
                x = self.doubleconv(x)
        return x


class Up(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation and optionally a BatchNorm follows each convolution.
    """
    def __init__(self, in_channels, out_channels, dim, merge_mode="concat",
                 up_mode="transpose", batchnorm=True, skip_connections=False):
        super(Up, self).__init__()
        self.merge_mode = merge_mode
        self.skip_connections = skip_connections
        self.up_mode = up_mode
        self.upconv = UpConv(in_channels, out_channels, dim, mode=up_mode)
        if self.merge_mode == "concat":
            self.doubleconv = DoubleConv(in_channels, out_channels, dim, batchnorm=batchnorm)
        else:
            self.doubleconv = DoubleConv(out_channels, out_channels, dim, batchnorm=batchnorm)

    def forward(self, x_down, x_up):
        x_down = self.upconv(x_down)

        if self.merge_mode == "concat":
            x = torch.cat((x_up, x_down), dim=1)
        elif self.merge_mode == "add":
            x = x_up + x_down
        else:
            x = x_down
        x = self.doubleconv(x)

        if self.skip_connections:
            x = x + x_down
        return x
