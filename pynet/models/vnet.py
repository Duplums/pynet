import torch.nn as nn
import torch
import numpy as np

class VNet(nn.Module):


    def __init__(self, nb_classes, in_channels=1, depth=5,
                 start_filters=16, batchnorm=True, mode="AE", input_size=None):
        assert mode in ['AE', 'classifier'], "Unknown mode selected, currently supported are: 'AE' and 'classifier'"
        if mode == 'classifier' and (input_size is None or len(input_size) != 3):
            raise ValueError('The input size must be set as HxWxD')

        super(VNet, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.nb_classes = nb_classes
        self.in_channels = in_channels
        self.start_filters = start_filters

        if self.mode == "AE":
            self.up = []
        self.down = []

        nconvs = [min(cnt+1, 3) for cnt in range(depth)] # Nb of convs in each Down module

        # Create the encoder pathway
        for cnt in range(depth):
            in_channels = self.in_channels if cnt == 0 else out_channels
            out_channels = self.start_filters * (2 ** cnt)
            dconv = False if cnt == 0 else True # apply a down conv ?
            self.down.append(
                Down(in_channels, out_channels,
                     nconv=nconvs[cnt], dconv=dconv,
                     batchnorm=batchnorm))

        if self.mode == "AE":
            # Create the decoder pathway
            # - careful! decoding only requires depth-1 blocks
            for cnt in range(depth - 1):
                in_channels = out_channels
                out_channels = in_channels // 2
                self.up.append(
                    Up(in_channels, out_channels,
                       nconv=nconvs[-1-cnt],
                       batchnorm=batchnorm))

        # Add the list of modules to current module
        self.down = nn.ModuleList(self.down)
        if self.mode == "AE":
            self.up = nn.ModuleList(self.up)

        # Get ouptut segmentation
        if self.mode == "AE":
            self.final_layer = nn.Conv3d(out_channels, self.nb_classes, kernel_size=1, groups=1, stride=1)
        else: # Classification
            (h, w, d) = np.array(self.input_size) // 2**(depth -1)
            self.final_layer = nn.Sequential(
            nn.Linear(out_channels*h*w*d, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, self.nb_classes))

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
        encoder_outs = encoder_outs[:-1][::-1]
        for cnt, module in enumerate(self.up):
            x_up = encoder_outs[cnt]
            x = module(x, x_up)

        x = self.final_layer(x)
        return x

class LUConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=5, stride=1, padding=0,
                 batchnorm=True, bias=True, mode="conv"):
        super(LUConv, self).__init__()
        if mode == "conv": # Usual Conv
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        elif mode == "transpose": # UpConv
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        if batchnorm:
            self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(True)
        self.ops = nn.Sequential(self.conv, self.bn, self.relu)

    def forward(self, x):
        x = self.ops(x)
        return x


class NConvs(nn.Module):
    def __init__(self, in_channels, out_channels, nconv=3,
                 kernel_size=5, stride=1, padding=0,
                 batchnorm=True, bias=True, mode="conv"):
        super(NConvs, self).__init__()

        self.ops = nn.Sequential(LUConv(in_channels, out_channels, kernel_size, stride, padding, batchnorm, bias, mode),
                                 *[LUConv(out_channels, out_channels, kernel_size, stride, padding, batchnorm, bias, mode)
                                  for _ in range(nconv-1)])

    def forward(self, x):
        x = self.ops(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, nconv=3, dconv=True, batchnorm=True):
        super(Down, self).__init__()

        self.dconv = dconv
        self.in_channels = in_channels
        if dconv:
            self.down_conv = NConvs(in_channels, out_channels, 1, kernel_size=2, stride=2, batchnorm=batchnorm)
            self.nconvs = NConvs(out_channels, out_channels, nconv, kernel_size=5, stride=1,
                                 padding=2, batchnorm=batchnorm)
        else:
            self.nconvs = NConvs(in_channels, out_channels, nconv,
                                 kernel_size=5, stride=1, padding=2, batchnorm=batchnorm)

    def forward(self, x):
        if self.dconv:
            x_down = self.down_conv(x)
        else:
            x_down = x
        x_out = self.nconvs(x_down)
        # Add the input in order to learn only the residual
        if self.in_channels == 1 or self.dconv:
            x = x_out + x_down
        else:
            x = x_out
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, nconv=3, batchnorm=True):
        super(Up, self).__init__()

        self.up_conv = NConvs(in_channels, out_channels, 1, kernel_size=2, stride=2, batchnorm=batchnorm, mode="transpose")
        self.nconvs = NConvs(in_channels, out_channels, nconv, kernel_size=5, stride=1, padding=2, batchnorm=batchnorm)

    def forward(self, x_down, x_up):
        x_down = self.up_conv(x_down)
        xcat = torch.cat((x_up, x_down), dim=1)
        x = self.nconvs(xcat)
        x = x + x_down
        return x




