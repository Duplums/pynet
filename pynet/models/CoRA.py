import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2, bias=False)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, inChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 16, kernel_size=5, padding=2, bias=False)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2, bias=False)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()

        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2, bias=False)
        self.relu1 = ELUCons(elu, outChans)
        self.bn1 = ContBatchNorm3d(outChans)

        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        #skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        # print(out.shape, skipxdo.shape)
        xcat = out
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2, bias=False)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, outChans, kernel_size=1, bias=True)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        return out


class CoRA(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, inChannels, outChannels, elu=True, with_labels=False):
        super(CoRA, self).__init__()
        self.with_labels = with_labels
        # Encoder 1
        self.in_tr = InputTransition(inChannels, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)

        # Decoder 1
        self.up_tr256 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(128, 64, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(64, 32, 1, elu)
        self.up_tr32 = UpTransition(32, 16, 1, elu)
        self.out_tr = OutputTransition(16, outChannels, elu)

        if self.with_labels:
            # Decoder 2
            self.up_tr256_2 = UpTransition(256, 128, 2, elu, dropout=True)
            self.up_tr128_2 = UpTransition(128, 64, 2, elu, dropout=True)
            self.up_tr64_2 = UpTransition(64, 32, 1, elu)
            self.up_tr32_2 = UpTransition(32, 16, 1, elu)
            self.out_tr_2 = OutputTransition(16, outChannels, elu)

        self.rec = nn.MSELoss()

    def forward(self, x, label=None):
        self.input = x # dim == (inChannels, H, W, D)
        self.out16 = self.in_tr(x) # dim == (16, H, W, D)
        self.out32 = self.down_tr32(self.out16) # (32, H//2, W//2, D//2)
        self.out64 = self.down_tr64(self.out32) # (64, H//4, W//4, D//4)
        self.out128 = self.down_tr128(self.out64) # (128, H//8, W//8, D//8)

        out256 = self.down_tr256(self.out128) # (256, H//16, W//16, D//16)



        self.up128 = self.up_tr256(out256, self.out128) # (128, H//8, W//8, D//8)
        self.up64 = self.up_tr128(self.up128, self.out64) # (64, H//4, W//4, D//4)
        self.up32 = self.up_tr64(self.up64, self.out32) # (32, H//2, W//2, D//2)
        self.up16 = self.up_tr32(self.up32, self.out16) # (16, H, W, D)
        self.out = self.out_tr(self.up16) # (outChannnels, H, W, D)

        return self.out

    def rec_loss(self, *args, **kwargs):
        return self.rec(self.up128, self.out128) + self.rec(self.up64, self.out64) + self.rec(self.up32, self.out32) + \
               self.rec(self.up16, self.out16) + self.rec(self.input, self.out)



