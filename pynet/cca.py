from typing import Callable
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

## code available at: https://github.com/moskomule/cca.pytorch
## This is an adaptation to the 3D case

__all__ = ["svcca_distance", "pwcca_distance", "CCAHook"]

_device = "cuda" if torch.cuda.is_available() else "cpu"

# Workaround a known torch bug...
def svd(tensor, *args, **kwargs):
    try:
        res = torch.svd(tensor, *args, **kwargs)
        res = (res.U, res.S, res.V)
    except RuntimeError:
        res = torch.svd(tensor.T, *args, **kwargs)
        res = (res.V, res.S, res.U)
    return res


def svd_reduction(tensor: torch.Tensor, accept_rate=0.99):
    left, diag, right = svd(tensor)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(ratio < accept_rate,
                      torch.ones(1).to(ratio.device),
                      torch.zeros(1).to(ratio.device)
                      ).sum()
    return tensor @ right[:, :int(num)]


def zero_mean(tensor: torch.Tensor, dim):
    return tensor - tensor.mean(dim=dim, keepdim=True)


def _svd_cca(x, y):
    u_1, s_1, v_1 = svd(x)
    u_2, s_2, v_2 = svd(y)
    uu = u_1.t() @ u_2
    try:
        u, diag, v = svd(uu)
    except RuntimeError as e:
        raise e
    a = v_1 @ s_1.reciprocal().diag() @ u
    b = v_2 @ s_2.reciprocal().diag() @ v
    return a, b, diag


def _cca(x, y, method):
    """
    Canonical Correlation Analysis,
    cf. Press 2011 "Cannonical Correlation Clarified by Singular Value Decomposition"
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd"  or "qr"
    :return: _cca vectors for input x, _cca vectors for input y, canonical correlations
    """
    assert x.size(0) == y.size(0), f"Number of data needs to be same but {x.size(0)} and {y.size(0)}"
    assert method in ("svd", "qr"), "Unknown method"

    x = zero_mean(x, dim=0)
    y = zero_mean(y, dim=0)
    return _svd_cca(x, y)


def svcca_distance(x, y, method="svd", accept_rate=0.99):
    """
    SVCCA distance proposed in Raghu et al. 2017
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd" (default) or "qr"
    """
    x = svd_reduction(x, accept_rate=accept_rate)
    y = svd_reduction(y, accept_rate=accept_rate)
    div = min(x.size(1), y.size(1))
    a, b, diag = _cca(x, y, method=method)
    res = dict(distance=(1 - diag.sum() / div).item(), cca_coefs=diag.cpu().numpy())
    return res


def pwcca_distance(x, y, method="svd"):
    """
    Project Weighting CCA proposed in Marcos et al. 2018
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd" (default) or "qr"
    """
    a, b, diag = _cca(x, y, method=method)
    alpha = (x @ a).abs().sum(dim=0)
    alpha = alpha / alpha.sum()
    return 1 - alpha @ diag


class CCAHook(object):
    """
    Hook to calculate CCA distance between outputs of layers
    >>> model = resnet18()
    >>> hook1 = CCAHook(model, "layer3.0.conv1")
    >>> hook2 = CCAHook(model, "layer3.0.conv2")
    >>> model.eval()
    >>> model(torch.randn(1200, 3, 224, 224))
    >>> hook1.distance(hook2, 8)
    :param model: nn.Module model
    :param name: name of the layer you use
    :param cca_distance ("pwcca_distance" or "svcca_distance"). "pwcca_distance" by default
    :param svd_device:
    """

    _supported_modules = (nn.Conv3d, nn.Linear)
    _cca_distance_function = {"svcca": svcca_distance,
                              "pwcca": pwcca_distance}

    def __init__(self,
                 model: nn.Module,
                 name: str, *,
                 cca_distance: str or Callable = pwcca_distance,
                 svd_device: str or torch.device = _device):

        self.model = model
        self.name = name
        _dict = {n: m for n, m in self.model.named_modules()}
        if self.name not in _dict.keys():
            raise NameError(f"No such name ({self.name}) in the model")
        if type(_dict[self.name]) not in self._supported_modules:
            raise TypeError(f"{type(_dict[self.name])} is not supported")

        self._module = _dict[self.name]
        self._module = {n: m for n, m in self.model.named_modules()}[self.name]
        self._hooked_value = None
        self._register_hook()
        if type(cca_distance) == str:
            cca_distance = self._cca_distance_function[cca_distance]
        self._cca_distance = cca_distance

        if svd_device not in ("cpu", "cuda"):
            raise RuntimeError(f"Unknown device name {svd_device}")

        if svd_device == "cpu":
            from multiprocessing import cpu_count

            torch.set_num_threads(cpu_count())
        self._svd_device = svd_device

    def clear(self):
        """
        clear the hooked tensor
        """
        self._hooked_value = None

    def distance(self, other, size: int or tuple = None):
        """
        returns cca distance between the hooked tensor and `other`'s hooked tensor.
        :param other: CCAHook
        :param size: if two tensor's size are
        :return: CCA distance
        """

        tensor1 = self.get_hooked_value()
        tensor2 = other.get_hooked_value()

        if tensor1.dim() != tensor2.dim():
            raise RuntimeError("tensor dimensions are incompatible!")
        tensor1 = tensor1.to(self._svd_device)
        tensor2 = tensor2.to(self._svd_device)
        if isinstance(self._module, nn.Linear):
            return self._cca_distance(tensor1, tensor2)
        elif isinstance(self._module, nn.Conv3d):
            return CCAHook._conv3d(tensor1, tensor2, self._cca_distance, size)

    def _register_hook(self):

        def hook(_, __, output):
            self._hooked_value = output

        self._module.register_forward_hook(hook)

    def get_hooked_value(self):
        if self._hooked_value is None:
            raise RuntimeError("Please do model.forward() before CCA!")
        return self._hooked_value

    @staticmethod
    def _conv3d_reshape(tensor, size, same_layer=True):
        b, c, h, w, d = tensor.shape
        if size is not None:
            if (size, size, size) > (h, w, d):
                raise RuntimeError(f"`size` should be smaller than the tensor's size but ({h}, {w}, {d})")
            else:
                tensor = F.adaptive_avg_pool3d(tensor, size)
        if same_layer:
            tensor = tensor.permute(1,0,2,3,4).reshape(c, -1) # size C x (HWDB) [data x neurons]
        else:
            tensor = tensor.reshape(b, -1) # size B x (CHWD) [data x neurons]
        return tensor

    @staticmethod
    def _conv3d(tensor1, tensor2, cca_function, size1, size2, same_layer=True, **kwargs):
        if same_layer:
            assert tensor1.shape == tensor2.shape, 'Wrong shape.'
        else:
            assert tensor1.shape[0] == tensor2.shape[0], 'tensor1 and tensor2 should have the same first dimension.'
        tensor1 = CCAHook._conv3d_reshape(tensor1, size1, same_layer)
        tensor2 = CCAHook._conv3d_reshape(tensor2, size2, same_layer)
        print("CCA computations between tensor 1 ({}) and tensor 2 ({})".format(tensor1.shape, tensor2.shape),
              flush=True)
        return cca_function(tensor1, tensor2, **kwargs)

    @staticmethod
    def data(dataset: Dataset, batch_size: int, *, num_workers: int = 2):
        """
        returns batch of data to calculate CCA distance
        :param dataset: torch.utils.data.Dataset
        :param batch_size:
        :param num_workers:
        :return: tensor
        """
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        input, _ = next(iter(data_loader))
        return input
