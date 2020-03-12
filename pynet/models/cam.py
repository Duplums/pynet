# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides reorganized networks to perform class activation map.
Networks must have features/classifier methods for the convolutional part of
the network, and the fully connected part.
"""


# Imports
import collections
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as func
import torch


# Global parameters
CAM_NETWORKS = {
    "vgg19": "35",
    "densenet201": "norm5",
    "resnet18": "layer4",
    "inception_v3": "Mixed_7c",
    "lenet": "conv3"
}

from pynet.cam import GradCam
import matplotlib.pyplot as plt

def plot_cam_heatmaps(name, data_manager, labels_mapping, model=None):
    loaders = data_manager.get_dataloader(test=True)
    heatmaps = []
    model, activation_layer_name = get_cam_network(name, model)
    grad_cam = GradCam(model, [activation_layer_name], labels_mapping, top=1)
    for i, dataitem in enumerate(loaders.test):
        if i > 1:
            break
        heatmaps.extend(grad_cam(dataitem.inputs).items())

    fig, axs = plt.subplots(nrows=2, ncols=len(heatmaps))
    fig.suptitle(name, fontsize="large")
    for cnt, (name, (img, arr, arr_highres)) in enumerate(heatmaps):
        axs[0, cnt].set_title(name)
        axs[0, cnt].matshow(arr)
        axs[0, cnt].set_axis_off()
        _img = img.data.numpy()[0].transpose((1, 2, 0))
        axs[1, cnt].imshow(_img)
        axs[1, cnt].imshow(arr_highres, alpha=0.6, cmap="jet")
        axs[1, cnt].set_axis_off()
    plt.show()

def get_cam_network(name, model=None):
    """ Reorganized a network to provide a features/classifier methods for the
    convolutional part of the network, and the fully connected part.

    Parameters
    ----------
    name: str
        the name of the network.

    Returns
    -------
    model: instance
        the pretrained/reorganized model.
    activation_layer_name: str
        the name of the activation layer in the network.
    """
    if name not in CAM_NETWORKS:
        raise ValueError("'{0}' network is not yet supported.".format(name))
    model = model or getattr(models, name)(pretrained=True)
    activation_layer_name = CAM_NETWORKS[name]
    if name == "resnet18":
        model = nn.Sequential(collections.OrderedDict([
            ("features", nn.Sequential(collections.OrderedDict(
                list(model.named_children())[:-2]))),
            ("pool", model.avgpool),
            ("classifier", model.fc)
        ]))
    elif name == "lenet":
        model = nn.Sequential(collections.OrderedDict([
            ("features", nn.Sequential(collections.OrderedDict(
                list(model.named_children())[:-4]))),
            ("pool", model.adaptive_maxpool),
            ("classifier", model.hidden_layer2)
        ]))
    elif name == "densenet201":
        model = nn.Sequential(collections.OrderedDict([
            ("features", model.features),
            ("pool", nn.AvgPool2d(kernel_size=7, stride=1)),
            ("classifier", model.classifier)
        ]))
    elif name == "inception_v3":
        class Inception3_features(nn.Module):
            def __init__(self, model):
                super(Inception3_features, self).__init__()
                self.part1 = nn.Sequential(collections.OrderedDict([
                    (name, mod)
                    for name, mod in list(model.named_children())[:3]]))
                self.part2 = nn.Sequential(collections.OrderedDict([
                    (name, mod)
                    for name, mod in list(model.named_children())[3: 5]]))

            def forward(self, x):
                x = self.part1(x)
                x = func.max_pool2d(x, kernel_size=3, stride=2)
                x = self.part2(x)
                x = func.max_pool2d(x, kernel_size=3, stride=2)
                return x
        features_mods = [
            (name, mod)
            for name, mod in list(model.named_children())[5: -1]
            if name != "AuxLogits"]
        model = nn.Sequential(collections.OrderedDict([
            ("pre", Inception3_features(model)),
            ("features", nn.Sequential(
                collections.OrderedDict(features_mods))),
            ("pool", nn.AvgPool2d(kernel_size=5)),
            ("classifier", model.fc)
        ]))
    return model, activation_layer_name
