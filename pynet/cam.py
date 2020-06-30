# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides tools to compute class activation map.
"""


# Imports
import logging
import skimage
import skimage.transform as sk_transform
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func


# Global parameters
logger = logging.getLogger("pynet")


class FeatureExtractor(object):
    """ Class for extracting activations and registering gradients from
    targetted intermediate layers.
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs(object):
    """ Class for making a forward pass, and getting:
    1- the network output.
    2- activations from intermeddiate targetted layers.
    3- gradients from intermeddiate targetted layers.
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(
            self.model.features, target_layers)

    def get_activations_gradient(self):
        return self.feature_extractor.gradients

    def get_activations(self, x):
        return self.feature_extractor(x)

    def __call__(self, x):
        if hasattr(self.model, "pre"):
            x = self.model.pre(x)
        target_activations, output = self.feature_extractor(x)
        if hasattr(self.model, "pool"):
            output = self.model.pool(output)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


class GradCam(object):
    """ Class for computing class activation map.
    """
    def __init__(self, model, target_layers, labels, top=1, dim="2d"):
        self.model = model
        self.labels = labels
        self.top = top
        self.dim = dim
        self.model.eval()
        self.extractor = ModelOutputs(self.model, target_layers)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):
        features, output = self.extractor(input)
        if output.shape[1] == 1: # if there is a single scalar, assume prob(y=1) = sigma(y)
            pseudo_output = torch.cat([torch.zeros(len(output), 1, device=output.device), output], dim=1)
            pred_prob = func.softmax(pseudo_output, dim=1).data
            output = torch.cat([-output, output], dim=1)
        else:
            pred_prob = func.softmax(output, dim=1).data
        probs, indices = pred_prob.data.max(dim=1)
        probs = probs.data.detach().cpu().numpy()
        indices = indices.data.detach().cpu().numpy()
        heatmaps = {}
        for cnt, (prob, index) in enumerate(zip(probs, indices)):
            if cnt == self.top:
                break
            label = self.labels[index]
            line = "{0:.3f} -> {1}".format(prob, label)
            ## Get the nb of classes (size(output) == [b, n_classes])
            nb_classes = output.size()[-1]
            one_hot = np.zeros((1, nb_classes), dtype=np.float32)
            logger.info(line)
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = torch.tensor(torch.from_numpy(one_hot), requires_grad=True, device=output.device)
            ## Get only Y_c where c == index which is the most probable class for sample i
            one_hot = torch.sum(one_hot * output)
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()

            ## Back-propagate the gradients
            one_hot.backward(retain_graph=True)
            ## Get only the last gradients computed through the last target layer
            gradients = self.extractor.get_activations_gradient()[-1]
            gradients = gradients.cpu().data.numpy()
            # Avg spatially the gradient accross the channels (on scalar per channel)
            if self.dim == "2d":
                pooled_gradients = np.mean(gradients, axis=(0, 2, 3))
            else:
                pooled_gradients = np.mean(gradients, axis=(0, 2, 3, 4))
            ## Get the activation map
            activations = features[-1]
            activations = activations.cpu().data.numpy()
            for cnt, weight in enumerate(pooled_gradients):
                activations[:, cnt] *= weight
            heatmap = np.mean(activations, axis=1).squeeze()
            heatmap = np.maximum(heatmap, 0)
            heatmap -= np.min(heatmap)
            heatmap /= np.max(heatmap)
            heatmap_highres = sk_transform.resize(
                heatmap, input.shape[2:])
            heatmaps[label] = (input, heatmap, heatmap_highres)
        return heatmaps
