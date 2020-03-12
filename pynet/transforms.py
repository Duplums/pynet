# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that defines common transformations that can be applied when the dataset
is loaded.
"""

# Imports
import collections
import numpy as np
from scipy.ndimage import rotate, affine_transform


class LabelMapping(object):

    def __init__(self, **mappings):
        self.mappings = mappings

    def __call__(self, label):
        if isinstance(label, list) or isinstance(label, np.ndarray):
            l_to_return = []
            for l in label:
                l_to_return.append(self.__call__(l))
            return l_to_return
        if label in self.mappings:
            return self.mappings[label]
        else:
            return float(label)


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr):
        return self.std * (arr - np.mean(arr))/(np.std(arr) + self.eps) + self.mean

class Crop(object):
    """Crop the given n-dimensional array either at a random location or centered"""
    def __init__(self, shape, type="center"):
        assert type in ["center", "random"]
        self.shape = shape
        self.copping_type = type

    def __call__(self, arr):
        assert isinstance(arr, np.ndarray)
        assert type(self.shape) == int or len(self.shape) == len(arr.shape)

        img_shape = arr.shape
        if type(self.shape) == int:
            size = [self.shape for _ in range(len(self.shape))]
        else:
            size = np.copy(self.shape)
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.copping_type == "center":
                delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
            elif self.copping_type == "random":
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(delta_before, delta_before + size[ndim]))

        return arr[tuple(indexes)]

class GaussianNoise:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, arr):
        return arr + np.random.normal(0, self.std, arr.shape)

class RandomAffineTransform3d:
    def __init__(self, angles, translate):
        ## angles == list of int or tuple indicating the range of degrees to select from in each direction
        ## translate == tuple of maximum absolute fraction translation shift in each direction
        if type(angles) in [int, float]:
            angles = [[-angles, angles] for _ in range(3)]
        elif type(angles) == list and len(angles) == 3:
            for i in range(3):
                if type(angles[i]) in [int, float]:
                    angles[i] = [-angles[i], angles[i]]
        else:
            raise ValueError("Unkown angles type: {}".format(type(angles)))
        self.angles = angles
        if type(translate) in [float, int]:
            translate = (translate, translate, translate)
        assert len(translate) == 3
        self.translate = translate

    def __call__(self, arr):
        assert len(arr.shape) == 4 # == (C, H, W, D)
        arr_shape = arr.shape

        angles = [np.deg2rad(np.random.random() * (angle_max - angle_min) + angle_min)
                  for (angle_min, angle_max) in self.angles]
        alpha, beta, gamma = angles[0], angles[1], angles[2]
        rot_x = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
        rot_y = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        rot_z = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        R = np.matmul(np.matmul(rot_z, rot_y), rot_x)
        middle_point = (np.asarray(arr_shape[1:]) - 1) / 2
        offset = middle_point - np.dot(middle_point, R)

        translation = [np.round(np.random.random() * (2*arr.shape[i+1]*t) - arr.shape[i+1]*t)
                       for i,t in enumerate(self.translate)]
        out = np.zeros(arr.shape, dtype=arr.dtype)

        for c in range(arr.shape[0]):
            affine_transform(arr[c], R.T, offset=offset+translation, output=out[c], mode='nearest')

        return out


if __name__ == '__main__':
    from pynet.plotting.image import plot_anat_array
    import nibabel

    t = RandomAffineTransform3d(40, 0.1)
    test_1 = np.array([nibabel.load('/neurospin/psy/hcp/derivatives/cat12vbm/sub-165941/mri/mwp1165941_3T_T1w_MPR1.nii').get_data()])
    plot_anat_array(test_1[0])
    test_1_trans = t(test_1)
    plot_anat_array(test_1_trans[0])


class Rotation(object):
    def __init__(self, angle, axes=(1,2), reshape=True, **kwargs):
        self.angle = angle
        self.axes = axes
        self.reshape = reshape
        self.rotate_kwargs = kwargs

    def __call__(self, arr):
        return rotate(arr, self.angle, axes=self.axes, reshape=self.reshape, **self.rotate_kwargs)

class RandomRotation(object):
    """ nd generalisation of https://pytorch.org/docs/stable/torchvision/transforms.html section RandomRotation"""
    def __init__(self, angles, axes=(0,2), reshape=True, **kwargs):
        if type(angles) in [int, float]:
            self.angles = [-angles, angles]
        elif type(angles) == list and len(angles) == 2 and angles[0] < angles[1]:
            self.angles = angles
        else:
            raise ValueError("Unkown angles type: {}".format(type(angles)))
        if axes is None:
            print('Warning: rotation plane will be determined randomly')
        self.axes = axes
        self.reshape = reshape
        self.rotate_kwargs = kwargs

    def __call__(self, arr):
        angle = np.random.random() * (self.angles[1] - self.angles[0]) + self.angles[0]
        return rotate(arr, angle, axes=self.axes, reshape=self.reshape, **self.rotate_kwargs)


class Padding(object):
    """ A class to pad an image.
    """
    def __init__(self, shape, nb_channels=1, fill_value=0):
        """ Initialize the instance.

        Parameters
        ----------
        shape: list of int
            the desired shape.
        nb_channels: int, default 1
            the number of channels.
        fill_value: int or list of int, default 0
            the value used to fill the array, if a list is given, use the
            specified value on each channel.
        """
        self.shape = shape
        self.nb_channels = nb_channels
        self.fill_value = fill_value
        if self.nb_channels > 1 and not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value] * self.nb_channels
        elif isinstance(self.fill_value, list):
            assert len(self.fill_value) == self.nb_channels

    def __call__(self, arr):
        """ Fill an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array.

        Returns
        -------
        fill_arr: np.array
            the zero padded array.
        """
        if len(arr.shape) - len(self.shape) == 1:
            data = []
            for _arr, _fill_value in zip(arr, self.fill_value):
                data.append(self._apply_padding(_arr, _fill_value))
            return np.asarray(data)
        elif len(arr.shape) - len(self.shape) == 0:
            return self._apply_padding(arr, self.fill_value)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, arr, fill_value):
        """ See Padding.__call__().
        """
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append((half_shape_i, half_shape_i))
            else:
                padding.append((half_shape_i, half_shape_i + 1))
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append((0, 0))
        fill_arr = np.pad(arr, padding, mode="constant",
                          constant_values=fill_value)
        return fill_arr


class Downsample(object):
    """ A class to downsample an array.
    """
    def __init__(self, scale, with_channels=True):
        """ Initialize the instance.

        Parameters
        ----------
        scale: int
            the downsampling scale factor in all directions.
        with_channels: bool, default True
            if set expect the array to contain the channels in first dimension.
        """
        self.scale = scale
        self.with_channels = with_channels

    def __call__(self, arr):
        """ Downsample an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array

        Returns
        -------
        down_arr: np.array
            the downsampled array.
        """
        if self.with_channels:
            data = []
            for _arr in arr:
                data.append(self._apply_downsample(_arr))
            return np.asarray(data)
        else:
            return self._apply_downsample(arr)

    def _apply_downsample(self, arr):
        """ See Downsample.__call__().
        """
        slices = []
        for cnt, orig_i in enumerate(arr.shape):
            if cnt == 3:
                break
            slices.append(slice(0, orig_i, self.scale))
        down_arr = arr[tuple(slices)]

        return down_arr
