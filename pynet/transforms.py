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


# Third party import
import numpy as np

class Normalize:
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr):
        assert type(arr) == np.ndarray
        return self.std * (arr - np.mean(arr))/(np.std(arr) + self.eps) + self.mean


class CenterCrop(object):

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, arr):
        assert type(arr) == np.ndarray
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
            delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
            indexes.append('%i:%i' % (delta_before, delta_before + size[ndim]))

        return eval('arr[%s]' % ','.join(indexes))



class ZeroPadding(object):
    """ A class to zero pad an image.
    """
    def __init__(self, shape):
        """ Initialize the instance.

        Parameters
        ----------
        shape: list of int
            the desired shape.
        """
        self.shape = shape

    def __call__(self, arr):
        """ Zero fill an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array

        Returns
        -------
        fill_arr: np.array
            the zero padded array.
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
        fill_arr = np.pad(arr, padding, mode="constant", constant_values=0)

        return fill_arr


class Downsample(object):
    """ A class to downsample an array.
    """
    def __init__(self, scale):
        """ Initialize the instance.

        Parameters
        ----------
        scale: int
            the downsampling scale factor in all directions.
        """
        self.scale = scale

    def __call__(self, arr):
        """ Downsample an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array
        scale: int
            the downsampling scale factor in all directions.

        Returns
        -------
        down_arr: np.array
            the downsampled array.
        """
        slices = []
        for cnt, orig_i in enumerate(arr.shape):
            if cnt == 3:
                break
            slices.append(slice(0, orig_i, self.scale))
        down_arr = arr[tuple(slices)]

        return down_arr
