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
import torch

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
            return label

class HardNormalization(object):
    def __init__(self, min=-1.0, max=1.0, eps=1e-8):
        self.min = min
        self.max = max
        self.eps = eps

    def __call__(self, arr):
        min_arr = torch.min(arr)
        max_arr = torch.max(arr)
        if torch.abs(min_arr - max_arr) < self.eps:
            return torch.zeros_like(arr)
        return ((self.max-self.min) * arr + (self.min*max_arr - self.max*min_arr))/(max_arr-min_arr)

class RandomFlip(object):
    def __init__(self, vflip=False, hflip=True, proba=0.5):
        self.vflip = vflip
        self.hflip = hflip
        self.prob = proba

    def __call__(self, arr):
        if self.vflip and np.random.rand() < self.prob:
            arr = torch.flip(arr, dims=[2])
        if self.hflip and np.random.rand() < self.prob:
            arr = torch.flip(arr, dims=[1])
        return arr

class RandomPatchInversion(object):
    def __init__(self, patch_size=10, data_threshold=0):
        self.data_threshold = data_threshold
        self.patch_size = patch_size

    def __call__(self, arr, label=None):
        assert isinstance(arr, torch.Tensor)
        if label is None:
            label = int(np.random.rand() < 0.5)
        assert label in [0, 1], "Unexpected label"
        if label == 1:
            # Selects 2 random non-overlapping patch
            mask = (arr > self.data_threshold)
            # Get a first random patch inside the mask
            patch1 = self.get_random_patch(mask)
            # Get a second one outside the first patch and inside the mask
            mask[patch1] = False
            patch2 = self.get_random_patch(mask)
            arr = arr.copy()
            data_patch1 = arr[patch1].copy()
            arr[patch1] = arr[patch2]
            arr[patch2] = data_patch1
            print(patch1, patch2)
        return arr, torch.tensor(label, device=arr.device)
    
    def get_random_patch(self, mask):
        # Warning: we assume the mask is convex
        possible_indices = mask.nonzero(as_tuple=True)
        if len(possible_indices[0]) == 0:
            raise ValueError("Empty mask")
        index = np.random.randint(len(possible_indices[0]))
        point = [min(ind[index], mask.shape[i]-self.patch_size) for i, ind in enumerate(possible_indices)]
        patch = tuple([slice(p, p + self.patch_size) for p in point])
        return patch
    
class Random90_3DRot(object):
    """Applies a rotation in {0, 90, 180, 270} in each direction and returns a label k in [0..23]"""
    def __init__(self, authorized_rot=None, axes=None):
        if authorized_rot is not None:
            assert set(authorized_rot) <= {0, 90, 180, 270}
        self.authorized_rot = list(authorized_rot or [0, 90, 180, 270])
        self.nb_rots = len(self.authorized_rot)
        self.num_classes = self.nb_rots * 3 * 2 # 3 axes, 2 directions or the nb of faces in a cube
        self.rot_to_k = {0: 0, 90: 1, 180: 2, 270: 3}

        # The 'front' is the axes (1, 2) here. It is arbitrary.
        self.authorized_axes = [(1, 2), (1, 3), (2, 3)]
        self.cube_face_to_back = [(2, (1, 3)), (1, (3, 2)), (1, (3, 1))]
        self.cube_face_to_front = [(0, (1, 2)), (1, (2, 3)), (1, (1, 3))]

        self.cube_face = None
        if axes is not None:
            assert axes in self.authorized_axes, "Axes must be in {}".format(self.authorized_axes)
            self.num_classes = self.nb_rots
            self.cube_face = 2 * self.authorized_axes.index(axes)

        # Small test to confirm that everything is ok (i.e the transformation T: label -> T(I) is injective
        # for any image I)
        self.test_unicity()

    def __call__(self, arr, label=None):
        assert len(arr.shape) == 4 and isinstance(arr, torch.Tensor)
        # Chose a label
        if label is None:
            label = np.random.randint(0, self.num_classes)
        assert label in np.arange(self.num_classes), "Unexpected label"
        # If the cube's face is already predefined, use it
        if self.cube_face is not None:
            cube_face = self.cube_face
            angle = self.authorized_rot[label]
        else:
            # Get the associated angle and cube's face
            angle_index, cube_face = np.unravel_index(label, (self.nb_rots, 6))
            angle = self.authorized_rot[angle_index]

        # From the cube's face, deduce the axis and direction
        (direction, face_axes) = (cube_face%2, self.authorized_axes[cube_face//2])

        # Put the selected face to front or back (front is the axes (0, 1) in 3D, (1, 2) in 4D with the channel)
        if direction == 0:
            (k, axes) = self.cube_face_to_front[cube_face//2]
            arr = torch.rot90(arr, k=k, dims=axes)

        elif direction == 1:
            (k, axes) = self.cube_face_to_back[cube_face//2]
            arr = torch.rot90(arr, k=k, dims=axes)

        # Rotate of the chosen angle in the direction selected
        arr = torch.rot90(arr, k=self.rot_to_k[angle], dims=face_axes)
        return arr, label

    def test_unicity(self):
        m = torch.arange(8).reshape((1, 2, 2, 2))
        list_permutation = []
        for k in range(self.num_classes):
            rotated_m, label = self(m, label=k)
            for m_0 in list_permutation:
                if rotated_m == m_0:
                    raise ValueError("Several labels map to the same configuration")
            list_permutation.append(rotated_m)
        return list_permutation

class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr):
        return self.std * (arr - torch.mean(arr))/(torch.std(arr.float()) + self.eps) + self.mean

class Crop(object):
    """Crop the given n-dimensional array either at a random location or centered"""
    def __init__(self, shape, type="center"):
        assert type in ["center", "random"]
        self.shape = shape
        self.copping_type = type

    def __call__(self, arr):
        assert isinstance(arr, torch.Tensor)
        assert type(self.shape) == int or len(self.shape) == len(arr.shape)

        img_shape = np.array(arr.shape)
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
        return arr + self.std * torch.randn_like(arr)

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
        assert len(arr.shape) == 4 and isinstance(arr, torch.Tensor) # == (C, H, W, D)

        arr_shape = np.array(arr.shape)
        angles = [np.deg2rad(np.random.random() * (angle_max - angle_min) + angle_min)
                  for (angle_min, angle_max) in self.angles]
        alpha, beta, gamma = angles[0], angles[1], angles[2]
        rot_x = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
        rot_y = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        rot_z = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        R = np.matmul(np.matmul(rot_z, rot_y), rot_x)
        middle_point = (np.asarray(arr_shape[1:]) - 1) / 2
        offset = middle_point - np.dot(middle_point, R)

        translation = [np.round(np.random.random() * (2*arr_shape[i+1]*t) - arr_shape[i+1]*t)
                       for i,t in enumerate(self.translate)]
        out = np.zeros(arr_shape, dtype=arr.dtype)
        for c in range(arr_shape[0]):
            affine_transform(arr[c].cpu().numpy(), R.T, offset=offset+translation, output=out[c], mode='nearest')

        return torch.tensor(out, device=arr.device)


if __name__ == '__main__':
    from pynet.plotting.image import plot_anat_array
    import nibabel

    t = RandomAffineTransform3d(40, 0.1)
    test_1 = np.array([nibabel.load('/neurospin/psy/hcp/derivatives/cat12vbm/sub-165941/mri/mwp1165941_3T_T1w_MPR1.nii').get_data()])
    plot_anat_array(test_1[0])
    test_1_trans = t(test_1)
    plot_anat_array(test_1_trans[0])


class Rotation(object):
    # TODO: convert it to handle torch tensors
    def __init__(self, angle, axes=(1,2), reshape=True, **kwargs):
        self.angle = angle
        self.axes = axes
        self.reshape = reshape
        self.rotate_kwargs = kwargs

    def __call__(self, arr):
        return rotate(arr, self.angle, axes=self.axes, reshape=self.reshape, **self.rotate_kwargs)

class RandomRotation(object):
    # TODO: convert it to handle torch tensors
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
    def __init__(self, shape, **kwargs):
        """ Initialize the instance.

        Parameters
        ----------
        shape: list of int
            the desired shape.
        **kwargs: kwargs given to torch.nn.functional.pad()
        """
        self.shape = shape
        self.kwargs = kwargs

    def __call__(self, arr):
        """ Fill an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array.

        Returns
        -------
        fill_arr: np.array
            the padded array.
        """
        if len(arr.shape) >= len(self.shape):
            return self._apply_padding(arr)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, arr):
        """ See Padding.__call__().
        """
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.extend([half_shape_i, half_shape_i])
            else:
                padding.extend([half_shape_i, half_shape_i + 1])
        for cnt in range(len(arr.shape) - len(padding)):
            padding.extend([0, 0])
        fill_arr = torch.nn.functional.pad(arr, padding[::-1], **self.kwargs)
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
