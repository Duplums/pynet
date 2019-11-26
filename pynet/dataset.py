# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides functions to load and split a dataset.
"""


import torch
from torch.utils.data import Dataset, DataLoader
import nibabel
import progressbar
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit


class DataManager:

    def __init__(self, input_path, dtype, shape, output_path, outputs,
                 number_of_folds, batch_size,
                 labels=None,
                 test_size=0.1,
                 stratify=False,
                 transforms=None,
                 **dataloader_kwargs):
        """ Splits an input binary memory-mapped dataset in test, train and validation.
        This function stratifies the data.

        Parameters
        ----------
        input_path: str
            the path to the binary memory-mapped data that will be splited.
        shape: tuple
            the shape of the input data
        dtype: np.dtype
            type of the inputs
        output_path: str
            the path to a .csv file
        outputs: list of str
            the name of the column(s) containing the ouputs.
        number_of_folds: int
            the number of folds that will be used in the cross validation.
        batch_size: int
            the size of each mini-batch (only applied on the train set).
        labels: str
            the name of the column containing the labels (required if stratify=True)
        stratify: bool
            whether the data should be stratified (ie the % of samples / per class is preserved) or not
        transforms: list of callable
            transforms a list of samples with pre-defined transformation
            ...
        test_size: float, default 0.1
            should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the test split.
        """

        df = pd.read_csv(output_path)
        self.inputs = np.memmap(input_path, dtype=dtype, mode='r+', shape=shape)
        self.dtype = dtype
        self.outputs = df[outputs].values
        self.labels = df[labels] if labels is not None else labels
        self.batch_size = batch_size
        self.data_loader_kwargs = dataloader_kwargs
        self.transforms = transforms

        # 1st step: split into train/test (get only indices)
        if stratify:
            splitter = StratifiedShuffleSplit(n_splits=1, random_state=0, test_size=test_size)
            train_index, test_index = next(splitter.split(self.inputs, self.labels))
        else:
            splitter = ShuffleSplit(n_splits=1, random_state=0, test_size=test_size)
            train_index, test_index = next(splitter.split(self.inputs, self.outputs))


        self.dataset = {
            'test': MMAPDataset(self.inputs, self.outputs,
                                test_index),
            'train': [],
            'validation': []
        }

        # 2nd step: split the training set into K folds (K-1 for training, 1 for validation, K times)
        if stratify:
            Kfold_splitter = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=0)
            gen = Kfold_splitter.split(np.ones(len(train_index)), self.labels[train_index])
        else:
            Kfold_splitter = KFold(n_splits=number_of_folds, shuffle=True, random_state=0)
            gen = Kfold_splitter.split(np.ones(len(train_index)))

        for fold_train_index, fold_val_index in gen:
            train_dataset = MMAPDataset(
                self.inputs, self.outputs,
                train_index[fold_train_index])
            val_dataset = MMAPDataset(
                self.inputs, self.outputs,
                train_index[fold_val_index])
            self.dataset["train"].append(train_dataset)
            self.dataset["validation"].append(val_dataset)

    def collate_fn(self, list_samples):
        # List of tuples (input, output)
        for k in range(len(list_samples)):
            input_, target_ = list_samples[k]
            for tf in self.transforms:
                input_ = tf(input_)
            list_samples[k] = torch.as_tensor(input_), torch.as_tensor(target_)
        return torch.stack([s[0] for s in list_samples], dim=0), torch.stack([s[1] for s in list_samples], dim=0).float()


    def get_dataloader(self, test=False, train=True, num_fold=0, with_validation=True):
        if test:
            return DataLoader(self.dataset["test"],
                              batch_size=self.batch_size,
                              collate_fn=DataManager.collate_fn,
                              **self.data_loader_kwargs)
        elif train:
            if num_fold is None:
                raise ValueError("num_fold arg must be set when generating the training data loader.")
            train_loader =  DataLoader(self.dataset["train"][num_fold],
                                       batch_size=self.batch_size,
                                       collate_fn=self.collate_fn,
                                       **self.data_loader_kwargs)
            if with_validation:
                val_loader = DataLoader(self.dataset["validation"][num_fold],
                                        batch_size=self.batch_size,
                                        collate_fn=self.collate_fn,
                                        **self.data_loader_kwargs)
                return train_loader, val_loader
            return train_loader


class MMAPDataset(Dataset):

    def __init__(self, inputs, outputs, indices):
        # indices == the list of index that is considered in this dataset.
        assert len(inputs) == len(outputs)
        self.inputs = inputs
        self.outputs = outputs
        self.indices = indices

    # Gets only the index needed (thanks to memory mapping)
    def __getitem__(self, item):
        return self.inputs[self.indices[item]], self.outputs[self.indices[item]]

    def __len__(self):
        return len(self.indices)


def split_dataset(path, dataloader, inputs, label, number_of_folds,
                  batch_size, outputs=None, transforms=None, test_size=0.1,
                  validation_size=0.25, nb_samples=None, verbose=0,
                  **dataloader_kwargs):
    """ Split an input tabular dataset in test, train and validation.
    This function stratifies the data.

    Parameters
    ----------
    path: str
        the path to the tabular data that will be splited.
    dataloader: class
        a pytorch Dataset derived class used to load the data in a mini-batch
        fashion.
    inputs: list of str
        the name of the column(s) containing the inputs
    label: str
        the name of the column containing the labels.
    number_of_folds: int
        the number of folds that will be used in the cross validation.
    batch_size: int
        the size of each mini-batch (only applied on the train set).
    outputs: list of str, default None
        the name of the column(s) containing the ouputs.
    transforms: list of callable
        transform the dataset using these functions: cropping, reshaping,
        ...
    flatten: bool, default False
        flatten the data array.
    slice_volume_index: int, default None
        convert 3D volume to N slices.
    squeeze_channel: bool, default False
        remove the channel dimension if 1.
    test_size: float, default 0.1
        should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the test split.
    validation_size: float, default 0.25
        should be between 0.0 and 1.0 and represent the proportion of the
        dataset (without the test dataset) to include in the validation split.
    nb_samples: int, default None
        cut the input tabular dataset.
    verbose: int, default 0
        the verbosity level.

    Retunrs
    -------
    dataset: dict
        the test, train, and validation loaders.
    """
    if outputs is not None:
        columns = inputs + outputs + [label]
    else:
        columns = inputs + [label]
    df = pd.read_csv(path, sep="\t")
    if nb_samples is not None:
        df = df.loc[:nb_samples]
    arr = df[columns].values
    arr_optim, arr_test = train_test_split(
        arr,
        test_size=test_size,
        random_state=1,
        stratify=arr[:, -1])
    dataset = {}
    testloader = dataloader(
        dataset=pd.DataFrame(arr_test, columns=columns),
        inputs=inputs,
        outputs=outputs,
        label=label,
        transforms=transforms,
        batch_size=len(arr_test),
        verbose=verbose,
        **dataloader_kwargs)
    dataset["test"] = testloader
    for fold_indx in range(number_of_folds):
        arr_train, arr_valid = train_test_split(
            arr_optim,
            test_size=validation_size,
            shuffle=True,
            stratify=arr_optim[:, -1])
        if batch_size == -1:
            batch_size = len(arr_train)
        trainloader = dataloader(
            dataset=pd.DataFrame(arr_train, columns=columns),
            inputs=inputs,
            outputs=outputs,
            label=label,
            transforms=transforms,
            batch_size=batch_size,
            verbose=verbose, 
            **dataloader_kwargs)
        validloader = dataloader(
            dataset=pd.DataFrame(arr_valid, columns=columns),
            inputs=inputs,
            outputs=outputs,
            label=label,
            transforms=transforms,
            batch_size=len(arr_valid),
            verbose=verbose,
            **dataloader_kwargs)
        dataset.setdefault("train", []).append(trainloader)
        dataset.setdefault("validation", []).append(validloader)
    return dataset


def dummy_dataset(nb_batch, batch_size, number_of_folds, shape, verbose=0):
    """ Create a dummy dataset.

    Parameters
    ----------
    nb_batch: int
        the number of batch.
    batch_size: int
        the size of each mini-batch.
    number_of_folds: int
        the number of folds that will be used in the cross validation.
    shape: nuplet
        the generated data shape.
    verbose: int, default 0
        the verbosity level.

    Retunrs
    -------
    dataset: dict
        the test, train, and validation loaders.
    """
    dataset = {
        "test": DummyDataset(
            nb_batch, batch_size, shape, verbose=verbose),
        "train": [DummyDataset(
            nb_batch, batch_size, shape, verbose=verbose)] * number_of_folds,
        "validation": [DummyDataset(
            nb_batch, batch_size, shape, verbose=verbose)] * number_of_folds}
    return dataset


class LoadDataset(Dataset):
    """ Class to load a dataset in a mini-batch fashion.

    Note: the image are expected to be in the FSL order X, Y, Z, N.
    """

    def __init__(self, dataset, inputs, label, outputs=None, batch_size=10,
                 transforms=None, flatten=False, slice_volume_index=None,
                 squeeze_channel=False, load=True, gray_to_rgb=False,
                 verbose=0):
        """ Initialize the class.

        Parameters
        ----------
        dataset: DataFrame
            a pandas dataframe.
        inputs: list of str
            the name of the column(s) containing the inputs
        label: str
            the name of the column containing the labels.
        outputs: list of str, default None
            the name of the column(s) containing the ouputs.
        batch_size: int, default 10
            the size of each mini-batch.
        transforms: list of callable
            transform the dataset using these functions: cropping, reshaping,
            ...
        flatten: bool, default False
            flatten the data array.
        slice_volume_index: int, default None
            convert 3D volume to N slices.
        squeeze_channel: bool, default False
            remove the channel dimension if 1.
        load: bool, default True
            start loading the full dataset or load the dataset on the fly to
            save memory (slower).
        gray_to_rgb: bool, default False
            convert grayscale image (1 channel) to RGB (3 channels): only
            available for 2D dataset.
        verbose: int, default 0
            the verbosity level.
        """
        self.dataset = dataset
        self.inputs = inputs
        self.label = label
        self.outputs = outputs
        self.batch_size = batch_size
        self.transforms = transforms or []
        self.flatten = flatten
        self.slice_volume_index = slice_volume_index
        self.squeeze_channel = squeeze_channel
        self.load = load
        self.gray_to_rgb = gray_to_rgb
        self.verbose = verbose
        self.all_labels = sorted(np.unique(self.dataset[label].values))
        div, self.rest = divmod(len(self.dataset), self.batch_size)
        self.nb_batch = div
        if self.rest > 0:
            self.nb_batch += 1
        self._loaded = {}
        self._batch = {}
        if self.load:
            nb_rows = len(self.dataset)
            for name in inputs + (outputs or []):
                with progressbar.ProgressBar(max_value=nb_rows) as bar:
                    for cnt, path in enumerate(self.dataset[name]):
                        self._loaded[path] = LoadDataset._load(
                            path, self.transforms, self.gray_to_rgb)
                        bar.update(cnt)

    def __len__(self):
        """ Get the number of mini-batch.

        Returns
        -------
        nb_batch: int
            the number of mini-batchs.
        """
        return self.nb_batch

    @classmethod
    def _load(cls, path, transforms=None, gray_to_rgb=False):
        """ Load a dataset.
        """
        if path.endswith((".png", ".jpg", ".jpeg")):
            im = Image.open(path)
            if gray_to_rgb:
                im_data = np.array(im).astype(np.uint8)
                #im_data.shape += (1, )
                #im_data = np.concatenate([im_data] * 3, axis=-1)
                im_data = cv2.cvtColor(im_data, cv2.COLOR_GRAY2RGB)
                im = Image.fromarray(im_data)
            arr = np.array(im)
            dim = 2
        elif path.endswith((".nii", ".nii.gz")):
            arr = nibabel.load(path).get_data()
            dim = arr.ndim
        else:
            raise ValueError(
                "Unsuppported file format: {0}.".format(path))
        for trf in (transforms or []):
            arr = trf(arr)
        arr = arr.astype(np.float32)
        return arr, dim   

    def _concat_features(self, row, names):
        """ Concate the inputs or ouputs in a 1, NB_FEATURES, X, Y, Z array.
        """
        arr = None
        names = names or []
        for name in names:
            path = row[name].values[0]
            _arr, dim = self._loaded.get(path, (None, None))
            if _arr is None:
                _arr, dim = LoadDataset._load(
                    path, self.transforms, self.gray_to_rgb)
            if dim == 2:
                if _arr.ndim == 2:
                    if self.flatten:
                        _arr = _arr.flatten()
                    _arr = np.expand_dims(_arr, axis=0)
                elif _arr.ndim == 3:
                    _arr = _arr.transpose(2, 0 , 1)
                    if self.flatten:
                        _arr = _arr.reshape(_arr.shape[0], -1)
            elif dim == 3:
                if self.flatten:
                    _arr = _arr.flatten()
                _arr = np.expand_dims(_arr, axis=0)
            elif dim == 4:
                _arr = np.ascontiguousarray(_arr.transpose(3, 0, 1, 2))
                if self.flatten:
                    _arr = _arr.reshape(_arr.shape[0], -1)
            else:
                raise ValueError(
                    "Unsupported image dimension: {0}.".format(path))
            _arr = np.expand_dims(_arr, axis=0)
            if arr is None:
                arr = _arr
            else:
                arr = np.concatenate((arr, _arr), axis=1)
        if self.slice_volume_index is not None and arr.ndim == 5:
            indices = [0, 1, 2]
            assert self.slice_volume_index in indices
            indices.remove(self.slice_volume_index)
            arr = np.squeeze(arr, axis=0)
            arr = arr.transpose(
                self.slice_volume_index + 1, 0, indices[0] + 1, indices[1] + 1)
        if self.squeeze_channel and arr is not None and arr.shape[1] == 1:
            arr = np.squeeze(arr, axis=1)
        return arr

    def __getitem__(self, index):
        """ Get the desired mini-batch.

        Parameters
        ----------
        index: int
            the index of the desired mini-batch.

        Returns
        -------
        data: dict
            a dictionary with the inputs, ouputs, and labels mini-batch arrays.
        """
        if index >= self.nb_batch:
            raise IndexError("Index '{0}' is nout of range.".format(index))
        data = self._batch.get(index)
        if data is None:
            if self.rest > 0 and index == self.nb_batch - 1:
                index_end = self.batch_size * index + self.rest
            else:
                index_end = self.batch_size * (index + 1)
            _input_arrs = []
            _output_arrs = []
            labels = []
            for _idx in range(self.batch_size * index, index_end):
                row = self.dataset.iloc[[_idx]]
                _input_arrs.append(self._concat_features(row, self.inputs))
                _arr = self._concat_features(row, self.outputs)
                if _arr is not None:
                    _output_arrs.append(_arr)
                labels.append(row[self.label].values[0])
            input_arr = np.concatenate(_input_arrs, axis=0)
            output_arr = None
            if len(_output_arrs) > 0:
                output_arr = np.concatenate(_output_arrs, axis=0)
            labels = np.asarray(labels)
            if self.verbose > 0:
                print("-" * 50)
                print("Mini Batch: ", index)
                print("Mini Batch Size: ", self.batch_size)
                print("Inputs: ", input_arr.shape)
                print("Ouputs: ", output_arr.shape
                                  if output_arr is not None else None)
                print("Labels: ", labels.shape)
            data = {
                "inputs": input_arr,
                "outputs": output_arr if output_arr is not None else None,
                "labels": labels
            }
            if self.load:
                self._batch[index] = data

        return data


class DummyDataset(Dataset):
    """ Class to load a toy dataset in a mini-batch fashion.
    """

    def __init__(self, nb_batch, batch_size, shape, verbose=0):
        """ Initialize the class.

        Parameters
        ----------
        nb_batch: int
            the number of batch.
        batch_size: int
            the size of each mini-batch.
        outputs: list of str, default None
            the name of the column(s) containing the ouputs.
        shape: nuplet
            the generated data shape.
        verbose: int, default 0
            the verbosity level.
        """
        self.nb_batch = nb_batch
        self.batch_size = batch_size
        self.shape = shape
        self.verbose = verbose

    def __len__(self):
        """ Get the number of mini-batch.

        Returns
        -------
        nb_batch: int
            the number of mini-batchs.
        """
        return self.nb_batch

    def __getitem__(self, index):
        """ Get the desired mini-batch.

        Parameters
        ----------
        index: int
            the index of the desired mini-batch.

        Returns
        -------
        data: dict
            a dictionary with the inputs, ouputs, and labels mini-batch arrays.
        """
        shape = [self.batch_size, 1] + list(self.shape)
        input_arr = np.random.random(shape).astype(np.single)
        output_arr = np.zeros(shape, dtype=np.single)
        labels = np.zeros((self.batch_size, ), dtype=int)
        if self.verbose > 0:
            print("-" * 50)
            print("Mini Batch: ", index)
            print("Mini Batch Size: ", self.batch_size)
            print("Inputs: ", input_arr.shape)
            print("Ouputs: ", output_arr.shape)
            print("Labels: ", labels.shape)
        data = {
            "inputs": input_arr,
            "outputs": output_arr if output_arr is not None else None,
            "labels": labels
        }
        return data 

