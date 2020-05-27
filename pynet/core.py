# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Core classes.
"""

# System import
import os
import pickle
from collections import OrderedDict

# Third party import
from torchvision import models
import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Package import
from pynet.datasets.core import DataManager
from pynet.utils import checkpoint
from pynet.history import History
from pynet.visualization import Visualizer
from pynet.observable import Observable
import pynet.metrics as mmetrics
from pynet.utils import reset_weights


class Base(Observable):
    """ Class to perform classification.
    """
    def __init__(self, optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, use_cuda=False,
                 pretrained=None, load_optimizer=True, **kwargs):
        """ Class instantiation.

        Observers will be notified, allowed signals are:
        - 'before_epoch'
        - 'after_epoch'

        Parameters
        ----------
        optimizer_name: str, default 'Adam'
            the name of the optimizer: see 'torch.optim' for a description
            of available optimizer.
        learning_rate: float, default 1e-3
            the optimizer learning rate.
        loss_name: str, default 'NLLLoss'
            the name of the loss: see 'torch.nn' for a description
            of available loss.
        metrics: list of str
            a list of extra metrics that will be computed.
        use_cuda: bool, default False
            wether to use GPU or CPU.
        pretrained: path, default None
            path to the pretrained model or weights.
        load_optimizer: boolean, default True
            if pretrained is set, whether to also load the optimizer's weights or not
        kwargs: dict
            specify directly a custom 'model', 'optimizer' or 'loss'. Can also
            be used to set specific optimizer parameters.
        """
        super().__init__(
            signals=["before_epoch", "after_epoch", "after_iteration"])
        self.optimizer = kwargs.get("optimizer")
        self.loss = kwargs.get("loss")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        for name in ("optimizer", "loss"):
            if name in kwargs:
                kwargs.pop(name)
        if "model" in kwargs:
            self.model = kwargs.pop("model")
        if self.optimizer is None:
            if optimizer_name not in dir(torch.optim):
                raise ValueError("Optimizer '{0}' uknown: check available "
                                 "optimizer in 'pytorch.optim'.")
            self.optimizer = getattr(torch.optim, optimizer_name)(
                self.model.parameters(),
                lr=learning_rate,
                **kwargs)
        if self.loss is None:
            if loss_name not in dir(torch.nn):
                raise ValueError("Loss '{0}' uknown: check available loss in "
                                 "'pytorch.nn'.")
            self.loss = getattr(torch.nn, loss_name)()
        self.metrics = {}
        for name in (metrics or []):
            if name not in mmetrics.METRICS:
                raise ValueError("Metric '{0}' not yet supported: you can try "
                                 "to fill the 'METRICS' factory, or ask for "
                                 "some help!".format(name))
            self.metrics[name] = mmetrics.METRICS[name]
        if use_cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: unset 'use_cuda' parameter.")
        if pretrained is not None:
            checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
            if hasattr(checkpoint, "state_dict"):
                self.model.load_state_dict(checkpoint.state_dict())
            elif isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    try:
                        self.model.load_state_dict(checkpoint["model"], strict=False)
                        print('Model loaded.', flush=True)
                    except BaseException as e:
                        print('Error while loading the model\'s weights: %s' % str(e), flush=True)
                
                if "optimizer" in checkpoint:
                    if load_optimizer:
                        try:
                            self.optimizer.load_state_dict(checkpoint["optimizer"])
                            for state in self.optimizer.state.values():
                                for k, v in state.items():
                                    if torch.is_tensor(v):
                                        state[k] = v.to(self.device)
                        except BaseException as e:
                            print('Error while loading the optimizer\'s weights: %s' % str(e))
                    else:
                        print("Warning: the optimizer's weights are not restored ! ", flush=True)
            else:
                self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)

    def training(self, managers, nb_epochs, checkpointdir=None, fold_index=None,
                 scheduler=None, with_validation=True, with_visualization=False,
                 nb_epochs_per_saving=1, exp_name=None, standard_optim=True):
        """ Train the model.

        Parameters
        ----------
        managers: a pynet DataManager or a list of DataManagers
            a manager containing the train and validation data.
        nb_epochs: int, default 100
            the number of epochs.
        checkpointdir: str, default None
            a destination folder where intermediate models/historues will be
            saved.
        fold_index: int, default None
            the index of the fold to use for the training, default use all the
            available folds.
        scheduler: torch.optim.lr_scheduler, default None
            a scheduler used to reduce the learning rate.
        with_validation: bool, default True
            if set use the validation dataset.
        with_visualization: bool, default False,
            whether it uses a visualizer that will plot the losses/metrics/images in a WebApp framework
            during the training process
        nb_epochs_per_saving: int, default 1,
            the number of epochs after which the model+optimizer's parameters are saved
        exp_name: str, default None
            the experience name that will be launched
        Returns
        -------
        train_history, valid_history: History
            the train/validation history.
        """
        if isinstance(managers, DataManager):
            managers = [managers]

        train_history = History(name="Train_%s"%(exp_name or ""))
        if with_validation is not None:
            valid_history = History(name="Validation_%s"%(exp_name or ""))
        else:
            valid_history = None
        train_visualizer, valid_visualizer = None, None
        if with_visualization:
            train_visualizer = Visualizer(train_history)
            if with_validation:
                valid_visualizer = Visualizer(valid_history, offset_win=10)
        print(self.loss)
        print(self.optimizer)
        folds = range(managers[0].number_of_folds)
        if fold_index is not None:
            folds = [fold_index]
        for fold in folds:
            loaders = [manager.get_dataloader(
                train=True,
                validation=True,
                fold_index=fold) for manager in managers]
            for epoch in range(nb_epochs):
                self.notify_observers("before_epoch", epoch=epoch, fold=fold)
                loss, values = self.train([loader.train for loader in loaders], train_history,
                                          train_visualizer, fold, epoch, standard_optim)
                train_history.summary()
                if scheduler is not None:
                    scheduler.step(loss)
                if checkpointdir is not None and (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) \
                        and epoch > 0:
                    checkpoint(
                        model=self.model,
                        epoch=epoch,
                        fold=fold,
                        outdir=checkpointdir,
                        name=exp_name,
                        optimizer=self.optimizer)
                    train_history.save(
                        outdir=checkpointdir,
                        epoch=epoch,
                        fold=fold)
                if with_validation:
                    _, _, _, loss, values = self.test([loader.validation for loader in loaders],
                                                standard_optim=standard_optim)
                    valid_history.log((fold, epoch), validation_loss=loss, **values)
                    valid_history.summary()
                    if valid_visualizer is not None:
                        valid_visualizer.refresh_current_metrics()
                    if checkpointdir is not None and (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) \
                            and epoch > 0:
                        valid_history.save(
                            outdir=checkpointdir,
                            epoch=epoch,
                            fold=fold)
                self.notify_observers("after_epoch", epoch=epoch, fold=fold)
            reset_weights(self.model)
        return train_history, valid_history

    def train(self, loaders, history=None, visualizer=None, fold=None, epoch=None, standard_optim=True):
        """ Train the model on the trained data.

        Parameters
        ----------
        loaders: a pytorch Dataloader or a list of Dataloaders

        Returns
        -------
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """
        if isinstance(loaders, DataLoader):
            loaders = [loaders]

        self.model.train()
        nb_batch = np.min([len(loader) for loader in loaders])
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")

        values = {}
        iteration = 0
        gen = zip(*loaders)

        if not standard_optim:
            loss, values = self.model(gen, pbar=pbar, visualizer=visualizer)
        else:
            losses = []
            for dataitems in gen:
                pbar.update()
                inputs = [dataitem.inputs.to(self.device) for dataitem in dataitems]
                list_targets = []
                for dataitem in dataitems:
                    _targets = []
                    for item in (dataitem.outputs, dataitem.labels):
                        if item is not None:
                            _targets.append(item.to(self.device))
                    if len(_targets) == 1:
                        _targets = _targets[0]
                    list_targets.append(_targets)
    
                self.optimizer.zero_grad()
                outputs = self.model(*inputs)
                batch_loss = self.loss(outputs, *list_targets)
                batch_loss.backward()
                losses.append(float(batch_loss))
                self.optimizer.step()
    
                for name, metric in self.metrics.items():
                    if name not in values:
                        values[name] = 0
                    values[name] += float(metric(outputs, *list_targets)) / nb_batch
    
                aux_losses = (self.model.get_aux_losses() if hasattr(self.model, 'get_aux_losses') else dict())
                aux_losses.update(self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())
    
                for name, aux_loss in aux_losses.items():
                    if name not in values:
                        values[name] = 0
                    values[name] += float(aux_loss) / nb_batch
                if iteration % 10 == 0:
                    if visualizer is not None:
                        visualizer.refresh_current_metrics()
                        if hasattr(self.model, "get_current_visuals"):
                            visuals = self.model.get_current_visuals()
                            visualizer.display_images(visuals, ncols=3)
                iteration += 1
            loss = np.mean(losses)
        if history is not None:
            history.log((fold, epoch), loss=loss, **values)
        pbar.close()
        return loss, values

    def testing(self, managers, with_logit=False, predict=False, with_visuals=False, saving_dir=None, exp_name=None,
                standard_optim=True):
        """ Evaluate the model.

        Parameters
        ----------
        manager: a pynet DataManager or a list of pynet DataManagers
            a manager containing the test data.
        with_logit: bool, default False
            apply a softmax to the result.
        predict: bool, default False
            take the argmax over the channels.
        with_visuals: bool, default False
            returns the visuals got from the model
        Returns
        -------
        y: array-like
            the predicted data.
        X: array-like
            the input data.
        y_true: array-like
            the true data if available.
        loss: float
            the value of the loss function if true data availble.
        values: dict
            the values of the metrics if true data availble.
        """
        if isinstance(managers, DataManager):
            managers = [managers]

        loaders = [manager.get_dataloader(test=True) for manager in managers]
        if with_visuals:
            y, y_true, X, loss, values, visuals = self.test(
                [loader.test for loader in loaders], with_logit=with_logit, predict=predict, with_visuals=with_visuals,
                standard_optim=standard_optim)
        else:
            y, y_true, X, loss, values = self.test(
                [loader.test for loader in loaders], with_logit=with_logit, predict=predict, with_visuals=with_visuals,
                standard_optim=standard_optim)

        if saving_dir is not None:
            with open(os.path.join(saving_dir, (exp_name or 'test')+'.pkl'), 'wb') as f:
                pickle.dump({'y_pred': y, 'y_true': y_true, 'loss': loss, 'metrics': values}, f)
        
        if with_visuals:
            return y, X, y_true, loss, values, visuals

        return y, X, y_true, loss, values

    def test(self, loaders, with_logit=False, predict=False, with_visuals=False, standard_optim=True):
        """ Evaluate the model on the test or validation data.

        Parameter
        ---------
        loader: a pytorch Dataset
            the data laoder.
        with_logit: bool, default False
            apply a softmax to the result.
        predict: bool, default False
            take the argmax over the channels.

        Returns
        -------
        y: array-like
            the predicted data.
        y_true: array-like
            the true data
        X: array_like
            the input data
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """
        if isinstance(loaders, DataLoader):
            loaders = [loaders]

        self.model.eval()
        nb_batch = np.min([len(loader) for loader in loaders])
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")
        loss = 0
        values = {}
        visuals = []

        with torch.no_grad():
            y, y_true, X = [], [], []
            outputs = None
            gen = zip(*loaders)
            if not standard_optim:
                loss, values, y, y_true, X = self.model(gen, pbar=pbar)
            else:
                for dataitems in gen:
                    pbar.update()
                    inputs = [dataitem.inputs.to(self.device) for dataitem in dataitems]
                    list_targets = []
                    for dataitem in dataitems:
                        targets = []
                        for item in (dataitem.outputs, dataitem.labels):
                            if item is not None:
                                targets.append(item.to(self.device))
                                y_true.extend(item.cpu().detach().numpy())
                        if len(targets) == 1:
                            targets = targets[0]
                        elif len(targets) == 0:
                            targets = None
                        if targets is not None:
                            list_targets.append(targets)

                        outputs = self.model(*inputs)
                        if with_visuals:
                            visuals.append(self.model.get_current_visuals())
                        if len(list_targets) > 0:
                            batch_loss = self.loss(outputs, *list_targets)
                            loss += float(batch_loss) / nb_batch

                    y.extend(outputs.cpu().detach().numpy())
                    for i in inputs:
                        X.extend(i.cpu().detach().numpy())

                    aux_losses = (self.model.get_aux_losses() if hasattr(self.model, 'get_aux_losses') else dict())
                    aux_losses.update(self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())
                    for name, aux_loss in aux_losses.items():
                        name += " on validation set"
                        if name not in values:
                            values[name] = 0
                        values[name] += aux_loss / nb_batch
                        
                # Now computes the metrics with (y, y_true)
                for name, metric in self.metrics.items():
                    name += " on validation set"
                    values[name] = metric(torch.tensor(y), torch.tensor(y_true))
            pbar.close()
            
            if len(visuals) > 0:
                visuals = np.concatenate(visuals, axis=0)
            try:
                if with_logit:
                    y = func.softmax(torch.tensor(y), dim=1).detach().cpu().numpy()
                if predict:
                    y = np.argmax(y, axis=1)
            except Exception as e:
                print(e)
        if with_visuals:
            return y, y_true, X, loss, values, visuals
        return y, y_true, X, loss, values
