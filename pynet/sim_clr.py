from pynet.core import Base
from pynet.datasets.core import ArrayDataset, DataItem
import torch
from torch.utils.data import SequentialSampler
from pynet.augmentation import *
from torchio.transforms import RandomAffine, RandomMotion
from pynet.transforms import Crop
import bisect
from tqdm import tqdm
import numpy as np

class SimCLRDataset(ArrayDataset):

    def __init__(self, *args, self_supervision=None, **kwargs):
        super().__init__(*args, self_supervision=self_supervision, **kwargs)

        # VERY IMPORTANT: What DA techniques will we use ?

        if self.self_supervision is None:
            compose_transforms = Transformer()
            # compose_transforms.register(add_swap, probability=0.5, num_iterations=20)
            # compose_transforms.register(flip, probability=0.5, axis=0)
            # compose_transforms.register(add_blur, probability=0.5, sigma=(0.1, 1))
            # compose_transforms.register(add_ghosting, intensity=1, probability=0.5, axis=0)
            # compose_transforms.register(RandomMotion(degrees=10, translation=10, num_transforms=2, p=1),
            #                            with_channel=True, probability=0.5)
            # compose_transforms.register(add_noise, sigma=(0.1, 1), probability=0.5)
            # compose_transforms.register(add_spike, n_spikes=2, probability=0.5)
            #compose_transforms.register(cutout, probability=1, patch_size=32, inplace=False)
            compose_transforms.register(Crop((64, 64, 64), "random", resize=True), probability=1)
            # compose_transforms.register(add_biasfield, probability=0.1, coefficients=0.5)
            # compose_transforms = RandomAffine(scales=1.0, degrees=40, translation=40, p=1)

            self.self_supervision = compose_transforms


    def __getitem__(self, item):
        """ Return the requested item.

             Returns
             -------
             item: namedtuple
                 a named tuple containing 'inputs', 'outputs', and 'labels' data.
             """

        if self.patch_size is not None:
            raise ValueError('Unexpected arg: patch size')
        if self.features_to_add is not None:
            raise ValueError('Unexpected arg: features_to_add')

        idx = self.indices[item]
        _outputs, _labels = None, None
        if self.concat_datasets:
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            sample_idx = idx - self.cumulative_sizes[dataset_idx-1] if dataset_idx > 0 else idx
            _inputs = self.inputs[dataset_idx][sample_idx]
        else:
            _inputs = self.inputs[idx]

        if self.labels is not None:
            _labels = self.labels[idx]
            for tf in self.labels_transforms:
                _labels = tf(_labels)

        # Apply the transformations to the data
        for tf in self.input_transforms:
            _inputs = tf(_inputs)

        # Now apply the self supervision twice to have 2 versions of the input
        if self.self_supervision is not None:
            np.random.seed()
            _inputs_i = self.self_supervision(_inputs)
            _inputs_j = self.self_supervision(_inputs)

        _inputs = np.stack((_inputs_i, _inputs_j), axis=0)

        return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels)

    def __str__(self):
        return '\nSimCLRDataset TF\n' + str(self.self_supervision)


class SimCLR(Base):

    def train(self, loader, visualizer=None, fold=None, epoch=None, **kwargs):
        """ Train the model on the trained data.

        Parameters
        ----------
        loader: a pytorch Dataloader

        Returns
        -------
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """

        self.model.train()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")

        values = {}
        iteration = 0

        losses = []
        y_pred, y_true = [], []
        for dataitem in loader:
            pbar.update()
            inputs = dataitem.inputs.to(self.device)
            labels = dataitem.labels.to(self.device) if dataitem.labels is not None else None
            self.optimizer.zero_grad()
            z_i = self.model(inputs[:,0,:])
            z_j = self.model(inputs[:,1,:])
            if labels is not None:
                batch_loss, logits, target = self.loss(z_i, z_j, labels)
            else:
                batch_loss, logits, target = self.loss(z_i, z_j)
            batch_loss.backward()
            self.optimizer.step()

            aux_losses = (self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())
            for name, aux_loss in aux_losses.items():
                if name not in values:
                    values[name] = 0
                values[name] += float(aux_loss) / nb_batch

            losses.append(float(batch_loss))
            y_true.extend(target.detach().cpu().numpy())
            y_pred.extend(logits.detach().cpu().numpy())
            if iteration % 10 == 0:
                if visualizer is not None:
                    visualizer.refresh_current_metrics()
                    visualizer.display_images(torch.cat((inputs[:2,0,:],inputs[:2,1,:])),
                                              ncols=2, middle_slices=True)
            iteration += 1
            for name, metric in self.metrics.items():
                if name not in values:
                    values[name] = 0
                values[name] += float(metric(logits, target))/nb_batch

        loss = np.mean(losses)

        pbar.close()
        return loss, values

    def features_avg_test(self, loader, M=10):
        """ Evaluate the model at test time using the feature averaging strategy as described in
        Improving Transformation Invariance in Contrastive Representation Learning, ICLR 2021, A. Foster

        :param
        loader: a pytorch Dataset
            the data loader.
        M: int, default 10
            nb of times we sample t~T such that we transform a sample x -> z := f(t(x))
        :returns
            y: array-like dims (n_samples, M, ...) where ... is the dims of the network's output
            the predicted data.
            y_true: array-like dims (n_samples, M,  ...) where ... is the dims of the network's output
            the true data
        """
        M = int(M)

        assert M//2 == M/2.0, "Nb of feature vectors averaged should be odd"

        if not isinstance(loader.sampler, SequentialSampler):
            raise ValueError("The dataloader must use the sequential sampler (avoid random_sampler option)")

        print(loader.dataset, flush=True)


        self.model.eval()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch*(M//2), desc="Mini-Batch")

        with torch.no_grad():
            y, y_true = [], []
            for _ in range(M//2):
                current_y, current_y_true = [[], []], []
                for dataitem in loader:
                    pbar.update()
                    inputs = dataitem.inputs.to(self.device)
                    if dataitem.labels is not None:
                        current_y_true.extend(dataitem.labels.cpu().detach().numpy())
                    z_i = self.model(inputs[:, 0, :])
                    z_j = self.model(inputs[:, 1, :])
                    current_y[0].extend(z_i.cpu().detach().numpy())
                    current_y[1].extend(z_j.cpu().detach().numpy())
                y.extend(current_y)
                y_true.extend([current_y_true, current_y_true])
            pbar.close()
            # Final dim: y [M, n_samples, ...] and y_true [M, n_samples, ...]
            # Sanity check
            assert np.all(np.array(y_true)[0,:] == np.array(y_true)), "Wrong iteration order through the dataloader"
            y = np.array(y).swapaxes(0, 1)
            y_true = np.array(y_true).swapaxes(0, 1)

        return y, y_true


    def test(self, loader, with_visuals=False, **kwargs):
        """ Evaluate the model on the validation data. The test is done in a usual way for a supervised task.

        Parameter
        ---------
        loader: a pytorch Dataset
            the data loader.

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

        self.model.eval()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")
        loss = 0
        values = {}
        visuals = []
        y, y_true, X = [], [], []

        with torch.no_grad():
            for dataitem in loader:
                pbar.update()
                inputs = dataitem.inputs.to(self.device)
                labels = dataitem.labels.to(self.device) if dataitem.labels is not None else None

                z_i = self.model(inputs[:, 0, :])
                z_j = self.model(inputs[:, 1, :])

                if with_visuals:
                    visuals.append(self.model.get_current_visuals())

                if labels is not None:
                    batch_loss, logits, target = self.loss(z_i, z_j, labels)
                else:
                    batch_loss, logits, target = self.loss(z_i, z_j)

                loss += float(batch_loss) / nb_batch
                y.extend(logits.detach().cpu().numpy())
                y_true.extend(target.detach().cpu().numpy())

                for i in inputs:
                    X.extend(i.cpu().detach().numpy())

                # Now computes the metrics with (y, y_true)
                for name, metric in self.metrics.items():
                    name += " on validation set"
                    if name not in values:
                        values[name] = 0
                    values[name] += metric(logits, target) / nb_batch
                aux_losses = (self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())
                for name, aux_loss in aux_losses.items():
                    name += " on validation set"
                    if name not in values:
                        values[name] = 0
                    values[name] += aux_loss / nb_batch

        pbar.close()

        if len(visuals) > 0:
            visuals = np.concatenate(visuals, axis=0)

        if with_visuals:
            return y, y_true, X, loss, values, visuals

        return y, y_true, X, loss, values