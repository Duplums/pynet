from pynet.core import Base
from pynet.datasets.core import ArrayDataset, DataItem
from pynet.augmentation import *
from pynet.transforms import Crop
from tqdm import tqdm
import torch
import numpy as np

class SimCLRDataset(ArrayDataset):

    def __init__(self, *args, self_supervision=None, **kwargs):
        super().__init__(*args, self_supervision=self_supervision, **kwargs)

        # VERY IMPORTANT: What DA techniques will we use ?

        if self.self_supervision is None:
            compose_transforms = Transformer()
            #compose_transforms.register(add_swap, probability=0.5, num_iterations=20)
            compose_transforms.register(flip, probability=0.5, axis=0)
            compose_transforms.register(add_blur, probability=0.5, sigma=(0.1, 1))
            compose_transforms.register(add_ghosting, intensity=1, probability=0.5, axis=0)
            compose_transforms.register(add_motion, probability=0.5)
            compose_transforms.register(add_noise, sigma=(0.1, 1), probability=0.5)
            compose_transforms.register(add_spike, n_spikes=2, probability=0.5)
            compose_transforms.register(Crop((64, 64, 64), "random", resize=True), probability=0.5)
            #compose_transforms.register(add_biasfield, probability=0.1, coefficients=0.5)

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

        _inputs = self.inputs[self.indices[item]]
        _labels, _outputs = (None, None)

        if self.labels is not None:
            _labels = self.labels[self.indices[item]]
            for tf in self.labels_transforms:
                _labels = tf(_labels)

        # Apply the transformations to the data
        for tf in self.input_transforms:
            _inputs = tf(_inputs)

        # Now apply the self supervision twice to have 2 versions of the input
        if self.self_supervision is not None:
            _inputs_i = self.self_supervision(_inputs)
            _inputs_j = self.self_supervision(_inputs)
        _inputs = np.stack((_inputs_i, _inputs_j), axis=0)

        return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels)


class SimCLR(Base):

    def train(self, loader, history=None, visualizer=None, fold=None, epoch=None, **kwargs):
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
                    visualizer.display_images(inputs[:2,0,:], ncols=2, middle_slices=True)
            iteration += 1
            for name, metric in self.metrics.items():
                if name not in values:
                    values[name] = 0
                values[name] += float(metric(logits, target))/nb_batch

        loss = np.mean(losses)

        if history is not None:
            history.log((fold, epoch), loss=loss, **values)
        pbar.close()
        return loss, values

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
                    print()
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