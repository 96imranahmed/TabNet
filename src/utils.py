import torch.nn as nn
import torch
import os
import time
import numpy as np
import pandas as pd
from collections import OrderedDict
from tabnet import TabNetModel
from torch.utils.tensorboard import SummaryWriter


class TrainingDataset(torch.utils.data.Dataset):
    """Creates a PyTorch Dataset object for a training dataset.

    Categorical targets in input are mapped internally to an ordinal
    number, as per supplied `output_mapping` dictionary.
    """

    is_categorical = False

    def __init__(self, X, y, output_mapping, categorical_mapping, columns, device=None):
        """Dataset initialization function.

        Parameters
        ----------
        X : Numpy array of input features
        y : Numpy array of input targets
        output_mapping : a mapping of categorical targets to ordinals
        categorical_mapping: mapping data to encode categorical variables
        columns: dataset columns identifiers
        """
        self.columns = columns
        self.device = device
        # Preprocess categoricals
        if categorical_mapping:
            X_slices = OrderedDict()
            for key, val in sorted(
                categorical_mapping.items(), key=lambda k: k[1]["idx"]
            ):
                X_slices[val["idx"]] = map_categoricals_to_ordinals(
                    X[:, val["idx"]], val["map"]
                )
            idx_slice = sorted([val["idx"] for key, val in categorical_mapping.items()])
            X_continuous = torch.from_numpy(
                np.delete(X, idx_slice, -1).astype(float)
            ).float()
            self.X = (X_continuous, X_slices)
        else:
            self.X = (torch.from_numpy(X).float(), OrderedDict())

        # Preprocess targets
        if output_mapping:
            self.y = map_categoricals_to_ordinals(y, output_mapping)
            self.n_output_dims = len(output_mapping.keys())
        else:
            self.y = torch.from_numpy(y.astype(float)).float()
            if len(self.y.size()) == 1:
                self.y = self.y.unsqueeze(-1)
            self.n_output_dims = list(self.y.size())[-1]

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, index):
        return (
            self.X[0][index, ...].to(self.device),
            OrderedDict(
                {key: self.X[1][key][index, ...].to(self.device) for key in self.X[1]}
            ),
            self.y[index, ...],
        )

    def random_batch(self, n_samples):
        """Generates a random batch of `n_samples` with replacement."""
        random_idx = np.random.randint(0, self.__len__() - 1, size=n_samples)
        return self.__getitem__(random_idx)


class InferenceDataset(torch.utils.data.Dataset):
    """Creates a PyTorch Dataset object for a set of points for inference."""

    def __init__(self, X, categorical_mapping=None, columns=None, device=None):
        """Dataset initialization function.

        Parameters
        ----------
        X : Numpy array of input features
        categorical_mapping: mapping data to encode categorical variables
        columns: dataset columns identifiers
        """
        self.columns = columns
        self.device = device

        # Preprocess categoricals
        if categorical_mapping:
            X_slices = OrderedDict()
            for key, val in sorted(
                categorical_mapping.items(), key=lambda k: k[1]["idx"]
            ):
                X_slices[val["idx"]] = map_categoricals_to_ordinals(
                    X[:, val["idx"]], val["map"]
                )
            idx_slice = sorted([val["idx"] for key, val in categorical_mapping.items()])
            X_continuous = torch.from_numpy(
                np.delete(X, idx_slice, -1).astype(float)
            ).float()
            self.X = (X_continuous, X_slices)
        else:
            self.X = (torch.from_numpy(X).float(), OrderedDict())

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, index):
        return (
            self.X[0][index, ...].to(self.device),
            OrderedDict(
                {key: self.X[1][key][index, ...].to(self.device) for key in self.X[1]}
            ),
        )


class EarlyStopping(object):
    """EarlyStopping class to end training early, once validation metrics stop
    improving.

    Implemented from: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    """

    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def generate_categorical_to_ordinal_map(inputs):
    """Generates mapping from discrete inputs to numbered outputs.

    For example: {"Dog": 0, "Cat": 1, "Human": 2}
    """
    if isinstance(inputs, pd.Series):
        inputs = inputs.values
    uq_inputs = np.unique(inputs)
    return dict(zip(list(uq_inputs), list(range(len(uq_inputs)))))


def map_categoricals_to_ordinals(categoricals, mapping):
    """Maps categoricals to their ordinals equivalents.

    Parameters
    ----------
    ordinals : Numpy array of input ordinals
    mapping : Dictionary of mapping of categoricals to ordinals

    Returns
    -------
    Torch tensor of ordinals
    """
    unmapped_targets = set(np.unique(categoricals).flatten()) - set(mapping.keys())
    if len(unmapped_targets) > 0:
        raise ValueError(
            "Mapping missing the following keys: {}".format(unmapped_targets)
        )
    return torch.from_numpy(
        np.vectorize(mapping.get)(categoricals).astype(float)
    ).long()


def map_categoricals_to_one_hot(categoricals, mapping):
    """Maps categoricals to their onehot equivalents.

    Parameters
    ----------
    ordinals : Numpy array of input ordinals
    mapping : Dictionary of mapping of categoricals to ordinals

    Returns
    -------
    Torch tensor of onehot encoded inputs
    """
    unmapped_elements = set(np.unique(categoricals).flatten()) - set(mapping.keys())
    if len(unmapped_elements) > 0:
        raise ValueError(
            "Mapping missing the following keys: {}".format(unmapped_elements)
        )
    return torch.from_numpy(
        np.squeeze(
            np.eye(len(mapping.keys()))[
                np.vectorize(mapping.get)(categoricals).reshape(-1)
            ]
        ).astype(float)
    ).long()


def map_ordinals_to_categoricals(ordinals, mapping):
    """Remaps ordinals back to their original categories.

    Parameters
    ----------
    ordinals : Numpy array / Torch tensor / List of input ordinals
    mapping : Dictionary of mapping of categoricals to ordinals

    Returns
    -------
    Numpy array of categoricals
    """
    if isinstance(ordinals, torch.Tensor):
        ordinals = ordinals.numpy()
    elif isinstance(ordinals, list):
        ordinals = np.array(ordinals)
    inv_target_mapping = {v: k for k, v in mapping.items()}
    return np.vectorize(inv_target_mapping.get)(ordinals).squeeze()
