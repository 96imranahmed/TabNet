import torch.nn as nn
import torch
import os
import time
import numpy as np
from tabnet import TabNetModel
from torch.utils.tensorboard import SummaryWriter


class TrainingDataset(torch.utils.data.Dataset):
    """Creates a PyTorch Dataset object for a training dataset.

    Categorical targets in input are mapped internally to an ordinal
    number, as per supplied `output_mapping` dictionary.
    """

    def __init__(self, X, y, output_mapping=None):
        """Dataset initialization function.

        Parameters
        ----------
        X : Numpy array of input features
        y : Numpy array of input targets
        output_mapping : a mapping of categorical targets to ordinals
        """
        self.X = torch.from_numpy(X).float()
        if output_mapping:
            unmapped_targets = set(np.unique(y).flatten()) - set(output_mapping.keys())
            if len(unmapped_targets) > 0:
                raise ValueError(
                    "Dataset has unmapped targets: {}".format(unmapped_targets)
                )
            self.y = torch.from_numpy(np.vectorize(output_mapping.get)(y)).long()
        else:
            self.y = torch.from_numpy(y).float()
        if len(self.y.size()) == 1:
            self.y = self.y.unsqueeze(-1)
        self.n_input_dims = list(self.X.size())[-1]
        if output_mapping:
            self.n_output_dims = len(output_mapping.keys())
        else:
            self.n_output_dims = list(self.y.size())[-1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index, ...], self.y[index, ...]

    def random_batch(self, n_samples):
        """Generates a random batch of `n_samples` with replacement."""
        random_idx = np.random.randint(0, len(self.X) - 1, size=n_samples)
        return self.X[random_idx, ...], self.y[random_idx, ...]


class InferenceDataset(torch.utils.data.Dataset):
    """Creates a PyTorch Dataset object for a set of points for inference."""

    def __init__(self, X):
        """Dataset initialization function.

        Parameters
        ----------
        X : Numpy array of input features
        """
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index, ...]


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
