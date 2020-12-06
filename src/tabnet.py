import torch.nn as nn
import torch
import math
from collections import OrderedDict
from sparsemax import Sparsemax


class TabNetModel(nn.Module):
    """Module for TabNet architecture."""

    params = {}

    def __init__(self, **kwargs):
        super(TabNetModel, self).__init__()
        self.params.update(kwargs)

        if self.params["feat_transform_fc_dim"] % 2 != 0:
            raise ValueError("Fully connected dimension size must be even")

        self.__embedding_layers = nn.ModuleDict()
        for key, val in sorted(
            self.params["categorical_config"].items(), key=lambda k: k[1]["idx"]
        ):
            self.__embedding_layers[str(val["idx"])] = nn.Embedding(
                val["n_dims"], self.params["embedding_dim"]
            )

        self.__feature_transformer_shared = SharedFeatureTransformer(**self.params)
        self.__feature_transformer_individual_base = IndividualFeatureTransformer(
            -1, **self.params
        )
        self.__feature_transformer_individual = nn.ModuleList(
            [
                IndividualFeatureTransformer(i, **self.params)
                for i in range(self.params["n_steps"])
            ]
        )
        self.__attentive_transformer = nn.ModuleList(
            [
                AttentiveTransformer(i, **self.params)
                for i in range(self.params["n_steps"])
            ]
        )
        self.__reconstruction_fc = nn.ModuleList(
            [
                nn.Linear(
                    int(self.params["feat_transform_fc_dim"] / 2),
                    self.params["n_input_dims"],
                )
                for i in range(self.params["n_steps"])
            ]
        )
        self.__output_fc = nn.Linear(
            int(self.params["feat_transform_fc_dim"] / 2), self.params["n_output_dims"]
        )

    def forward(self, X_continuous, X_embedding, init_mask):
        if len(list(X_continuous.size())) != 2:
            raise ValueError(
                "Shape mismatch: expected order 2 tensor"
                ", got order {} tensor".format(len(list(X_continuous.size())))
            )
        X = torch.cat(
            [X_continuous]
            + [
                self.__embedding_layers[str(key)](val)
                for key, val in X_embedding.items()
            ],
            dim=-1,
        )
        X = X.mul(init_mask)  # Mask input data according to the provided mask
        X_bn = nn.BatchNorm1d(self.params["n_input_dims"])(X)
        X_bn_pp = self.__feature_transformer_individual_base(
            self.__feature_transformer_shared(X_bn)
        )

        feat_prev = X_bn_pp
        p_prev = (
            init_mask  # Can be initialized differently depending on training regime
        )
        gamma = self.params["gamma"] * torch.ones_like(X)

        step_wise_outputs = []
        step_wise_feature_reconstruction = []
        step_wise_masks = []

        for i in range(self.params["n_steps"]):
            mask = self.__attentive_transformer[i](feat_prev, p_prev)
            X_i_in = torch.mul(mask, X_bn)
            feat_prev = self.__feature_transformer_individual[i](
                self.__feature_transformer_shared(X_i_in)
            )
            p_prev = torch.mul(
                p_prev, gamma - mask
            ).data  # Prevent gradient calculation on p_prev
            step_wise_masks.append(mask)
            step_wise_outputs.append(nn.functional.relu(feat_prev))
            step_wise_feature_reconstruction.append(
                self.__reconstruction_fc[i](feat_prev)
            )

        reconstructions = torch.stack(step_wise_feature_reconstruction, dim=0).sum(
            dim=0, keepdim=False
        )

        logits = self.__output_fc(
            torch.stack(step_wise_outputs, dim=0).sum(dim=0, keepdim=False)
        )

        return X, logits, reconstructions, tuple(step_wise_masks)


class SharedFeatureTransformer(nn.Module):

    params = {}

    def __init__(self, **kwargs):
        super(SharedFeatureTransformer, self).__init__()
        self.params.update(kwargs)
        self.fc_one = nn.Linear(
            self.params["n_input_dims"], self.params["feat_transform_fc_dim"]
        )
        self.bn_one = nn.BatchNorm1d(self.params["feat_transform_fc_dim"])
        self.fc_two = nn.Linear(
            int(self.params["feat_transform_fc_dim"] / 2),
            self.params["feat_transform_fc_dim"],
        )
        self.bn_two = nn.BatchNorm1d(self.params["feat_transform_fc_dim"])
        pass

    def forward(self, X):
        X_slice_one = nn.functional.glu(self.bn_one(self.fc_one(X)))
        X_slice_two = nn.functional.glu(self.bn_two(self.fc_two(X_slice_one)))
        return torch.add(X_slice_two, X_slice_one).mul(math.sqrt(0.5))


class IndividualFeatureTransformer(nn.Module):

    params = {}
    step_id = 0

    def __init__(self, step_id, **kwargs):
        super(IndividualFeatureTransformer, self).__init__()
        self.step_id = step_id
        self.params.update(kwargs)
        self.fc_one = nn.Linear(
            int(self.params["feat_transform_fc_dim"] / 2),
            self.params["feat_transform_fc_dim"],
        )
        self.bn_one = nn.BatchNorm1d(self.params["feat_transform_fc_dim"])
        self.fc_two = nn.Linear(
            int(self.params["feat_transform_fc_dim"] / 2),
            self.params["feat_transform_fc_dim"],
        )
        self.bn_two = nn.BatchNorm1d(self.params["feat_transform_fc_dim"])

    def forward(self, X):
        X_slice_one = nn.functional.glu(self.bn_one(self.fc_one(X)))
        X_slice_one = torch.add(X_slice_one, X).mul(math.sqrt(0.5))
        X_slice_two = nn.functional.glu(self.bn_two(self.fc_two(X_slice_one)))
        return torch.add(X_slice_one, X_slice_two).mul(math.sqrt(0.5))


class AttentiveTransformer(nn.Module):

    params = {}
    step_id = 0

    def __init__(self, step_id, **kwargs):
        super(AttentiveTransformer, self).__init__()
        self.step_id = step_id
        self.params.update(kwargs)

        self.fc = nn.Linear(
            int(self.params["feat_transform_fc_dim"] / 2), self.params["n_input_dims"]
        )
        self.bn = nn.BatchNorm1d(self.params["n_input_dims"])
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, a_i_prev, p_i_prev):
        return self.sparsemax(torch.mul(p_i_prev, self.bn(self.fc(a_i_prev))))