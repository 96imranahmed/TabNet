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
                    int(self.params["n_dims_d"] + self.params["n_dims_a"]),
                    self.params["n_input_dims"],
                )
                for i in range(self.params["n_steps"])
            ]
        )
        self.__output_fc = nn.Linear(
            self.params["n_dims_d"], self.params["n_output_dims"]
        )

        self.__batch_norm = nn.BatchNorm1d(
            self.params["n_input_dims"],
            momentum=self.params["___momentum"],
        )

    def forward(self, X_continuous, X_embedding, init_mask, mask_input=False):
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
                if key != -1
            ],
            dim=-1,
        )
        if mask_input:
            X = init_mask * X
        X_bn = self.__batch_norm(X)
        a_i_minus_1 = self.__feature_transformer_individual_base(
            self.__feature_transformer_shared(X_bn)
        )[..., self.params["n_dims_d"] :]
        p_i_minus_1 = (
            init_mask  # Can be initialized differently depending on training regime
        )
        gamma = self.params["gamma"] * torch.ones_like(X)

        step_wise_outputs = []
        step_wise_feature_reconstruction = []
        step_wise_masks = []

        for i in range(self.params["n_steps"]):
            mask_i = self.__attentive_transformer[i](p_i_minus_1, a_i_minus_1)
            feat_transform_i = self.__feature_transformer_individual[i](
                self.__feature_transformer_shared(mask_i * X_bn)
            )
            d_i, a_i = (
                feat_transform_i[..., : self.params["n_dims_d"]],
                feat_transform_i[..., self.params["n_dims_d"] :],
            )
            p_i_minus_1 = p_i_minus_1 * (gamma - mask_i)
            step_wise_masks.append(mask_i)
            step_wise_outputs.append(nn.functional.relu(d_i))
            step_wise_feature_reconstruction.append(
                self.__reconstruction_fc[i](feat_transform_i)
            )
            a_i_minus_1 = a_i

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
            self.params["n_input_dims"],
            (self.params["n_dims_a"] + self.params["n_dims_d"]) * 2,
        )
        self.bn_one = nn.BatchNorm1d(
            (self.params["n_dims_a"] + self.params["n_dims_d"]) * 2,
            momentum=self.params["batch_norm_momentum"],
        )
        self.fc_two = nn.Linear(
            (self.params["n_dims_a"] + self.params["n_dims_d"]),
            (self.params["n_dims_a"] + self.params["n_dims_d"]) * 2,
        )
        self.bn_two = nn.BatchNorm1d(
            (self.params["n_dims_a"] + self.params["n_dims_d"]) * 2,
            momentum=self.params["batch_norm_momentum"],
        )
        self.dropout = nn.Dropout(p=self.params["dropout_p"])

    def forward(self, X):
        X_slice_one = nn.functional.glu(self.bn_one(self.fc_one(X)))
        X_slice_two = nn.functional.glu(self.bn_two(self.fc_two(X_slice_one)))
        return self.dropout((X_slice_two + X_slice_one) * math.sqrt(0.5))


class IndividualFeatureTransformer(nn.Module):

    params = {}
    step_id = 0

    def __init__(self, step_id, **kwargs):
        super(IndividualFeatureTransformer, self).__init__()
        self.step_id = step_id
        self.params.update(kwargs)
        self.fc_one = nn.Linear(
            (self.params["n_dims_a"] + self.params["n_dims_d"]),
            (self.params["n_dims_a"] + self.params["n_dims_d"]) * 2,
        )
        self.bn_one = nn.BatchNorm1d(
            (self.params["n_dims_a"] + self.params["n_dims_d"]) * 2,
            momentum=self.params["batch_norm_momentum"],
        )
        self.fc_two = nn.Linear(
            (self.params["n_dims_a"] + self.params["n_dims_d"]),
            (self.params["n_dims_a"] + self.params["n_dims_d"]) * 2,
        )
        self.bn_two = nn.BatchNorm1d(
            (self.params["n_dims_a"] + self.params["n_dims_d"]) * 2,
            momentum=self.params["batch_norm_momentum"],
        )
        self.dropout = nn.Dropout(p=self.params["dropout_p"])

    def forward(self, X):
        X_slice_one = nn.functional.glu(self.bn_one(self.fc_one(X)))
        X_slice_one = self.dropout((X_slice_one + X) * math.sqrt(0.5))
        X_slice_two = nn.functional.glu(self.bn_two(self.fc_two(X_slice_one)))
        return (X_slice_one + X_slice_two) * math.sqrt(0.5)


class AttentiveTransformer(nn.Module):

    params = {}
    step_id = 0

    def __init__(self, step_id, **kwargs):
        super(AttentiveTransformer, self).__init__()
        self.step_id = step_id
        self.params.update(kwargs)

        self.fc = nn.Linear(self.params["n_dims_a"], self.params["n_input_dims"])
        self.bn = nn.BatchNorm1d(
            num_features=self.params["n_input_dims"],
            momentum=self.params["batch_norm_momentum"],
        )
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, p_i_prev, a_i_prev):
        return self.sparsemax(p_i_prev * self.bn(self.fc(a_i_prev)))