import torch.nn as nn
import torch
import math
from sparsemax import Sparsemax


class TabNetModel(nn.Module):

    params = {}

    def __init__(self, **kwargs):
        super(TabNetModel, self).__init__()
        self.params.update(kwargs)

        if self.params["feat_transform_fc_dim"] % 2 != 0:
            raise ValueError("Fully connected dimension size must be even")

        self.feature_transformer_shared = SharedFeatureTransformer(**self.params)
        self.feature_transformer_individual_base = IndividualFeatureTransformer(
            -1, **self.params
        )
        self.feature_transformer_individual = nn.ModuleList(
            [
                IndividualFeatureTransformer(i, **self.params)
                for i in range(self.params["n_steps"])
            ]
        )
        self.attentive_transformer = nn.ModuleList(
            [
                AttentiveTransformer(i, **self.params)
                for i in range(self.params["n_steps"])
            ]
        )
        self.reconstruction_fc = nn.ModuleList(
            [
                nn.Linear(
                    int(self.params["feat_transform_fc_dim"] / 2),
                    self.params["n_input_dims"],
                )
                for i in range(self.params["n_steps"])
            ]
        )
        self.output_fc = nn.Linear(
            int(self.params["feat_transform_fc_dim"] / 2), self.params["n_output_dims"]
        )

    def forward(self, X, init_mask):
        if len(list(X.size())) != 2:
            raise ValueError(
                "Shape mismatch: expected order 2 tensor"
                ", got order {} tensor".format(len(list(X.size())))
            )
        X_bn = nn.BatchNorm1d(self.params["n_input_dims"])(X)
        X_bn_pp = self.feature_transformer_individual_base(
            self.feature_transformer_shared(X_bn)
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
            mask = self.attentive_transformer[i](feat_prev, p_prev)
            X_i_in = torch.mul(mask, X_bn)
            feat_prev = self.feature_transformer_individual[i](
                self.feature_transformer_shared(X_i_in)
            )
            p_prev = torch.mul(p_prev, gamma - mask)
            step_wise_masks.append(mask)
            step_wise_outputs.append(nn.functional.relu(feat_prev))
            step_wise_feature_reconstruction.append(
                self.reconstruction_fc[i](feat_prev)
            )

        reconstructions = torch.stack(step_wise_feature_reconstruction, dim=0).sum(
            dim=0, keepdim=False
        )

        logits = self.output_fc(
            torch.stack(step_wise_outputs, dim=0).sum(dim=0, keepdim=False)
        )

        return logits, reconstructions, tuple(step_wise_masks)


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