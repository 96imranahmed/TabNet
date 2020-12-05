import os
import sys
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("../../src/"))
from train import TabNet

data_params = {
    "data_path": "../data/adult_census.csv",
    "target": "target",
    "random_seed": 42,
    "model_save_dir": "../../runs/adult_census/",
    "model_save_name": "adult_census",
    "columns": [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "target",
    ],
    "categorical_variables": [
        "workclass",
        "education",
        "occupation",
        "marital-status",
        "relationship",
        "race",
        "sex",
        "native-country",
    ],
    "drop_cols": ["fnlwgt"],
}

train_params = {
    "batch_size": 8192,
    "run_self_supervised_training": False,
    "run_supervised_training": True,
    "early_stopping": True,
    "early_stopping_min_delta_pct": 0,
    "early_stopping_patience": 1000,
    "max_epochs_supervised": 4000,
    "max_epochs_self_supervised": 4000,
    "epoch_save_frequency": 150,
    "train_generator_shuffle": True,
    "train_generator_n_workers": 0,
    "epsilon": 1e-7,
    "learning_rate": 0.02,
    "learning_rate_decay_factor": 0.95,
    "learning_rate_decay_step_rate": 1000,
    "sparsity_regularization": 0.0001,
    "p_mask": 0.8,
}

model_params = {
    "categorical_variables": data_params["categorical_variables"],
    "n_steps": 5,
    "feat_transform_fc_dim": 16,
    "embedding_dim": 3,
    "discrete_outputs": True,
    "gamma": 1.5,
}

if __name__ == "__main__":
    data = pd.read_csv(
        data_params["data_path"], header=None, names=data_params["columns"]
    )

    X, y = (
        data[
            data.columns.difference([data_params["target"]] + data_params["drop_cols"])
        ],
        data[data_params["target"]],
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=data_params["random_seed"]
    )

    fc_tabnet_model = TabNet(model_params=model_params)
    fc_tabnet_model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        train_params=train_params,
        save_params={
            "model_name": data_params["model_save_name"],
            "tensorboard_folder": "../../runs/",
            "save_folder": data_params["model_save_dir"],
        },
    )
    save_file = fc_tabnet_model.model_save_path
    fc_tabnet_model = TabNet(save_file=save_file)

    fc_xgboost_model = XGBClassifier(n_estimators=1000)
    fc_xgboost_model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20
    )

    y_tabnet_val_pred = fc_tabnet_model.predict(X_val)
    print(
        "TabNet accuracy: {}".format(
            np.round((y_tabnet_val_pred == y_val).sum() / len(y_val), 3)
        )
    )