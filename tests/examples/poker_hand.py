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
    "data_path_train": "../data/poker_hand_train.csv",
    "data_path_test": "../data/poker_hand_test.csv",
    "target": "target",
    "random_seed": 999,
    "model_save_dir": "../../runs/poker_hand/",
    "model_save_name": "poker_hand",
    "columns": ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "target"],
    "categorical_variables": [],
    "drop_cols": [],
}

train_params = {
    "batch_size": 4096,
    "run_self_supervised_training": False,
    "run_supervised_training": True,
    "early_stopping": True,
    "early_stopping_min_delta_pct": 0,
    "early_stopping_patience": 20,
    "max_epochs_supervised": 1500,
    "max_epochs_self_supervised": 1500,
    "epoch_save_frequency": 500,
    "train_generator_shuffle": True,
    "train_generator_n_workers": 0,
    "epsilon": 1e-7,
    "learning_rate": 0.02,
    "learning_rate_decay_factor": 0.4,
    "learning_rate_decay_step_rate": 2500,
    "weight_decay": 0.01,
    "sparsity_regularization": 0.0001,
    "p_mask": 0.2,
    "validation_batch_size": 128,
}

model_params = {
    "categorical_variables": data_params["categorical_variables"],
    "n_steps": 4,
    "n_dims_d": 16,
    "n_dims_a": 16,
    "batch_norm_momentum": 0.95,
    "dropout_p": 0.4,
    "embedding_dim": 1,
    "discrete_outputs": True,
    "gamma": 1.5,
}

if __name__ == "__main__":
    data_train = pd.read_csv(
        data_params["data_path_train"], header=None, names=data_params["columns"]
    )

    X_train, y_train = (
        data_train[
            data_train.columns.difference(
                [data_params["target"]] + data_params["drop_cols"]
            )
        ],
        data_train[data_params["target"]],
    )

    data_val = pd.read_csv(
        data_params["data_path_test"], header=None, names=data_params["columns"]
    )

    X_val, y_val = (
        data_val[
            data_val.columns.difference(
                [data_params["target"]] + data_params["drop_cols"]
            )
        ],
        data_val[data_params["target"]],
    )

    X_train_copy, X_val_copy, y_train_copy, y_val_copy = (
        X_train.copy(),
        X_val.copy(),
        y_train.copy(),
        y_val.copy(),
    )

    # XGBoost training
    # print("############# Training XGBoost model")
    # ac_xgboost_model = XGBClassifier(n_estimators=250, verbosity=1).fit(
    #     X_train_copy,
    #     y_train_copy,
    #     eval_set=[(X_val_copy, y_val_copy)],
    #     early_stopping_rounds=20,
    # )
    # y_val_predict_xgb = ac_xgboost_model.predict(X_val_copy)

    # TabNet training
    print("############# Training TabNet model")
    ac_tabnet_model = TabNet(model_params=model_params)
    ac_tabnet_model.fit(
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

    y_val_predict_tabnet = ac_tabnet_model.predict(X_val)

    print(
        "TabNet accuracy: {}".format(
            np.round((y_val_predict_tabnet == y_val).sum() / len(y_val), 3)
        )
    )
    print(
        "XGBoost accuracy: {}".format(
            np.round((y_val_predict_xgb == y_val_copy).sum() / len(y_val_copy), 3)
        )
    )