import os
import sys
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("../../src/"))
from train import TabNet

data_params = {
    "data_path": "../data/adult_census.csv",
    "target": "target",
    "random_seed": 999,
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
    "n_steps": 5,
    "n_dims_d": 32,
    "n_dims_a": 32,
    "batch_norm_momentum": 0.98,
    "dropout_p": 0.8,
    "embedding_dim": 2,
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
        X, y, test_size=0.2, random_state=data_params["random_seed"]
    )

    X_train_copy, X_val_copy, y_train_copy, y_val_copy = (
        X_train.copy(),
        X_val.copy(),
        y_train.copy(),
        y_val.copy(),
    )

    # Preprocess data for XGBoost
    for column in X_train_copy:
        if X_train_copy.dtypes[column] == np.object:
            enc = LabelEncoder()
            X_train_copy[column] = enc.fit_transform(X_train_copy[column])
            X_val_copy[column] = enc.transform(X_val_copy[column])
    enc = LabelEncoder()
    y_train_copy = enc.fit_transform(y_train_copy)
    y_val_copy = enc.fit_transform(y_val_copy)

    # XGBoost training
    print("############# Training XGBoost model")
    ac_xgboost_model = XGBClassifier(n_estimators=100, verbosity=1).fit(
        X_train_copy,
        y_train_copy,
        eval_set=[(X_val_copy, y_val_copy)],
        early_stopping_rounds=20,
    )
    y_val_predict_xgb = ac_xgboost_model.predict(X_val_copy)

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