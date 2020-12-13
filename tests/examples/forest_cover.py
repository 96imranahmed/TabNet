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
    "data_path": "../data/covtype.csv",
    "target": "Cover_Type",
    "random_seed": 42,
    "model_save_dir": "../../runs/forest_cover/",
    "model_save_name": "forest_cover",
    "categorical_variables": [],
}

train_params = {
    "batch_size": 8192,
    "run_self_supervised_training": False,
    "run_supervised_training": True,
    "early_stopping": True,
    "early_stopping_min_delta_pct": 0,
    "early_stopping_patience": 5,
    "max_epochs_supervised": 5,
    "max_epochs_self_supervised": 4,
    "epoch_save_frequency": 5,
    "train_generator_shuffle": True,
    "train_generator_n_workers": 0,
    "epsilon": 1e-7,
    "learning_rate": 0.01,
    "learning_rate_decay_factor": 0.95,
    "learning_rate_decay_step_rate": 1000,
    "sparsity_regularization": 0.0001,
    "p_mask": 0.8,
}

model_params = {
    "discrete_outputs": True,
    "categorical_variables": data_params["categorical_variables"],
}

if __name__ == "__main__":
    data = pd.read_csv(data_params["data_path"])

    X, y = (
        data[data.columns.difference([data_params["target"]])].values,
        data[data_params["target"]].values,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=data_params["random_seed"]
    )

    y_train_copy, y_val_copy = (
        y_train.copy(),
        y_val.copy(),
    )
    enc = LabelEncoder()
    y_train_copy = enc.fit_transform(y_train_copy)
    y_val_copy = enc.fit_transform(y_val_copy)

    # XGBoost training
    print("############# Training XGBoost model")
    fc_xgboost_model = XGBClassifier(n_estimators=250, verbosity=1)
    fc_xgboost_model.fit(
        X_train,
        y_train_copy,
        eval_set=[(X_val, y_val_copy)],
        early_stopping_rounds=20,
    )
    y_val_predict_xgb = fc_xgboost_model.predict(X_val)

    print(
        "XGBoost accuracy: {}".format(
            np.round((y_val_predict_xgb == y_val_copy).sum() / len(y_val_copy), 3)
        )
    )

    # TabNet training
    print("############# Training TabNet model")
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
    fc_tabnet_model = TabNet(save_file=save_file)
    y_val_predict_tabnet = fc_tabnet_model.predict(X_val)

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