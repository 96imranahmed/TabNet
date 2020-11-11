import os
import sys
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from train import TabNet

sys.path.append(os.path.abspath("../../src/"))

data_params = {
    "data_path": "../data/covtype.csv",
    "target": "Cover_Type",
    "random_seed": 42,
    "model_save_dir": "../../runs/forest_cover/",
    "model_save_name": None,
}

if __name__ == "__main__":
    data = pd.read_csv(data_params["data_path"])

    X, y = (
        data[data.columns.difference([data_params["target"]])].values,
        data[data_params["target"]].values,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=data_params["random_seed"]
    )

    fc_tabnet_model = TabNet(model_params={"discrete_outputs": True})
    fc_tabnet_model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        train_params={},
        save_params={
            "model_name": "forest_cover",
            "tensorboard_folder": "../../runs/",
            "save_folder": "../../runs/forest_cover/",
        },
    )

    fc_xgboost_model = XGBClassifier(n_estimators=1000)
    fc_xgboost_model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20
    )