import torch.nn as nn
import torch
import os
import time
import numpy as np
import pandas as pd
import warnings
from utils import TrainingDataset, InferenceDataset, EarlyStopping
from tabnet import TabNetModel
from torch.utils.tensorboard import SummaryWriter


class TabNet(object):

    default_train_params = {
        "batch_size": 8192,
        "run_self_supervised_training": False,
        "run_supervised_training": True,
        "early_stopping": True,
        "early_stopping_min_delta_pct": 0,
        "early_stopping_patience": 20,
        "max_epochs_supervised": 500,
        "max_epochs_self_supervised": 500,
        "epoch_save_frequency": 100,
        "train_generator_shuffle": True,
        "train_generator_n_workers": 0,
        "epsilon": 1e-7,
        "learning_rate": 0.01,
        "learning_rate_decay_factor": 0.95,
        "learning_rate_decay_step_rate": 1000,
        "sparsity_regularization": 0.0001,
        "p_mask": 0.8,
    }
    default_save_params = {
        "model_name": "forest_cover",
        "tensorboard_folder": "../runs/",
        "save_folder": "../runs/model_backups/",
    }
    default_model_params = {
        "n_steps": 5,
        "feat_transform_fc_dim": 128,
        "discrete_outputs": False,
        "gamma": 1.5,
    }
    model_save_path = None
    model = None

    def __init__(self, model_params={}, use_cuda=True, save_file=None):
        self.__configure_device(use_cuda)

        c_model_save = self.__load_model(save_file)
        if c_model_save:
            self.model_params, self.model = c_model_save
        else:
            self.model_params = self.default_model_params
            self.model_params.update(model_params)

    def __get_tensorboard_writer(self, test_batch):
        """Creates a tensorboard writer object, and pre-populates the model
        graph on Tensorboard with a test_batch.

        Parameters
        ----------
        test_batch : a test batch to pre-populate the Tensorboard graph

        Returns
        -------
        writer : torch.utils.tensorboard.SummaryWriter object
        """
        if not os.path.isdir(self.save_params["tensorboard_folder"]):
            os.mkdir(self.save_params["tensorboard_folder"])

        writer = SummaryWriter(
            "{}/{}_{}.run".format(
                self.save_params["tensorboard_folder"],
                int(time.time()),
                self.save_params["model_name"],
            )
        )
        writer.add_graph(
            self.model,
            [test_batch, torch.ones_like(test_batch)],
        )
        return writer

    def __load_model(self, save_file):
        """Loads model from a local file.

        Parameters
        ----------
        save_file : the path to the model save file

        Returns
        -------
        model_params : Model parameters
        model : Model initialized with saved weights
        """
        if save_file is None:
            return
        if not os.path.isfile(save_file):
            print(
                "File {} not found. Run `train` to configure a new model".format(
                    save_file
                )
            )
            return
        try:
            model_params, model_state_dict = torch.load(
                save_file, map_location=self.device
            )
            model = TabNetModel(**model_params)
            model.load_state_dict(model_state_dict)
            return model_params, model
        except:
            print(
                "File {} not correctly formatted. Run `train` to configure a new model".format(
                    save_file
                )
            )
            return

    def __save_model(self, save_identifier=None, self_supervised=False):
        """Saves current model parameters to a local file.

        Parameters
        ----------
        save_identifier : the model identifier to be used in the savefile name
        self_supervised : flag to identify whether the model has only been
        trained with self-supervision

        Returns
        -------
        None : Model is saved to disk
        """
        if not os.path.isdir(self.save_params["save_folder"]):
            os.mkdir(self.save_params["save_folder"])

        model_stub = "predictive_model"
        if self_supervised:
            model_stub = "self_supervised_model"

        self.model_save_path = "{}/{}_{}_{}_{}.pt".format(
            self.save_params["save_folder"],
            int(time.time()),
            self.save_params["model_name"],
            model_stub,
            save_identifier,
        )
        print("Saving model to: {}".format(self.model_save_path))
        torch.save((self.model_params, self.model.state_dict()), self.model_save_path)

    def __configure_device(self, use_cuda):
        """Configures torch device, based on configured properties and
        available devices."""
        use_cuda_available = torch.cuda.is_available() and use_cuda
        if use_cuda and use_cuda != use_cuda_available:
            print("Device configuration: Cuda not available - check GPU configuration.")
        self.device = torch.device("cuda:0" if use_cuda_available else "cpu")
        print(
            "Device configuration: using {} for training/inference".format(self.device)
        )
        torch.backends.cudnn.benchmark = True  # Cudnn configuration

    def __get_discrete_target_map(self, targets):
        """Generates mapping from discrete targets to ordinal output.

        For example: {"Dog": 0, "Cat": 1, "Human": 2}
        """
        uq_targets = np.unique(targets)
        return dict(zip(list(uq_targets), list(range(len(uq_targets)))))

    def __recover_categorical_predictions(self, ordinal_predictions):
        """Remaps ordinal_predictions back to their original categories."""
        if isinstance(ordinal_predictions, torch.Tensor):
            ordinal_predictions = ordinal_predictions.numpy()
        elif isinstance(ordinal_predictions, list):
            ordinal_predictions = np.array(ordinal_predictions)
        inv_target_mapping = {
            v: k for k, v in self.model_params["discrete_target_mapping"].items()
        }
        return np.vectorize(inv_target_mapping.get)(ordinal_predictions).squeeze()

    def __train(
        self,
        train_generator,
        val_generator=None,
        epochs=None,
        self_supervised=False,
        step_offset=0,
    ):
        """Internal function to fit a TabNet model to an input dataset, with a
        provided number of epochs. Supports early stopping (configured through
        model training parameters at the top of this file).

        If no validation dataset is provided, no validation metrics will be logged
        and early stopping functionality will not be available.

        Parameters
        ----------
        train_generator : torch.utils.DataLoader generator for the training dataset
        val_generator : torch.utils.DataLoader generator for the validation dataset
        epochs : the maximum number of training epochs
        self_supervised : flag to use a self-supervision training objective, as
        opposed to a prediction training objective
        step_offset : the step offset to start logging from

        Returns
        -------
        step : The number of steps the model was trained for
        """
        # Define optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.train_params["learning_rate"]
        )
        lambda_scale = lambda epoch: self.train_params["learning_rate_decay_factor"]
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambda_scale
        )

        # Define predictive criterion
        criterion = None
        if self.model_params["discrete_outputs"]:
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()

        # Define logging criterion
        print_stub_name = "Predictive"
        if self_supervised:
            print_stub_name = "Self-supervised"

        # Define early stopping
        es_tracker = None
        criterion_val_loss = None
        if self.train_params["early_stopping"]:
            es_tracker = EarlyStopping(
                min_delta=self.train_params["early_stopping_min_delta_pct"],
                patience=self.train_params["early_stopping_patience"],
                percentage=True,
            )
        # Training
        step = step_offset
        for c_epoch in range(epochs):
            loss_avg = []
            for batch_idx, (x_batch, y_batch) in enumerate(train_generator):
                step += 1
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                criterion_loss, reconstruction_loss, sparsity_loss = None, None, None

                ones_mask = torch.ones_like(x_batch)
                if self_supervised:
                    self_supervised_mask = torch.bernoulli(
                        ones_mask * self.train_params["p_mask"]
                    )

                    y_pred_logits, x_reconstruct_batch, masks = self.model(
                        x_batch.mul(ones_mask - self_supervised_mask),
                        ones_mask - self_supervised_mask,
                    )

                    # Define reconstruction loss
                    std = torch.std(x_batch, dim=0)
                    std[
                        std == 0
                    ] = 1  # Correct for cases where std_dev along dimension 0
                    reconstruction_loss = torch.norm(
                        self_supervised_mask * (x_reconstruct_batch - x_batch) / std
                    ).sum()
                    self.tensorboard_writer.add_scalar(
                        "{}/Reconstruction loss".format(print_stub_name),
                        reconstruction_loss,
                        step,
                    )
                else:
                    y_pred_logits, x_reconstruct_batch, masks = self.model(
                        x_batch, ones_mask
                    )

                    # Define criterion loss
                    y_batch_criterion = y_batch
                    if self.model_params["discrete_outputs"]:
                        y_batch_criterion = torch.squeeze(y_batch_criterion)
                    criterion_loss = criterion(y_pred_logits, y_batch_criterion)
                    self.tensorboard_writer.add_scalar(
                        "{}/Criterion loss".format(print_stub_name),
                        criterion_loss,
                        step,
                    )

                # Define sparsity loss
                sparsity_loss = (
                    -1
                    / (self.train_params["batch_size"] * self.model_params["n_steps"])
                    * torch.stack(
                        [
                            torch.mul(
                                c_mask,
                                torch.log(c_mask + self.train_params["epsilon"]),
                            ).sum()
                            for c_mask in masks
                        ]
                    ).sum()
                )
                self.tensorboard_writer.add_scalar(
                    "{}/Sparsity loss".format(print_stub_name),
                    sparsity_loss,
                    step,
                )

                loss = self.train_params["sparsity_regularization"] * sparsity_loss
                if self_supervised:
                    loss += reconstruction_loss  # Only optimise reconstruction in self-supervised regime
                else:
                    loss += criterion_loss  # Only optimise target criterion in prediction phase

                # Store in loss buffer
                loss_avg.append(loss)

                self.tensorboard_writer.add_scalar(
                    "{}/Total loss".format(print_stub_name),
                    loss,
                    step,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % self.train_params["learning_rate_decay_step_rate"] == 0:
                    scheduler.step()
                    print(
                        "Decaying learning rate. Revised learning rate: {}".format(
                            scheduler.get_last_lr()[0].item()
                        )
                    )

            if (c_epoch > 0) and (
                c_epoch % self.train_params["epoch_save_frequency"] == 0
            ):
                self.__save_model("epoch_{}".format(c_epoch), self_supervised)

            if val_generator is not None:
                if self_supervised:
                    reconstruction_val_loss = self.__validation_reconstruct_loss(
                        val_generator
                    )
                    self.tensorboard_writer.add_scalar(
                        "{}/Validation reconstruction loss".format(print_stub_name),
                        reconstruction_val_loss,
                        step,
                    )
                    print(
                        "{} - Epoch: {}, Step: {}, Loss: {}, {}: {}".format(
                            print_stub_name,
                            c_epoch + 1,
                            step,
                            torch.mean(torch.stack(loss_avg)),
                            "Validation reconstruction loss",
                            np.round(reconstruction_val_loss.item(), 4),
                        )
                    )
                else:
                    y_val_pred, y_val_pred_logits, y_val = self.__validation_predict(
                        val_generator
                    )
                    metric_value, metric_name = None, None
                    if self.model_params["discrete_outputs"]:
                        metric_name = "Validation accuracy"
                        metric_value = torch.true_divide(
                            (y_val_pred == y_val).sum(), y_val.size(0)
                        )
                    else:
                        metric_name = "Validation MSE"
                        metric_value = torch.square(y_val_pred - y_val).mean()

                    criterion_val_loss = criterion(y_val_pred_logits, y_val.squeeze())

                    self.tensorboard_writer.add_scalar(
                        "{}/{}".format(print_stub_name, metric_name),
                        metric_value,
                        step,
                    )
                    self.tensorboard_writer.add_scalar(
                        "{}/Validation criterion loss".format(print_stub_name),
                        criterion_val_loss,
                        step,
                    )
                    print(
                        "{} - Epoch: {}, Step: {}, Total train loss: {}, {}: {}, {}: {}".format(
                            print_stub_name,
                            c_epoch + 1,
                            step,
                            np.round(torch.mean(torch.stack(loss_avg)).item(), 4),
                            "Validation criterion loss",
                            np.round(criterion_val_loss.item(), 4),
                            metric_name,
                            np.round(metric_value.item(), 4),
                        )
                    )
                    if self.train_params["early_stopping"] and es_tracker.step(
                        criterion_val_loss
                    ):
                        print("Early stopping criterion met - ending training...")
                        break
            else:
                print(
                    "{} - Epoch: {}, Step: {}, Total train loss: {}".format(
                        print_stub_name,
                        c_epoch + 1,
                        step,
                        np.round(torch.mean(torch.stack(loss_avg)).item(), 4),
                    )
                )
        self.__save_model("final", self_supervised)
        return step

    def __validation_reconstruct_loss(self, generator):
        """Returns model reconstruction loss for an input dataset.

        Parameters
        ----------
        generator : torch.utils.DataLoader generator for the dataset

        Returns
        -------
        loss : Reconstruction loss across the full input dataset
        """
        self.model.eval()
        with torch.no_grad():
            out_loss = []
            for batch_idx, (x_batch, y_batch) in enumerate(generator):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                ones_mask = torch.ones_like(x_batch)
                self_supervised_mask = torch.bernoulli(
                    ones_mask * self.train_params["p_mask"]
                )
                y_pred_logits, x_reconstruct_batch, masks = self.model(
                    x_batch.mul(ones_mask - self_supervised_mask),
                    ones_mask - self_supervised_mask,
                )
                # Define reconstruction loss
                std = torch.std(x_batch, dim=0)
                std[std == 0] = 1  # Correct for cases where std_dev along dimension 0
                out_loss.append(
                    torch.norm(
                        self_supervised_mask * (x_reconstruct_batch - x_batch) / std
                    ).sum()
                )
        self.model.train()
        return torch.stack(out_loss).mean()

    def __validation_predict(self, generator):
        """Returns prediction probabilities, logits, and targets for an input
        dataset. Typically used to produce predictions for the validation set.

        Parameters
        ----------
        generator : torch.utils.DataLoader generator for the dataset

        Returns
        -------
        y_pred_agg : Model predictions probabilities for the input dataset
        y_pred_logits_agg: Model prediction logits for the input dataset
        """
        self.model.eval()
        with torch.no_grad():
            out_y = []
            out_y_pred_logits = []
            out_y_pred = []
            for batch_idx, (x_batch, y_batch) in enumerate(generator):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                ones_mask = torch.ones_like(x_batch)
                y_val_pred = None
                y_val_pred_logits, x_val_reconstruct_batch, masks = self.model(
                    x_batch, ones_mask
                )
                if self.model_params["discrete_outputs"]:
                    y_val_pred = torch.argmax(
                        nn.functional.softmax(y_val_pred_logits, dim=-1), dim=-1
                    )
                else:
                    y_val_pred = y_val_pred_logits.squeeze()
                out_y_pred_logits.append(y_val_pred_logits)
                out_y_pred.append(y_val_pred)
                out_y.append(y_batch)
            y_pred_agg = torch.cat(out_y_pred, dim=0).squeeze()
            y_pred_logits_agg = torch.cat(out_y_pred_logits, dim=0).squeeze()
            y_agg = torch.cat(out_y, dim=0).squeeze()
        self.model.train()
        return (y_pred_agg, y_pred_logits_agg, y_agg)

    def fit(
        self, X_train, y_train, X_val=None, y_val=None, train_params={}, save_params={}
    ):
        """Prepares and runs full train loop, for a given set of training and
        saving parameters. If a validation set is not provided, validation
        metrics will not be provided during training, and early stopping
        functionality will not be available.

        Parameters
        ----------
        X_train : Numpy array of training features
        y_train : Numpy array of training targets
        X_val : (Optional) Numpy array of validation features
        y_val : (Optional) Numpy array of validation targets
        train_params : A dictionary of training parameters
        save_params: A dictionary of save params

        Returns
        -------
        None
        """
        self.save_params = self.default_save_params
        self.save_params.update(save_params)

        self.train_params = self.default_train_params
        self.train_params.update(train_params)

        if self.model is None and self.model_params["discrete_outputs"]:
            self.model_params[
                "discrete_target_mapping"
            ] = self.__get_discrete_target_map(y_train)

        # Build generators for train / validation
        train_data = TrainingDataset(
            X_train, y_train, self.model_params["discrete_target_mapping"]
        )
        train_generator = torch.utils.data.DataLoader(
            train_data,
            **{
                "batch_size": self.train_params["batch_size"],
                "shuffle": self.train_params["train_generator_shuffle"],
                "num_workers": self.train_params["train_generator_n_workers"],
            }
        )
        val_data = None
        val_generator = None
        if X_val is not None:
            val_data = TrainingDataset(
                X_val, y_val, self.model_params["discrete_target_mapping"]
            )
            val_generator = torch.utils.data.DataLoader(
                val_data,
                **{"batch_size": self.train_params["batch_size"], "shuffle": False}
            )

        # Update with correct dimensions
        if self.model is None:
            self.model_params.update(
                {
                    "n_input_dims": train_data.n_input_dims,
                    "n_output_dims": train_data.n_output_dims,
                }
            )
            self.model = TabNetModel(**self.model_params)

        self.tensorboard_writer = self.__get_tensorboard_writer(
            train_data.random_batch(self.train_params["batch_size"])[0]
        )
        self.model.train()  # Enable training mode

        if not (
            self.train_params["run_self_supervised_training"]
            and self.train_params["run_supervised_training"]
        ):
            raise ValueError(
                "No training scheme defined: set `run_self_supervised_training` or `run_supervised_training` to True"
            )

        step = 0
        if not self.train_params["run_self_supervised_training"]:
            print("Training model with self-supervision objective")
            step = self.__train(
                train_generator=train_generator,
                val_generator=val_generator,
                epochs=self.train_params["max_epochs_self_supervised"],
                self_supervised=True,
                step_offset=step,
            )
        if not self.train_params["run_supervised_training"]:
            print("Training model with predictive objective self-supervised model...")
            step = self.__train(
                train_generator=train_generator,
                val_generator=val_generator,
                epochs=self.train_params["max_epochs_supervised"],
                self_supervised=False,
                step_offset=step,
            )

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()

    def __predict(self, X, batch_size=1024):
        """Internal method to produce predictions for provided input and
        batch_size features."""
        if not self.model:
            raise ValueError(
                "Model not yet initialized. Run `train` to fit a model to an input dataset"
            )

        self.model.eval()
        with torch.no_grad():
            pred_data = InferenceDataset(X)
            pred_generator = torch.utils.data.DataLoader(
                pred_data, **{"batch_size": batch_size, "shuffle": False}
            )

            out_y_pred = []
            for batch_idx, x_batch in enumerate(pred_generator):
                x_batch = x_batch.to(self.device)
                ones_mask = torch.ones_like(x_batch)
                y_val_pred = None
                y_val_pred_logits, x_val_reconstruct_batch, masks = self.model(
                    x_batch, ones_mask
                )
                if self.model_params["discrete_outputs"]:
                    y_val_pred = nn.functional.softmax(y_val_pred_logits, dim=-1)
                else:
                    y_val_pred = y_val_pred_logits.squeeze()
                out_y_pred.append(y_val_pred)
            y_pred_agg = torch.cat(out_y_pred, dim=0).squeeze()
        return y_pred_agg

    def predict_proba(self, X, batch_size=1024):
        """Uses model to predict for a given set of input features.

        Returns probability distribution for classification tasks.
        Raises ValueError if `discrete_outputs` is set to False.

        Parameters
        ----------
        X : Numpy array of features to predict on

        Returns
        -------
        y : Pandas array of predictions for input features, with column
        names corresponding to associated class names

        Raises
        ------
        ValueError: If model is not yet trained, or model is not a Classifier
        """
        if not self.model_params["discrete_outputs"]:
            raise ValueError("`Predict_proba` not available for regression models")
        else:
            ret_data = pd.DataFrame(self.__predict(X, batch_size=batch_size).numpy())
            ret_data.columns = self.__recover_categorical_predictions(
                np.array(ret_data.columns)
            )
            return ret_data

    def predict(self, X, batch_size=1024):
        """Uses model to predict for a given set of input features.

        Returns probability distribution for classification tasks.

        Parameters
        ----------
        X : Numpy array of features to predict on

        Returns
        -------
        y : List of predictions for input features
        """
        if self.model_params["discrete_outputs"]:
            return self.__recover_categorical_predictions(
                torch.argmax(self.__predict(X, batch_size=batch_size), dim=-1)
            )
        else:
            return self.__predict(X, batch_size=batch_size).numpy()
