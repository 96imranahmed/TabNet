import torch.nn as nn
import torch
import os
import time
import numpy as np
from model.tabnet import TabNetModel
from torch.utils.tensorboard import SummaryWriter


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, output_mapping=None):
        self.X = torch.from_numpy(X).float()
        if output_mapping:
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
        random_idx = np.random.randint(0, len(self.X) - 1, size=n_samples)
        return self.X[random_idx, ...], self.y[random_idx, ...]


class EarlyStopping(object):
    # Implemented from: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
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


class TabNet(object):

    default_train_params = {
        "batch_size": 8192,
        "self_supervised_pre_training": False,
        "early_stopping": True,
        "early_stopping_min_delta_pct": 0,
        "early_stopping_patience": 20,
        "max_epochs_supervised": 500,
        "max_epochs_self_supervised": 500,
        "epoch_save_frequency": 100,
        "use_cuda": True,
        "train_generator_shuffle": True,
        "train_generator_n_workers": 0,
        "epsilon": 1e-7,
        "learning_rate": 0.01,
        "learning_rate_decay_factor": 0.95,
        "learning_rate_decay_step_rate": 1000,
        "sparsity_regularization": 0.0001,
        "p_mask": 0.3,
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
    discrete_target_mapping = None

    def __init__(self, model_params={}):
        self.model_params = self.default_model_params
        self.model_params.update(model_params)

    def create_tensorboard_writer(self, test_batch):
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

    def save_model(self, save_identifier=None, self_supervised=False):
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

    def generate_discrete_target_map(self, targets):
        uq_targets = np.unique(targets)
        return dict(zip(list(uq_targets), list(range(len(uq_targets)))))

    def train_routine(
        self, train_generator, val_generator=None, epochs=None, self_supervised=False
    ):
        # Define optimizer
        scheduler = None
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.train_params["learning_rate"]
        )
        if val_generator:
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
        step = 0
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
                self.save_model("epoch_{}".format(c_epoch), self_supervised)

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
        self.save_model("final", self_supervised)

    def __validation_reconstruct_loss(self, generator):
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
        return torch.stack(out_loss).sum()

    def __validation_predict(self, generator):
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
        return (y_pred_agg, y_pred_logits_agg, y_agg)

    def train(
        self, X_train, y_train, X_val=None, y_val=None, train_params={}, save_params={}
    ):
        self.save_params = self.default_save_params
        self.save_params.update(save_params)

        self.train_params = self.default_train_params
        self.train_params.update(train_params)

        use_cuda = torch.cuda.is_available() and self.train_params["use_cuda"]
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        if self.default_model_params["discrete_outputs"]:
            self.discrete_target_mapping = self.generate_discrete_target_map(y_train)

        # Build generators for train / validation
        train_data = Dataset(X_train, y_train, self.discrete_target_mapping)
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
            val_data = Dataset(X_val, y_val, self.discrete_target_mapping)
            val_generator = torch.utils.data.DataLoader(
                val_data,
                **{"batch_size": self.train_params["batch_size"], "shuffle": False}
            )

        # Update with correct dimensions
        self.model_params.update(
            {
                "n_input_dims": train_data.n_input_dims,
                "n_output_dims": train_data.n_output_dims,
            }
        )

        self.model = TabNetModel(**self.model_params)

        self.tensorboard_writer = self.create_tensorboard_writer(
            train_data.random_batch(self.train_params["batch_size"])[0]
        )

        if not self.train_params["self_supervised_pre_training"]:
            print("Training predictive model (WITHOUT self-supervised pre-training...)")
            self.train_routine(
                train_generator,
                val_generator,
                self.train_params["max_epochs_supervised"],
                False,
            )
        else:
            print("Training self-supervised model...")
            self.train_routine(
                train_generator,
                val_generator,
                self.train_params["max_epochs_self_supervised"],
                True,
            )
            print("Training predictive model (WITH self-supervised pre-training...)")
            self.train_routine(
                train_generator,
                val_generator,
                self.train_params["max_epochs_supervised"],
                False,
            )

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
