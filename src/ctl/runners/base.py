"""Module containing the Base Runner class for experiments"""

import os
import shutil as sh
import warnings
import datetime
import git
from tqdm.auto import tqdm
from animus import IExperiment
from animus.core import set_global_seed
import pandas as pd
from tabulate import tabulate
import numpy as np

# torch imports
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ctl imports
from ctl.train.procedures import setup_data
from ctl.train.early_stopping import EarlyStopping
from ctl.utils.general import makedir, makedir_overwrite, save_patch_diff
from ctl.utils.configs import setup_model, setup_train, save_config


class Base(IExperiment):
    """
    Base runner
    """

    def __init__(self, engine, config):
        super().__init__()

        self.engine = engine
        self.config = config
        self.mode = self.config["mode"]

        if "save_console_logs" in self.config["run"].keys():
            self.save_console_logs = self.config["run"]["save_console_logs"]
        else:
            self.save_console_logs = None
        self.batch_size = config["run"]["batch_size"]
        self.num_epochs = self.config["run"]["num_epochs"]
        self.workers = config["run"]["workers"]
        self.log_every = self.config["run"]["log_every"]
        self.checkpoint_every = self.config["run"]["checkpoint_every"]
        self.overwrite = self.config["run"]["overwrite"]
        self.save_dir = self.config["run"]["save_dir"]
        self.verbose = self.config["run"]["verbose"]
        self.experiment_name = self.config["run"]["experiment_name"]
        self.checkpoints_dir = self.config["run"]["checkpoints_dir"]
        self.save_checkpoints = self.config["run"]["save_checkpoints"]
        self.graphs_kwargs = self.config["run"]["graphs"]
        self.project_dir = self.config["run"]["project_dir"]
        self.interpolation = self.config["run"]["interpolation"]
        self.deterministic = self.config["run"]["deterministic"]
        self.seed = self.config["run"]["seed"]
        self.output = None
        self.path_to_checkpoints_dir = None  # should be str
        self.thresholds = config["model"]["thresholds"]

        # early stopping
        self.early_stopping = None

        if self.deterministic:
            set_global_seed(self.seed)

        self.path_to_logs_csv = ""  # path to save bect_epoch's metrics from experiment
        self._setup_exp()

    def _setup_exp(self):
        makedir(self.save_dir)

        path_to_experiment = os.path.join(self.save_dir, self.experiment_name)
        makedir_overwrite(path_to_experiment, self.overwrite)

        self.config["run"]["datetime"] = str(datetime.datetime.now())
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            self.config["run"]["git"]["current_hash_commit"] = sha
        except git.exc.InvalidGitRepositoryError:
            self.config["run"]["git"]["current_hash_commit"] = None
            warnings.warn(
                "InvalidGitRepositoryError: the target directory is not a git repository or does not exist."
            )

        # last_hash_commit = save_patch_diff(
        #     self.config["run"]["git"]["branch"],
        #     path_to_experiment=path_to_experiment,
        # )

        # self.config["run"]["git"]["hash_commit"] = last_hash_commit

    def _setup_data(self) -> None:

        try:
            datasets = setup_data(self.config)
        except ValueError as ex:
            self.exception = ex
            self._run_event("on_exception")

        self.datasets = {}
        for key, dataset in datasets.items():
            if dataset:
                self.datasets[key] = self.engine.prepare(
                    DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle=("train" in key),
                        num_workers=self.workers,
                        collate_fn=dataset.collate_fn,
                    )
                )
                if len(self.datasets[key].dataset) == 0:
                    self.exception = ValueError(f"Length of {key} data = 0")
                    self._run_event("on_exception")

        if self.mode == "train" and (
            "train" not in self.datasets or "valid" not in self.datasets
        ):
            raise ValueError(
                f"Experiment must contain `train` and `valid` dataloader, now {self.datasets.keys() = }"
            )
        if self.mode == "test" and (
            "valid" not in self.datasets and "test" not in self.datasets
        ):
            raise ValueError(
                f"Test experiment must contain `valid` or `test` dataloader, now {self.datasets.keys() = }"
            )

    def _setup_model(self) -> None:
        self.model = setup_model(self.config["model"], self.engine)

    def _setup_train(self) -> None:
        (self.criterion, self.optimizer, self.scheduler, self.metrics) = setup_train(
            self.config, self.model
        )
        self.prime_metric = self.config["run"]["prime_metric"]
        self.criterion, self.optimizer, self.scheduler = self.engine.prepare(
            self.criterion, self.optimizer, self.scheduler
        )

        self.path_to_checkpoints_dir = os.path.join(
            self.save_dir,
            self.experiment_name,
            self.checkpoints_dir,
        )

        makedir_overwrite(
            os.path.join(self.save_dir, self.experiment_name, self.project_dir),
            self.overwrite,
        )
        makedir_overwrite(
            os.path.join(
                self.save_dir,
                self.experiment_name,
                self.graphs_kwargs["dir_to_save"],
            ),
            self.overwrite,
        )
        if self.mode == "train":
            makedir_overwrite(
                self.path_to_checkpoints_dir,
                self.overwrite,
            )

    def on_dataset_start(self, exp: "IExperiment"):
        super().on_dataset_start(exp)
        self.total_loss = 0
        self.total_metrics = {"name": [], "value": []}

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        if self.config["run"]["early_stopping"]["use"]:
            self.early_stopping = EarlyStopping(
                track_metric=self.config["run"]["early_stopping"]["track_metric"],
                threshold=self.config["run"]["early_stopping"]["threshold"],
                patience=self.config["run"]["early_stopping"]["patience"],
            )
        self.best_epoch = 0
        self.best_metric = -np.inf
        self.engine.setup(self._local_rank, self._world_size)
        with self.engine.local_main_process_first():
            self._setup_data()
            self._setup_model()
            self._setup_train()
        self.writer = SummaryWriter(
            os.path.join(self.save_dir, self.experiment_name, self.project_dir)
        )
        if self.mode == "test":
            self.num_epochs = 1
            self.datasets.pop("train", None)

    def run_dataset(self) -> None:
        self.model.train(self.is_train_dataset)
        with torch.set_grad_enabled(self.is_train_dataset):
            for self.batch in tqdm(
                self.dataset,
                disable=(not self.engine.is_local_main_process or not self.verbose),
            ):
                self._run_event("on_batch_start")
                self.run_batch()
                self._run_event("on_batch_end")

    def on_dataset_end(self, exp: "IExperiment"):
        self.total_loss /= self.dataset_batch_step
        values = self.metrics.compute()
        for key, value in values.items():
            self.total_metrics["name"].append(key)
            self.total_metrics["value"].append(value)
        self.dataset_metrics = {"loss": self.total_loss, "metrics": self.total_metrics}
        # self.dataset_metrics = self.engine.mean_reduce_ddp_metrics(self.dataset_metrics)
        super().on_dataset_end(exp)

    def _log_every(self):
        self._log_writer()
        if self.mode == "train":
            self._save_model(
                os.path.join(
                    self.path_to_checkpoints_dir,
                    "epoch_{0}.pth".format(self.epoch_step),
                )
            )
            if self.verbose:
                self._log_console()
        elif self.mode == "test":
            if self.verbose:
                self._log_console_test()

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)

        if self.mode == "train":
            metric_diff = (
                self.epoch_metrics["valid"]["metrics"]["value"][self.prime_metric]
                - self.best_metric
            )

            if metric_diff > 0 or not hasattr(self, "best_checkpoint"):
                self.best_metric = self.epoch_metrics["valid"]["metrics"]["value"][
                    self.prime_metric
                ]
                self.best_epoch = self.epoch_step
                self.best_checkpoint = os.path.join(
                    self.path_to_checkpoints_dir,
                    "epoch_{0}.pth".format(self.best_epoch),
                )

        self._save_logs()

        if self.epoch_step % self.log_every == 0:
            self._log_every()
        self.scheduler.step(self.total_loss, self.total_metrics)

        # early stopping checker
        if self.config["run"]["early_stopping"]["use"] and self.early_stopping.check(
            self.total_loss, self.total_metrics
        ):
            self.need_early_stop = True
            print("Early stopping triggered!")

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self._save_config()
        if self.mode == "train":
            self._save_best_model()
        self.engine.cleanup()

    def run_epoch(self) -> None:
        for self.dataset_key, self.dataset in self.datasets.items():
            if self.dataset_key == "test" and self.mode == "train":
                continue
            self._run_event("on_dataset_start")
            self.run_dataset()
            self._run_event("on_dataset_end")

    def _run_local(self, local_rank: int = -1, world_size: int = 1) -> None:
        self._local_rank, self._world_size = local_rank, world_size
        self._run_event("on_experiment_start")
        self.run_experiment()
        self._run_event("on_experiment_end")

    def _run(self):
        self.engine.spawn(self._run_local)

    def _save_model(self, dest, entire=False):
        self.engine.wait_for_everyone()
        if entire:
            model_to_save = self.engine.unwrap_model(self.model)
        else:
            model_to_save = self.engine.unwrap_model(self.model).state_dict()
        torch.save(model_to_save, dest)

    def _save_best_model(self):
        best_chekpoint_path = self.best_checkpoint
        new_best_chekpoint_path = os.path.join(
            self.save_dir,
            self.experiment_name,
            f"epoch_{self.best_epoch}.pth",
        )
        sh.copy(best_chekpoint_path, new_best_chekpoint_path)
        if not self.save_checkpoints:
            sh.rmtree(
                self.path_to_checkpoints_dir,
            )

    def _save_config(self):
        save_config(
            self.config,
            os.path.join(self.save_dir, self.experiment_name, "config.yaml"),
            self.overwrite,
        )

    def _log_writer(self):
        for key in self.epoch_metrics.keys():
            self.writer.add_scalars(
                "loss",
                {"{0}".format(key): self.epoch_metrics[key]["loss"]},
                self.epoch_step,
            )
            for name, value in zip(
                self.epoch_metrics[key]["metrics"]["name"],
                self.epoch_metrics[key]["metrics"]["value"],
            ):
                self.writer.add_scalars(
                    name, {"{0}".format(key): value}, self.epoch_step
                )
            self.writer.add_scalar(
                "lr", self.optimizer.param_groups[0]["lr"], self.epoch_step
            )

    def _log_console_test(self):
        """
        +-----------+----------------------+
        | Parameter |  Valid      Test     |
        +===========+======================+
        |   loss    |  0.6789     0.6888   |
        +-----------+----------------------+
        |Precision* |  0.9867     0.9634   |
        |  Recall   |  0.9513     0.9125   |
        +-----------+----------------------+
        """
        print("+-----------+----------------------+")
        print("| Parameter |  Valid      Test     |")
        print("+===========+======================+")
        print(
            "|{0:^11.11}|  {1:<9.4G}  {2:<9.4G}|".format(
                "loss",
                self.epoch_metrics.get("valid", {}).get("loss", np.nan),
                self.epoch_metrics.get("test", {}).get("loss", np.nan),
            )
        )
        print("+-----------+----------------------+")

        key_name = list(self.epoch_metrics.keys())[0]
        n_values = len(self.epoch_metrics[key_name]["metrics"]["name"])

        valid_names = (
            self.epoch_metrics.get("valid", {}).get("metrics", {}).get("name", np.nan)
        )
        test_names = (
            self.epoch_metrics.get("test", {}).get("metrics", {}).get("name", np.nan)
        )
        names = valid_names if valid_names is not np.nan else test_names
        valid_values = (
            self.epoch_metrics.get("valid", {}).get("metrics", {}).get("value", np.nan)
        )
        test_values = (
            self.epoch_metrics.get("test", {}).get("metrics", {}).get("value", np.nan)
        )

        for i in range(n_values):
            valid_value = valid_values[i] if valid_values is not np.nan else np.nan
            test_value = test_values[i] if test_values is not np.nan else np.nan
            print(
                "|{0:^11.11}|  {1:<9.4G}  {2:<9.4G}|".format(
                    names[i] + ("*" if i == self.prime_metric else ""),
                    valid_value,
                    test_value,
                )
            )

        print("+-----------+----------------------+")
        # print("\033[14A")

    def _log_console(self):
        """
        +----------------------------------+
        | Epoch:   1/100       lr: 1.00E-04|
        | [/                             ] |
        +-----------+----------------------+
        | Parameter |  Train      Valid    |
        +===========+======================+
        |   loss    |  0.061      0.0382   |
        +-----------+----------------------+
        |  auroc*   |  0.6343     0.9954   |
        |  f1score  |  0.8293     0.961    |
        |specificity|  0.05263    0.913    |
        | precision |  0.7391     0.9487   |
        |  recall   |  0.9444     0.9737   |
        | accuracy  |  0.7123     0.9508   |
        +-----------+----------------------+
        """
        sleshs = "".join(
            ["/" for i in range(int(self.epoch_step / self.num_epochs * 30))]
        )
        spaces = "".join(
            [" " for i in range(30 - int(self.epoch_step / self.num_epochs * 30))]
        )
        print("+----------------------------------+")
        print(
            "| Epoch: {0:>3d}/{1:<3d}      lr: {2:<.2E} |".format(
                self.epoch_step,
                self.num_epochs,
                [group["lr"] for group in self.optimizer.param_groups][0],
            )
        )
        print("| [{0}{1}] |".format(sleshs, spaces))
        print("+-----------+----------------------+")
        print("| Parameter |  Train      Valid    |")
        print("+===========+======================+")
        print(
            "|{0:^11.11}|  {1:<9.4G}  {2:<9.4G}|".format(
                "loss",
                self.epoch_metrics["train"]["loss"],
                self.epoch_metrics["valid"]["loss"],
            )
        )
        print("+-----------+----------------------+")
        for k, (train_name, train_value, valid_name, valid_value) in enumerate(
            zip(
                self.epoch_metrics["train"]["metrics"]["name"],
                self.epoch_metrics["train"]["metrics"]["value"],
                self.epoch_metrics["valid"]["metrics"]["name"],
                self.epoch_metrics["valid"]["metrics"]["value"],
            )
        ):
            if train_name != valid_name:
                raise ValueError("Train metrics and valid metrics are not the same.")
            if k == self.prime_metric:
                train_name += "*"
            print(
                "|{0:^11.11}|  {1:<9.4G}  {2:<9.4G}|".format(
                    train_name, train_value, valid_value
                )
            )

        print("+-----------+----------------------+")
        # print("\033[14A")

    def _save_logs(self):
        self.path_to_logs_csv = os.path.join(
            self.save_dir, self.experiment_name, "logs.csv"
        )

        metrics = ["loss", "metrics"]
        names = []
        values = []

        for metric in metrics:
            for key in self.epoch_metrics:
                if metric in self.epoch_metrics[key]:
                    if metric == "metrics":
                        sub_names = self.epoch_metrics[key]["metrics"]["name"]
                        sub_values = self.epoch_metrics[key]["metrics"]["value"]

                        for k, (sub_name, sub_value) in enumerate(
                            zip(sub_names, sub_values)
                        ):
                            if k == self.prime_metric:
                                sub_name += "*"

                            names.append(f"{key}_{sub_name}")
                            values.append(sub_value)
                    else:
                        names.append(f"{key}_{metric}")
                        values.append(self.epoch_metrics[key][metric])

        try:
            pd_log = pd.read_csv(self.path_to_logs_csv, index_col=0)
        except FileNotFoundError:
            names_dict = {name: [] for name in names}
            pd_log = pd.DataFrame({**names_dict})

        # Use the loc method to add the new row to the DataFrame
        row_values = dict(zip(names, values))
        pd_log.loc[len(pd_log)] = row_values
        pd_log.to_csv(self.path_to_logs_csv)
