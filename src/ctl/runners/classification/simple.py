"""A Runner for the binary classification"""

import os
from tqdm import tqdm

from animus import IExperiment

# torch imports
import torch

# ctl imports
from ctl.runners.base import Base
from ctl.utils.graphs import (
    plot_roc_curve,
    plot_pr_curve,
    plot_histogram_distribution,
    plot_f1_surve,
)


class SimpleClassificationRunner(Base):
    """
    A runner class for simple classification tasks.

    This class is designed to handle the training and evaluation of a model for
    simple classification tasks. It extends the 'Base' class and implements the
    'run_batch' method, which processes a single batch of data, computes loss,
    and updates the model's parameters if it's a training dataset.

    Attributes
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model parameters during training.
    model : torch.nn.Module
        The classification model to be trained and evaluated.
    criterion : torch.nn.Module
        The loss criterion used to compute the training loss.
    metrics : Metrics
        A metrics object for tracking performance during training and evaluation.
    is_train_dataset : bool
        Indicates whether the dataset being processed is for training or evaluation.

    Methods
    -------
    run_batch() -> None
        Processes a single batch of data, computes loss, and updates model parameters
        if it's a training dataset.

    Note
    ----
    This class assumes a classification task where the model predicts class logits,
    and the labels are expected to be present in the batch[1] as 'labels'.
    """

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self.threshold_curve = self.graphs_kwargs["threshold_curve"]
        self.thresholds = []
        if not self.threshold_curve in ["pr", "rs"]:
            raise ValueError(
                "Threshold (for classififcation task) is choosen based on"
                + " only on Precision-Racall (pr) or Recall-Specificity (rs) curve."
                + f" Not {self.threshold_curve}."
            )

    def run_batch(self) -> None:
        self.optimizer.zero_grad()
        self.output = {"cls_logits": self.model(self.batch[0])}
        gt_labels = self.batch[1]["labels"]
        loss = self.criterion(self.output["cls_logits"], gt_labels)
        self.total_loss += loss.sum().item()
        self.metrics.add_batch(
            predictions=self.output,
            references=self.batch[1],
            keep_in_ram=not self.is_train_dataset,
        )
        if self.is_train_dataset:
            # self.optimizer.zero_grad()
            self.engine.backward(loss)
            self.optimizer.step()

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
        if self.dataset_key.startswith("val"):
            self.config["model"]["thresholds"]["cls"] = self._optimal_threshold()
            self.thresholds.append(self.config["model"]["thresholds"]["cls"])
        self.metrics.thresholds = self.config["model"]["thresholds"]  # class Metric
        super().on_dataset_end(exp)

    def on_experiment_end(self, exp: "IExperiment") -> None:
        """Save threshold for best model"""
        self.config["model"]["thresholds"]["cls"] = self.thresholds[self.best_epoch - 1]
        super().on_experiment_end(exp)

    def _optimal_threshold(self) -> float:
        """Choose optimal threshold"""

        output_probs = [
            torch.sigmoid(predictions["cls_logits"])
            for predictions in self.metrics.stored_preds_batches
        ]
        y_true = [
            references["labels"] for references in self.metrics.stored_refs_batches
        ]

        output_probs = torch.cat(output_probs)[:, 0]
        y_true = torch.cat(y_true)[:, 0]

        ext = self.graphs_kwargs["extension"]

        pr_threshold_opt = plot_pr_curve(
            output_arr=output_probs,
            y_true=y_true,
            path_to_save=os.path.join(
                self.save_dir,
                self.experiment_name,
                self.graphs_kwargs["dir_to_save"],
                "pr_curve_" + f"epoch_{self.epoch_step}" + f"{ext}",
            ),
        )

        auroc_threshold_opt = plot_roc_curve(
            output_arr=output_probs,
            y_true=y_true,
            path_to_save=os.path.join(
                self.save_dir,
                self.experiment_name,
                self.graphs_kwargs["dir_to_save"],
                "roc_curve_" + f"epoch_{self.epoch_step}" + f"{ext}",
            ),
        )

        if self.threshold_curve == "pr":
            threshold_opt = pr_threshold_opt
        else:
            threshold_opt = auroc_threshold_opt

        plot_histogram_distribution(
            output_arr=output_probs,
            y_true=y_true,
            threshold_opt=threshold_opt,
            path_to_save=os.path.join(
                self.save_dir,
                self.experiment_name,
                self.graphs_kwargs["dir_to_save"],
                "output_distribution_" + f"epoch_{self.epoch_step}" + f"{ext}",
            ),
        )

        plot_f1_surve(
            output_arr=output_probs,
            y_true=y_true,
            path_to_save=os.path.join(
                self.save_dir,
                self.experiment_name,
                self.graphs_kwargs["dir_to_save"],
                "f1_curve_" + f"epoch_{self.epoch_step}" + f"{ext}",
            ),
        )

        return threshold_opt
