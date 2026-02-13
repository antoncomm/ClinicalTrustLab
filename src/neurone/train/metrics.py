"""
Metrics for evaluating model performance.
"""

from typing import Tuple, Any, Dict

import torch
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    AUROC,
    Specificity,
)

from neurone.utils.general import data_dict_to_cpu


class Metric:
    """
    The general Multi metric class to provide formatting inputs and computing results.

    Examples:
        >>> metrics = [{"AUROC": "kwargs": {}}]
        >>> metric = Metric(*metrics, task="binary", thresholds={"cls": 0.5})
        >>> metric.add_batch(model_output, gt)
        >>> values = metric.compute()

    Supported metrics:
        - Accuracy
        - Precision
        - Recall
        - F1
        - ROC-AUC
        - Specificity
        - Mean IoU
        - mAP for detection

    Supported tasks:
        - Classification
        - Segmentation
        - Detection

    Args:
        metrics (list):
            list of dicts with definitions of evaluate metrics, for example
        thresholds (list):
            List of thresholds
        **shared_kwargs:
            Arguments which will be passed to every metric in initialization step,
            in addition to personal argument of each metric, specified in config
    """

    def __init__(
        self,
        metrics,
        thresholds=None,
        **shared_kwargs,
    ) -> None:
        self.task = shared_kwargs.get("task")
        metrics_list = {}
        for metric_key, metric_values in metrics.items():
            # specific metric kwargs can overwrite shared_kwargs
            metric_kwargs = {**shared_kwargs, **metric_values["kwargs"]}
            new_metric = globals()[metric_key](**metric_kwargs)
            metrics_list[metric_key] = {
                "format_function": f"format_for_{metric_key.lower()}",
                "name": metric_key.lower(),
                "exmplr": new_metric,
            }
        self.metrics = metrics_list
        self.thresholds = thresholds
        self.stored_preds_batches = []
        self.stored_refs_batches = []

    def format_output(self, metric, predictions, references):
        """
        Method to find specific method of formatting for specific metric.
        """
        return getattr(self, metric)(predictions, references)

    def add_batch(
        self, predictions: Any, references: Any, keep_in_ram: bool = False
    ) -> None:
        """
        Processes a batch of predictions and references, with behavior determined
        by the `keep_in_ram` flag.

        If `keep_in_ram` is True, the batch is added to a list for later computation,
        allowing batches to be accumulated for batch-wise metric computation. This is
        useful for scenarios where metric computation needs to be deferred or batch
        results need to be aggregated before final calculation.

        Args:
            predictions (Any): The model's predictions for the current batch. The
                specific format can vary depending on the metric being calculated.
            references (Any): The ground truth references for the current batch,
                matching the format expected by the metric computation.
            keep_in_ram (bool, optional): Flag to control the storage behavior of the
                batches. If True, batches are stored in RAM for later computation.
                If False, metrics are immediately updated with the provided batch.
                Defaults to False.
        """
        predictions = data_dict_to_cpu(predictions)
        references = data_dict_to_cpu(references)
        if keep_in_ram:
            self._write_batch(predictions=predictions, references=references)
        else:
            self._update(predictions=predictions, references=references)

    def _write_batch(self, predictions: Any, references: Any) -> None:
        self.stored_preds_batches.append(predictions)
        self.stored_refs_batches.append(references)

    def _update(self, predictions: Any, references: Any) -> None:
        for metric_object in self.metrics.values():
            formated_predictions, formated_references = self.format_output(
                metric_object["format_function"],
                predictions=predictions,
                references=references,
            )
            metric_object["exmplr"].update(formated_predictions, formated_references)

    def compute(self) -> Dict:
        """
        Method to thrigger compute method of all metrics.
        """
        for predictions, references in zip(
            self.stored_preds_batches, self.stored_refs_batches
        ):
            self._update(predictions=predictions, references=references)
        self.stored_preds_batches = []
        self.stored_refs_batches = []
        return self._compute()

    def _compute(self) -> Dict:
        values = {}
        for metric_class in self.metrics.values():
            # exmplr = self.metrics[metric]["exmplr"]
            value = metric_class["exmplr"].compute()

            formatted_value = {}
            # scalar metric output
            if isinstance(value, torch.Tensor) and value.dim() == 0:
                formatted_value = {metric_class["name"]: value.item()}
            # vector metric output
            elif isinstance(value, torch.Tensor) and value.dim() == 1:
                formatted_value = {
                    f"{metric_class['name']}_{i}": v.item()
                    for i, v in enumerate(value, 0)
                }
            # dict metric output
            elif isinstance(value, dict):
                for subname, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor) and subvalue.dim() == 0:
                        formatted_value[subname] = (
                            subvalue.item()
                            if isinstance(subvalue, torch.Tensor)
                            else subvalue
                        )
                    # vector metric output
                    elif isinstance(subvalue, torch.Tensor) and subvalue.dim() == 1:
                        for i, subsubvalue in enumerate(subvalue, 0):
                            formatted_value[f"{subname}_{i}"] = subsubvalue.item()
                    else:
                        raise ValueError("Unknown format of the metric result.")
            else:
                raise ValueError("Unknown format of the metric result.")
            values.update(formatted_value)
            metric_class["exmplr"].reset()

        return values

    def _prepare_confusion_matrix_based(self, predictions, references) -> None:
        """
        Method to format data for Accuracy, Precision, Recall and F1 metrics.

        Args:
            predictions(Dict): a dictionary with predictions
            references(Dict): a dictionary with references
        Returns:
            Formatted predictions and references.
        """
        if self.task == "binary":

            if (
                predictions["cls_logits"].shape[-1] == 1
                and predictions["cls_logits"].shape != references["labels"].shape
            ):
                predictions["cls_logits"] = predictions["cls_logits"].squeeze(
                    -1
                )  # rm last dim, converting model output into binary cls format

            predictions = self.do_format(
                predictions["cls_logits"],
                activation_flag=True,
                threshold_flag=True,
                threshold_type="cls",
            )
        elif self.task == "multiclass":
            predictions = self.do_format(
                predictions["cls_logits"],
                activation_flag=True,
                threshold_flag=False,
                threshold_type=None,
            )
        else:
            raise ValueError(f"Unsupported task {self.task}")

        references = references["labels"]

        if self.task == "multiclass":
            predictions = predictions.argmax(1)

        return predictions, references

    def do_format(
        self, inputs, activation_flag=False, threshold_flag=False, threshold_type="cls"
    ):
        """
        Method to convert tensors to the required format for further use in metric,
        which can involving applying activation and thresholding.
        """
        if activation_flag:
            if self.task == "multiclass":
                activation = "softmax"
            else:
                activation = "sigmoid"
            inputs = self.do_activation(inputs, activation)
        if threshold_flag:
            inputs = self.do_threshold(inputs, threshold_type)

        return inputs

    def do_threshold(self, inputs, threshold_type="cls", threshold=None):
        """
        Method to threshold data.
        """
        if threshold:
            thr = threshold
        else:
            thr = self.thresholds[threshold_type]
        return (inputs > thr) * 1.0

    def do_activation(self, inputs, activation="sigmoid"):
        """
        Method to apply activation function to data.
        """
        if activation == "sigmoid":
            return torch.sigmoid(inputs)
        if activation == "softmax":
            return torch.softmax(inputs, dim=1)
        raise ValueError(f"Unknown function {activation}.")

    def format_for_accuracy(self, predictions, references) -> Tuple:
        """
        Method to format data for Accuracy metric.
        """
        return self._prepare_confusion_matrix_based(predictions, references)

    def format_for_specificity(self, predictions, references) -> Tuple:
        """
        Method to format data for Specificity metric.
        """
        return self._prepare_confusion_matrix_based(predictions, references)

    def format_for_precision(self, predictions, references) -> Tuple:
        """
        Method to format data for Precision metric.
        """
        return self._prepare_confusion_matrix_based(predictions, references)

    def format_for_recall(self, predictions, references) -> Tuple:
        """
        Method to format data for Recall metric.
        """
        return self._prepare_confusion_matrix_based(predictions, references)

    def format_for_f1score(self, predictions, references) -> Tuple:
        """
        Method to format data for f1 metric.
        """
        return self._prepare_confusion_matrix_based(predictions, references)

    def format_for_auroc(self, predictions, references) -> Tuple:
        """
        Method to format data for ROC AUC metric.

        Note: applying activation is crucial for correct calculation in torchmetrics
        """
        predictions = self.do_format(predictions["cls_logits"], True, False, "cls")
        references = references["labels"]

        return predictions, references
