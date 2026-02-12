"""Graphs functions"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve


def plot_pr_curve(
    output_arr: np.ndarray, y_true: np.ndarray, path_to_save: str
) -> float:
    """
    Plots the Precision-Recall (PR) curve with interpolated,
    non-interpolated precision-recall curves, and f1 score at optimal score

    Return optimal threshold by pr_curve
    """
    p, r, thresholds = precision_recall_curve(
        y_true,
        output_arr,
    )
    f1 = 2 * p * r / (p + r)
    index = np.argmax(f1)
    threshold_opt = float(thresholds[index])

    plt.figure(figsize=(8, 6))

    plt.plot(r, p, label="w/o interpolation", color="#aab6d4")
    plt.scatter(
        x=r[index],
        y=p[index],
        s=80,
        facecolors="none",
        edgecolors="r",
    )
    f = f1[index]
    p_max = 1.05
    r_max = 1.05
    r0 = p_max * f / (2 * p_max - f)
    r = np.linspace(r0, r_max, 100)
    p = -f * r / (f - 2 * r)
    plt.plot(
        r,
        p,
        label=f"f1 = {f:.2f} (at thr = {threshold_opt:.2f})",
        linestyle="--",
        color="gray",
    )

    plt.xlim([0.0 - 0.05, r_max])
    plt.ylim([0.0 - 0.05, p_max])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower left")

    plt.savefig(path_to_save, bbox_inches="tight", transparent=True)
    plt.close()

    return threshold_opt


def plot_f1_surve(
    output_arr: np.ndarray, y_true: np.ndarray, path_to_save: str
) -> None:
    """Plot f1 based on given thresholds"""

    p, r, thresholds = precision_recall_curve(
        y_true,
        output_arr,
    )
    p = p[1:]
    r = r[1:]
    f1 = 2 * p * r / (p + r)

    s_max = 1.05
    f_max = 1.05

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1, color="black")
    plt.xlabel("threshold")
    plt.ylabel("f1")
    plt.title("f1 score curve")
    plt.xlim([0.0 - 0.05, s_max])
    plt.ylim([0.0 - 0.05, f_max])

    plt.savefig(path_to_save, bbox_inches="tight", transparent=True)
    plt.close()


def plot_roc_curve(
    output_arr: np.ndarray, y_true: np.ndarray, path_to_save: str
) -> float:
    """
    Compute the optimal threshold for binary classification based on the
    Receiver Operating Characteristic (ROC) curve.

    This function calculates the optimal threshold for binary
    classification by maximizing the geometric mean (G-Mean)
    of the True Positive Rate (Sensitivity/Recall) and True Negative Rate
    (Specificity) obtained from the ROC curve. The threshold
    that yields the highest G-Mean is selected as the optimal threshold.

    The function also generates and saves the ROC curve plot and a
    histogram plot of output values categorized by labels.

    Reference
    ---------
    https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/?__cf_chl_rt_tk=zhB04KY9PDvuojB5VIQHLqygkhxqUKGelmcUO2CxA10-1694174565-0-gaNycGzNFrs
    """

    auroc = roc_auc_score(y_true=y_true, y_score=output_arr)
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=output_arr)

    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    threshold_opt = thresholds[index]

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUROC = {auroc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    # Save the ROC curve plot to the specified path
    plt.savefig(path_to_save, bbox_inches="tight", transparent=True)

    # Close the plot to free up memory (optional)
    plt.close()

    return float(threshold_opt)


def plot_histogram_distribution(
    output_arr: np.ndarray, y_true: np.ndarray, threshold_opt: float, path_to_save: str
) -> None:
    """
    Plot histogram distribution of the output array depending on the true labels.

    Parameters
    ----------
        output_arr (np.ndarray): Array containing output values.
        y_true (np.ndarray): Array containing true labels (0 for negative, 1 for positive).
        threshold_opt (float): Threshold value used for plotting.
        path_to_save (str): Path to save the generated plot.

    Raises
    ------
        ValueError: If the length of `output_arr` does not match the length of `y_true`.
    """
    # Plot histogram (histplot) of output_arr depending on y labels using Seaborn
    plt.figure(figsize=(8, 6))
    sns.histplot(data=output_arr[y_true == 0], color="blue", label="Negative", kde=True)
    sns.histplot(data=output_arr[y_true == 1], color="red", label="Positive", kde=True)
    # Add a vertical line at x=threshold_opt
    plt.axvline(
        x=threshold_opt,
        color="green",
        linestyle="--",
        label=f"Threshold = {threshold_opt:.2f}",
    )

    plt.xlabel("Output Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of Output Values by Label")
    plt.legend()

    plt.savefig(path_to_save, bbox_inches="tight", transparent=True)
    plt.close()
