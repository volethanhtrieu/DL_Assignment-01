"""
Evaluation Metrics & Confusion Matrix Suite for Fashion-MNIST
CO3133: Deep Learning and Its Applications - Assignment 01 (HCMUT)
Milestone 1 (M1) Deliverable

Features:
- compute_metrics: Loss, Overall Accuracy, Macro-F1 Score, Per-class Precision/Recall/F1
- plot_confusion_matrix: Publication-quality confusion matrix heatmap across all 10 Fashion-MNIST classes
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.metrics._classification import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )
except ImportError:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )
import torch

from data_setup import CLASS_NAMES


def compute_metrics(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    loss: Optional[float] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Computes all mandatory evaluation metrics required by the Course Handbook rubric:
    1. Loss (CrossEntropyLoss)
    2. Overall Accuracy (%)
    3. Macro-F1 Score (%) — Unweighted mean across all 10 classes

    Also computes per-class metrics for detailed error analysis.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class indices.
        loss: Optional scalar cross-entropy loss.

    Returns:
        Dictionary containing overall metrics and per-class breakdowns.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0

    # Per-class precision, recall, and F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class_metrics[name] = {
            "precision": float(precision[i] * 100.0),
            "recall": float(recall[i] * 100.0),
            "f1": float(f1[i] * 100.0),
            "support": int(support[i]),
        }

    results = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_metrics,
    }

    if loss is not None:
        results["loss"] = float(loss)

    return results


def print_metrics_summary(metrics: Dict[str, Union[float, Dict]]):
    """Prints a clean, formatted academic summary of model performance."""
    print("=" * 68)
    print(f"{'MANDATORY EVALUATION METRICS SUMMARY':^68}")
    print("=" * 68)
    if "loss" in metrics:
        print(f"  Cross-Entropy Loss:   {metrics['loss']:.4f}")
    print(f"  Overall Accuracy:     {metrics['accuracy']:.2f}%")
    print(f"  Macro-F1 Score:       {metrics['macro_f1']:.2f}% (Handbook Mandated)")
    print(f"  Weighted-F1 Score:    {metrics['weighted_f1']:.2f}%")
    if "inference_time_total_sec" in metrics:
        fps_str = f" ({metrics['inference_throughput_fps']:.1f} samples/s)" if "inference_throughput_fps" in metrics else ""
        print(f"  Inference Time Total: {metrics['inference_time_total_sec']:.4f}s{fps_str}")
    if "inference_time_ms_per_sample" in metrics:
        print(f"  Inference Latency:    {metrics['inference_time_ms_per_sample']:.4f} ms/sample")
    print("-" * 68)
    print(f"{'Class ID & Name':<22} | {'Precision':^11} | {'Recall':^11} | {'F1-Score':^11}")
    print("-" * 68)
    for i, name in enumerate(CLASS_NAMES):
        c_stats = metrics["per_class"][name]
        print(
            f"{i}: {name:<19} | {c_stats['precision']:^10.2f}% | "
            f"{c_stats['recall']:^10.2f}% | {c_stats['f1']:^10.2f}%"
        )
    print("=" * 68)


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    save_path: str = "confusion_matrix.png",
    model_name: str = "Model",
    normalize: bool = True,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders and exports a publication-grade confusion matrix heatmap.

    Args:
        y_true: Ground truth targets.
        y_pred: Model predictions.
        save_path: Destination path for PNG file.
        model_name: Name of model for title.
        normalize: If True, values are percentages per true class.
        show: If True, invokes plt.show().

    Returns:
        Tuple of (fig, ax).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100.0
        fmt = ".1f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion (%)" if normalize else "Count", rotation=270, labelpad=15)

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=10)
    title_norm = " (Normalized %)" if normalize else ""
    ax.set_title(f"{model_name} — Confusion Matrix{title_norm}", pad=14, fontweight="bold")

    # Annotate values inside each cell
    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val_str = f"{cm_display[i, j]:{fmt}}"
            if normalize:
                val_str += "%"
            ax.text(
                j,
                i,
                val_str,
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=7.5,
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" Confusion matrix saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    # Self-contained test
    print("Testing compute_metrics and plot_confusion_matrix...")
    np.random.seed(42)
    dummy_true = np.random.randint(0, 10, 200)
    # Simulate predictions with ~80% accuracy
    dummy_pred = dummy_true.copy()
    corrupt_idx = np.random.choice(200, 40, replace=False)
    dummy_pred[corrupt_idx] = np.random.randint(0, 10, 40)

    res = compute_metrics(dummy_true, dummy_pred, loss=0.4521)
    print_metrics_summary(res)
    fig, ax = plot_confusion_matrix(dummy_true, dummy_pred, save_path="demo_confusion_matrix.png")
    plt.close(fig)
    print("Metrics module verified successfully!")
