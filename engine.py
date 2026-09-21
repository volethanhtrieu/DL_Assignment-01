"""
Training & Evaluation Pipeline (Train/Val Loops, Metrics & History Tracking)
CO3133: Deep Learning & Its Applications - Assignment 01 (HCMUT)
Milestone 1 (M1) Deliverable

Features:
- train_epoch: Forward, backward, optimizer step, running metrics (Loss, Accuracy, Macro-F1)
- validate_epoch: Evaluation loop under torch.no_grad(), computing Loss, Accuracy, Macro-F1
- fit: Full training lifecycle manager with history logging and best-checkpoint saving
- plot_training_history: Publication-ready loss & metric curve visualizer
- evaluate_model: Inference execution timing (s, ms/sample, throughput) and metric computation
- plot_prediction_examples: Visualizes sample correct vs. incorrect predictions (Handbook Sec 12.1)
"""

import os
import time
import json
import csv
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.metrics._classification import f1_score
except ImportError:
    from sklearn.metrics import f1_score

from data_setup import CLASS_NAMES
from metrics import compute_metrics, plot_confusion_matrix, print_metrics_summary


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Executes one complete training epoch over the dataloader.

    Args:
        model: PyTorch model (nn.Module).
        dataloader: DataLoader for training set.
        criterion: Loss function (e.g. nn.CrossEntropyLoss).
        optimizer: Optimization algorithm (e.g. Adam, SGD).
        device: Target compute device (CUDA / CPU).

    Returns:
        Dictionary containing 'loss', 'accuracy', and 'macro_f1'.
    """
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    total_samples = 0
    correct_samples = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = images.size(0)
        total_samples += batch_size

        # Zero gradients from previous step
        optimizer.zero_grad()

        # Forward pass (outputs are unnormalized logits)
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass & parameter update
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item() * batch_size

        # Track accuracy and collect predictions for Macro-F1
        _, preds = torch.max(outputs, dim=1)
        correct_samples += (preds == targets).sum().item()

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_samples / total_samples) * 100.0
    epoch_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "macro_f1": epoch_macro_f1,
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates model performance over the validation or test dataloader.

    Args:
        model: PyTorch model (nn.Module).
        dataloader: DataLoader for validation or test set.
        criterion: Loss function.
        device: Target compute device.

    Returns:
        Dictionary containing 'loss', 'accuracy', and 'macro_f1'.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    total_samples = 0
    correct_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = images.size(0)
            total_samples += batch_size

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, dim=1)
            correct_samples += (preds == targets).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_samples / total_samples) * 100.0
    epoch_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "macro_f1": epoch_macro_f1,
    }


class EarlyStopping:
    """
    Early stopping monitor to stop training when a monitored metric stops improving.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        mode (str): One of {'max', 'min'}. In 'max' mode (e.g. macro-F1, accuracy),
                    training stops when quantity stops increasing. In 'min' mode (e.g. loss),
                    training stops when quantity stops decreasing.
        verbose (bool): If True, prints messages when waiting for improvement.
    """
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_val: float) -> bool:
        if self.best_score is None:
            self.best_score = current_val
            return False

        if self.mode == "max":
            improved = current_val > (self.best_score + self.min_delta)
        else:
            improved = current_val < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_val
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"       -> [EarlyStopping] No improvement for {self.counter}/{self.patience} epoch(s). (Best: {self.best_score:.2f}%)")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_dir: str = "./checkpoints",
    model_name: str = "model",
    early_stopping_patience: Optional[int] = 3,
    early_stopping_min_delta: float = 0.01,
    restore_best_weights: bool = True,
) -> Dict[str, List[float]]:
    """
    Executes complete training and validation cycle across all epochs.
    Tracks training history and automatically saves the best checkpoint
    based on validation macro-F1 score.
    Supports early stopping if no improvement is observed for N consecutive epochs.

    Returns:
        History dictionary with lists of metrics per epoch.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_time": [],
    }

    best_val_f1 = -1.0

    # Initialize early stopping monitor
    early_stopper = (
        EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="max",
        )
        if early_stopping_patience is not None and early_stopping_patience > 0
        else None
    )

    print("=" * 78)
    patience_str = f" | EarlyStopping: {early_stopping_patience} epochs" if early_stopper else ""
    print(f"STARTING TRAINING: {model_name.upper()} (Max: {epochs} Epochs{patience_str}) on Device: {device}")
    print("=" * 78)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^10} | {'Train F1':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Val F1':^10} | {'Time (s)':^8}")
    print("-" * 78)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Training & Validation passes
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["epoch_time"].append(elapsed)

        # Checkpoint saving on best Val Macro-F1
        is_best = val_metrics["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                },
                best_checkpoint_path,
            )

        best_marker = " *" if is_best else ""
        print(
            f"{epoch:^7d} | "
            f"{train_metrics['loss']:^10.4f} | "
            f"{train_metrics['accuracy']:^9.2f}% | "
            f"{train_metrics['macro_f1']:^9.2f}% | "
            f"{val_metrics['loss']:^10.4f} | "
            f"{val_metrics['accuracy']:^9.2f}% | "
            f"{val_metrics['macro_f1']:^9.2f}%{best_marker} | "
            f"{elapsed:^8.2f}"
        )

        # Check early stopping condition
        if early_stopper is not None and early_stopper(val_metrics["macro_f1"]):
            print("-" * 78)
            print(f"[EARLY STOPPING TRIGGERED] Stopped early at epoch {epoch} due to no improvement for {early_stopping_patience} consecutive epochs.")
            break

    # Restore best checkpoint weights to ensure the returned model is optimal
    if restore_best_weights and os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("-" * 78)
        print(f"Restored best model weights from Epoch {checkpoint['epoch']} (Val Macro-F1: {checkpoint['val_macro_f1']:.2f}%)")

    # Automatically export training history to JSON and CSV for reporting
    json_history_path = os.path.join(checkpoint_dir, f"{model_name}_history.json")
    csv_history_path = os.path.join(checkpoint_dir, f"{model_name}_history.csv")
    save_history(history, json_path=json_history_path, csv_path=csv_history_path)

    print("-" * 78)
    print(f"Training Complete! Best Validation Macro-F1: {best_val_f1:.2f}%")
    print(f"Best Model Checkpoint saved at: {best_checkpoint_path}")
    print("=" * 78)

    return history


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    val_macro_f1: float,
    checkpoint_path: str,
):
    """
    Saves a model checkpoint with complete performance metadata and optimizer state.
    """
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "val_macro_f1": float(val_macro_f1),
        },
        checkpoint_path,
    )
    print(f" Checkpoint saved to: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[int, float]]:
    """
    Loads model and optional optimizer state from a checkpoint.
    Returns metadata dict: {'epoch', 'val_loss', 'val_accuracy', 'val_macro_f1'}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f" Loaded checkpoint from {checkpoint_path} (Epoch {checkpoint.get('epoch', 0)}, Val Macro-F1: {checkpoint.get('val_macro_f1', 0.0):.2f}%)")
    return {
        "epoch": checkpoint.get("epoch", 0),
        "val_loss": checkpoint.get("val_loss", 0.0),
        "val_accuracy": checkpoint.get("val_accuracy", 0.0),
        "val_macro_f1": checkpoint.get("val_macro_f1", 0.0),
    }


def save_history(
    history: Dict[str, List[float]],
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
):
    """
    Exports training history to JSON and CSV formats for easy report drafting and analysis.
    """
    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f" History exported to JSON: {json_path}")

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        keys = list(history.keys())
        num_rows = len(history[keys[0]])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + keys)
            for i in range(num_rows):
                writer.writerow([i + 1] + [history[k][i] for k in keys])
        print(f" History exported to CSV: {csv_path}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str = "Model",
    confusion_matrix_path: Optional[str] = "confusion_matrix.png",
    prediction_examples_path: Optional[str] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Comprehensive evaluation function:
    - Evaluates model on test/val set.
    - Measures inference execution time (total seconds, ms/sample latency, throughput).
    - Computes CrossEntropyLoss, Overall Accuracy, and Macro-F1 (Handbook mandated).
    - Prints full academic metrics breakdown (Precision, Recall, F1 per class).
    - Exports publication-ready confusion matrix heatmap.
    - Optionally exports correct & incorrect prediction examples visualization.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_infer_time = time.perf_counter()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)
            total_samples += batch_size

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_infer_time = time.perf_counter() - start_infer_time
    latency_ms = (total_infer_time / total_samples) * 1000.0 if total_samples > 0 else 0.0
    throughput = total_samples / total_infer_time if total_infer_time > 0 else 0.0

    total_loss = running_loss / total_samples
    metrics = compute_metrics(all_targets, all_preds, loss=total_loss)
    metrics["inference_time_total_sec"] = float(total_infer_time)
    metrics["inference_time_ms_per_sample"] = float(latency_ms)
    metrics["inference_throughput_fps"] = float(throughput)

    print_metrics_summary(metrics)

    if confusion_matrix_path:
        plot_confusion_matrix(
            all_targets,
            all_preds,
            save_path=confusion_matrix_path,
            model_name=model_name,
        )

    if prediction_examples_path:
        plot_prediction_examples(
            model=model,
            dataloader=dataloader,
            device=device,
            save_path=prediction_examples_path,
            model_name=model_name,
        )

    return metrics


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = "learning_curves.png",
    model_name: str = "Model",
    show: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Renders and returns publication-ready training & validation curves.
    Panel 1: Cross-Entropy Loss
    Panel 2: Overall Accuracy (%)
    Panel 3: Macro-F1 Score (%)

    Args:
        history: History dict containing train/val metrics per epoch.
        save_path: Optional path to save image file (e.g. 'learning_curves.png').
        model_name: Name of model for title display.
        show: If True, calls plt.show() interactively.

    Returns:
        Tuple of (fig, axes) for interactive display or downstream editing.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1769aa", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ef4444", linewidth=2, linestyle="--")
    axes[0].set_title(f"{model_name} — Loss Curves", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Panel 2: Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#1769aa", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", color="#10b981", linewidth=2, linestyle="--")
    axes[1].set_title(f"{model_name} — Accuracy Curves", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # Panel 3: Macro-F1
    axes[2].plot(epochs, history["train_macro_f1"], label="Train Macro-F1", color="#1769aa", linewidth=2)
    axes[2].plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="#8b5cf6", linewidth=2, linestyle="--")
    axes[2].set_title(f"{model_name} — Macro-F1 Curves", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1 (%)")
    axes[2].legend(frameon=True)
    axes[2].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(f"{model_name} Training & Validation Performance Curves", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" Learning curves saved to: {save_path}")

    if show:
        plt.show()

    return fig, axes


def plot_prediction_examples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str] = CLASS_NAMES,
    num_samples: int = 6,
    save_path: Optional[str] = "prediction_examples.png",
    model_name: str = "Model",
    mean: float = 0.2860,
    std: float = 0.3530,
    show: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualizes sample images the model predicted correctly vs. incorrectly.
    Mandatory deliverable per Course Handbook Section 12.1.

    Args:
        model: PyTorch model.
        dataloader: DataLoader containing evaluation images.
        device: Device to run evaluation on.
        class_names: Class labels mapping (default: Fashion-MNIST).
        num_samples: Number of correct and incorrect samples to display (columns).
        save_path: Destination path for saving PNG.
        model_name: Model title.
        mean: Normalization mean (default Fashion-MNIST: 0.2860).
        std: Normalization standard deviation (default Fashion-MNIST: 0.3530).
        show: If True, invokes plt.show().

    Returns:
        Tuple of (fig, axes).
    """
    model.eval()
    all_correct = []
    all_incorrect = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probabilities, dim=1)

            for i in range(images.size(0)):
                t_cls = int(targets[i].item())
                p_cls = int(preds[i].item())
                conf = float(confidences[i].item() * 100.0)

                # Squeeze to 2D (28, 28) and denormalize
                img_raw = images[i].detach().cpu().squeeze().numpy()
                img_denorm = np.clip(img_raw * std + mean, 0.0, 1.0)
                item = (img_denorm, t_cls, p_cls, conf)

                if t_cls == p_cls:
                    if len(all_correct) < 200:
                        all_correct.append(item)
                else:
                    if len(all_incorrect) < 200:
                        all_incorrect.append(item)

            if len(all_correct) >= 50 and len(all_incorrect) >= 50:
                break

    # Prioritize unique true classes for visual diversity across the columns
    def _select_diverse(items, target_k):
        selected_indices = []
        seen = set()
        for idx, it in enumerate(items):
            if it[1] not in seen:
                selected_indices.append(idx)
                seen.add(it[1])
                if len(selected_indices) == target_k:
                    return [items[i] for i in selected_indices]
        for idx in range(len(items)):
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) == target_k:
                    return [items[i] for i in selected_indices]
        return [items[i] for i in selected_indices]

    correct_samples = _select_diverse(all_correct, num_samples)
    incorrect_samples = _select_diverse(all_incorrect, num_samples)

    n_cols = max(len(correct_samples), len(incorrect_samples), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.8 * n_cols, 6.8))
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    def _style_subplot(ax, img, title_text, title_color, border_color):
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title_text, fontsize=8.5, fontweight="bold", color=title_color, pad=5)
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(2.5)

    # Row 0: Correct Predictions
    for col in range(n_cols):
        ax = axes[0, col]
        if col < len(correct_samples):
            img, true_c, pred_c, conf = correct_samples[col]
            c_name = class_names[true_c] if true_c < len(class_names) else str(true_c)
            title = f"[CORRECT]\nTrue: {c_name}\nPred: {c_name} ({conf:.1f}%)"
            _style_subplot(ax, img, title, "#15803d", "#22c55e")
        else:
            ax.axis("off")
        if col == 0:
            ax.set_ylabel("CORRECT\nPREDICTIONS", fontsize=10.5, fontweight="bold", color="#15803d", labelpad=8)

    # Row 1: Incorrect Predictions
    for col in range(n_cols):
        ax = axes[1, col]
        if col < len(incorrect_samples):
            img, true_c, pred_c, conf = incorrect_samples[col]
            t_name = class_names[true_c] if true_c < len(class_names) else str(true_c)
            p_name = class_names[pred_c] if pred_c < len(class_names) else str(pred_c)
            title = f"[MISTAKE]\nTrue: {t_name}\nPred: {p_name} ({conf:.1f}%)"
            _style_subplot(ax, img, title, "#b91c1c", "#ef4444")
        else:
            if len(incorrect_samples) == 0 and col == n_cols // 2:
                ax.text(0.5, 0.5, "100% Accuracy\n(No errors)", ha="center", va="center", fontsize=10, color="#15803d")
            ax.axis("off")
        if col == 0:
            ax.set_ylabel("INCORRECT\nPREDICTIONS", fontsize=10.5, fontweight="bold", color="#b91c1c", labelpad=8)

    plt.suptitle(
        f"{model_name} — Correct & Incorrect Prediction Examples (Handbook Sec 12.1)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.subplots_adjust(hspace=0.55)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" Prediction examples saved to: {save_path}")

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    print("=" * 78)
    print("RUNNING ENGINE.PY SELF-CHECK: PIPELINE & VISUALS GENERATION")
    print("=" * 78)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Create dummy dataset for pipeline verification
    torch.manual_seed(42)
    dummy_x = torch.randn(200, 1, 28, 28)
    dummy_y = torch.randint(0, 10, (200,))
    
    train_dataset = torch.utils.data.TensorDataset(dummy_x[:160], dummy_y[:160])
    val_dataset = torch.utils.data.TensorDataset(dummy_x[160:], dummy_y[160:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Softmax Regression model (Linear classifier)
    test_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.01)

    # Run 3 training epochs to test full history tracking and checkpointing
    history = fit(
        model=test_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=12,
        device=device,
        model_name="demo_linear",
    )

    # Generate and save visual learning curves (Bước 4)
    fig, axes = plot_training_history(
        history=history,
        save_path="demo_learning_curves.png",
        model_name="Demo Linear Classifier",
        show=False,
    )
    plt.close(fig)

    assert os.path.exists("demo_learning_curves.png"), "Error: learning curve visual was not saved!"
    print("\nVisual verification successful: demo_learning_curves.png successfully generated and returned!")

    # Full Evaluation & Confusion Matrix Check & Prediction Examples & Inference Timing
    print("\nEvaluating best restored model on validation set (Computing Loss, Accuracy, Macro-F1, Inference Time)...")
    eval_results = evaluate_model(
        model=test_model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        model_name="Demo Linear Classifier",
        confusion_matrix_path="demo_confusion_matrix.png",
        prediction_examples_path="demo_prediction_examples.png",
    )
    assert os.path.exists("demo_confusion_matrix.png"), "Error: confusion matrix visual was not saved!"
    assert os.path.exists("demo_prediction_examples.png"), "Error: prediction examples visual was not saved!"
    assert "inference_time_total_sec" in eval_results, "Error: inference_time_total_sec missing from eval results!"
    assert "inference_time_ms_per_sample" in eval_results, "Error: inference_time_ms_per_sample missing from eval results!"
    assert "inference_throughput_fps" in eval_results, "Error: inference_throughput_fps missing from eval results!"
    print(f"\nVerified inference timing: {eval_results['inference_time_total_sec']:.4f}s total | Latency: {eval_results['inference_time_ms_per_sample']:.4f} ms/sample | Throughput: {eval_results['inference_throughput_fps']:.1f} samples/s")
    print("\nAll Step 3, Step 4, and Handbook Gap verifications passed successfully!")
    print("=" * 78)
