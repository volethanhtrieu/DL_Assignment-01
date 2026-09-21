"""
Official Training & Evaluation Script on Fashion-MNIST
CO3133: Deep Learning and Its Applications - Assignment 01 (HCMUT)
Milestone 1 (M1) Checkpoint — Group Goat

Usage:
    python train.py --model all --epochs 10
    python train.py --model linear --epochs 10 --lr 0.001
    python train.py --model mlp --epochs 12 --lr 0.001
"""

import os
import argparse
import time
from typing import Dict

import torch
import torch.nn as nn

from data_setup import get_fashion_mnist_dataloaders, CLASS_NAMES
from models import get_model, count_parameters
from engine import fit, evaluate_model, plot_training_history


def train_single_model(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    output_dir: str = "./outputs",
) -> Dict:
    """
    Trains, validates, evaluates on test set, and exports all deliverables for a single model.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "#" * 78)
    print(f"EXPERIMENT: {model_name.upper()} CLASSIFIER ON FASHION-MNIST")
    print("#" * 78)

    # 1. Instantiate Model
    model = get_model(model_name).to(device)
    param_count = count_parameters(model)
    print(f"Model Architecture: {model_name.upper()} | Trainable Parameters: {param_count:,}")

    # 2. Criterion & Optimizer (Handbook mandate: CrossEntropyLoss directly with logits)
    criterion = nn.CrossEntropyLoss()
    if model_name == "linear":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Full Training Lifecycle with Early Stopping and Best Checkpoint Saving
    t0 = time.time()
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        early_stopping_patience=4,
        early_stopping_min_delta=0.05,
        restore_best_weights=True,
    )
    total_train_time = time.time() - t0

    # 4. Export Learning Curves (Loss, Accuracy, Macro-F1)
    curves_path = os.path.join(figures_dir, f"learning_curves_{model_name}.png")
    plot_training_history(
        history=history,
        save_path=curves_path,
        model_name=f"{model_name.upper()} Classifier",
        show=False,
    )

    # 5. Final Evaluation on Strictly Unseen Benchmark Test Set (10,000 images)
    print("\n" + "=" * 78)
    print(f"FINAL EVALUATION ON UNSEEN TEST SET (10,000 IMAGES) — {model_name.upper()}")
    print("=" * 78)
    cm_path = os.path.join(figures_dir, f"confusion_matrix_{model_name}.png")
    pred_examples_path = os.path.join(figures_dir, f"prediction_examples_{model_name}.png")
    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        model_name=f"{model_name.upper()} Classifier",
        confusion_matrix_path=cm_path,
        prediction_examples_path=pred_examples_path,
    )

    return {
        "model_name": model_name,
        "parameters": param_count,
        "train_time_sec": total_train_time,
        "infer_time_sec": test_metrics.get("inference_time_total_sec", 0.0),
        "infer_latency_ms": test_metrics.get("inference_time_ms_per_sample", 0.0),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train M1 Models on Fashion-MNIST")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["linear", "mlp", "cnn", "all"],
        help="Model to train: 'linear', 'mlp', 'cnn', or 'all'",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per model")
    parser.add_argument("--batch_size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    args = parser.parse_args()

    # Hardware Device Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} " + (f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # Load Official Stratified DataLoaders (50k / 10k / 10k)
    print("\nLoading Fashion-MNIST DataLoaders (Stratified Split, Seed 42)...")
    train_loader, val_loader, test_loader, classes = get_fashion_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        seed=42,
        augment_train=False,
    )

    models_to_train = ["linear", "mlp", "cnn"] if args.model == "all" else [args.model]
    summary_results = []

    for m_name in models_to_train:
        res = train_single_model(
            model_name=m_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            output_dir=args.output_dir,
        )
        summary_results.append(res)

    # Print Comparative Summary Table for Report
    print("\n" + "=" * 98)
    print(f"{'M1 CHECKPOINT COMPARATIVE BENCHMARK SUMMARY TABLE':^98}")
    print("=" * 98)
    print(f"{'Model Architecture':<20} | {'Params':^10} | {'Test Loss':^10} | {'Test Acc':^10} | {'Test F1':^10} | {'Train (s)':^9} | {'Infer (s)':^9} | {'Latency':^11}")
    print("-" * 98)
    for r in summary_results:
        print(
            f"{r['model_name'].upper():<20} | "
            f"{r['parameters']:^10,d} | "
            f"{r['test_loss']:^10.4f} | "
            f"{r['test_accuracy']:^9.2f}% | "
            f"{r['test_macro_f1']:^9.2f}% | "
            f"{r['train_time_sec']:^9.1f} | "
            f"{r['infer_time_sec']:^9.2f} | "
            f"{r['infer_latency_ms']:^8.3f} ms"
        )
    print("=" * 98)
    print(f"All outputs, checkpoints, and figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
