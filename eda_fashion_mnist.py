"""
Exploratory Data Analysis (EDA) Script for Fashion-MNIST
CO3133 Assignment 01 - Group Goat
Generates class sample grids and distribution statistics.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from data_setup import get_fashion_mnist_dataloaders, CLASS_NAMES


def plot_sample_grid(data_loader, save_path="eda_samples.png"):
    """
    Visualizes a 2x5 grid of sample images with one example per class.
    """
    images, labels = next(iter(data_loader))
    
    # Denormalize image for correct visual display
    from data_setup import MEAN, STD
    images_denorm = images * STD + MEAN
    images_denorm = torch.clamp(images_denorm, 0.0, 1.0)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    found_classes = set()
    sample_idx = 0
    
    for i in range(len(labels)):
        label = labels[i].item()
        if label not in found_classes and len(found_classes) < 10:
            found_classes.add(label)
            ax = axes[label]
            img = images_denorm[i].squeeze().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{label}: {CLASS_NAMES[label]}", fontsize=10)
            ax.axis("off")
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Sample visualization saved to: {save_path}")


if __name__ == "__main__":
    import torch
    train_dl, val_dl, test_dl, _ = get_fashion_mnist_dataloaders(batch_size=128, num_workers=0)
    plot_sample_grid(train_dl, save_path="fashion_mnist_samples.png")
