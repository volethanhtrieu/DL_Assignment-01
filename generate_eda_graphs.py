"""
Comprehensive Exploratory Data Analysis (EDA) Visualization Suite
CO3133: Deep Learning and Its Applications - Assignment 01 (HCMUT)
Milestone 1 (M1) Deliverable

Generates 5 publication-ready figures satisfying all M1 EDA requirements:
1. Representative Class Samples Grid (10 classes x 5 instances)
2. Class Distribution & Split Partition Analysis (Train/Val/Test balance check)
3. Pixel Intensity Distribution & Background Sparsity Analysis
4. Class Prototypes: Mean Silhouette & Intra-Class Variance Heatmaps
5. 2D Dimensionality Reduction & Cluster Separation (PCA)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
import torch
from torchvision import datasets, transforms
from data_setup import CLASS_NAMES, MEAN, STD, get_stratified_indices

# Configure clean, academic plot style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
})

OUTPUT_DIR = "./eda_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_raw_data(data_dir="./data"):
    """Loads raw Fashion-MNIST datasets without normalization for EDA."""
    train_full = datasets.FashionMNIST(root=data_dir, train=True, download=True)
    test_full = datasets.FashionMNIST(root=data_dir, train=False, download=True)
    
    train_idx, val_idx = get_stratified_indices(train_full.targets, val_samples_per_class=1000, seed=42)
    
    train_images = train_full.data[train_idx].numpy()
    train_labels = train_full.targets[train_idx].numpy()
    
    val_images = train_full.data[val_idx].numpy()
    val_labels = train_full.targets[val_idx].numpy()
    
    test_images = test_full.data.numpy()
    test_labels = test_full.targets.numpy()
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


# ==============================================================================
# Figure 1: Representative Samples Grid (10 classes x 5 instances)
# ==============================================================================
def plot_representative_samples(train_images, train_labels, num_cols=5):
    fig, axes = plt.subplots(10, num_cols, figsize=(num_cols * 1.8, 10 * 1.5))
    
    for c in range(10):
        c_indices = np.where(train_labels == c)[0][:num_cols]
        for col, idx in enumerate(c_indices):
            ax = axes[c, col]
            ax.imshow(train_images[idx], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{c}: {CLASS_NAMES[c]}", fontsize=9, fontweight="bold", rotation=0, labelpad=55, va="center")
            if c == 0:
                ax.set_title(f"Sample #{col+1}", fontsize=10)
                
    plt.suptitle("Fashion-MNIST Representative Samples Across 10 Classes\n(Input Resolution: 28 x 28 Grayscale)", y=0.99)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_representative_samples.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")


# ==============================================================================
# Figure 2: Class Distribution & Partition Split Balance
# ==============================================================================
def plot_class_distribution(train_labels, val_labels, test_labels):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    train_counts = [np.sum(train_labels == c) for c in range(10)]
    val_counts = [np.sum(val_labels == c) for c in range(10)]
    test_counts = [np.sum(test_labels == c) for c in range(10)]
    
    x = np.arange(10)
    width = 0.26
    
    bars1 = ax.bar(x - width, train_counts, width, label="Train (50,000 total)", color="#1769aa", alpha=0.9)
    bars2 = ax.bar(x, val_counts, width, label="Validation (10,000 total)", color="#16a6b6", alpha=0.9)
    bars3 = ax.bar(x + width, test_counts, width, label="Test (10,000 total)", color="#f59e0b", alpha=0.9)
    
    ax.set_xlabel("Fashion-MNIST Classes", fontweight="bold", labelpad=10)
    ax.set_ylabel("Number of Samples", fontweight="bold")
    ax.set_title("Dataset Partition & Class Distribution Balance Check\n(Stratified Split, Seed 42 — Zero Imbalance)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{name}" for c, name in enumerate(CLASS_NAMES)], rotation=0, ha="center", fontsize=9)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 6000)
    
    # Add value labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{yval:,}", ha="center", va="bottom", fontsize=7.5, color="#1769aa")
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{yval:,}", ha="center", va="bottom", fontsize=7.5, color="#16a6b6")
    for bar in bars3:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{yval:,}", ha="center", va="bottom", fontsize=7.5, color="#d97706")
        
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_class_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")


# ==============================================================================
# Figure 3: Pixel Intensity Distribution & Background Sparsity
# ==============================================================================
def plot_pixel_distribution(train_images):
    # Flatten sample for histogram
    sample_flat = train_images[:5000].flatten()
    normalized_flat = ((sample_flat / 255.0) - MEAN) / STD
    
    zero_ratio = np.mean(sample_flat == 0) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Raw pixels
    ax1.hist(sample_flat, bins=50, color="#1769aa", alpha=0.8, edgecolor="none", density=True)
    ax1.set_title("Raw Pixel Value Distribution [0, 255]", fontweight="bold")
    ax1.set_xlabel("Pixel Intensity (uint8)")
    ax1.set_ylabel("Density")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.axvline(0, color="red", linestyle="--", alpha=0.8, label=f"Background zeros (~{zero_ratio:.1f}%)")
    ax1.legend(loc="upper right")
    
    # Normalized pixels
    ax2.hist(normalized_flat, bins=50, color="#16a6b6", alpha=0.8, edgecolor="none", density=True)
    ax2.set_title(f"Normalized Pixel Distribution\n($\\mu={MEAN}, \\sigma={STD}$)", fontweight="bold")
    ax2.set_xlabel("Standardized Pixel Value")
    ax2.set_ylabel("Density")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.axvline(0, color="green", linestyle="--", alpha=0.8, label="Zero-mean center")
    ax2.legend(loc="upper right")
    
    plt.suptitle(f"Pixel Intensity Analysis & Background Sparsity (Background Sparsity: {zero_ratio:.1f}%)", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_pixel_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")


# ==============================================================================
# Figure 4: Class Prototypes (Mean & Variance Heatmaps)
# ==============================================================================
def plot_class_prototypes(train_images, train_labels):
    fig, axes = plt.subplots(2, 10, figsize=(18, 4.5))
    
    for c in range(10):
        c_imgs = train_images[train_labels == c] / 255.0
        mean_img = np.mean(c_imgs, axis=0)
        std_img = np.std(c_imgs, axis=0)
        
        # Mean prototype
        ax_mean = axes[0, c]
        im_m = ax_mean.imshow(mean_img, cmap="viridis", vmin=0, vmax=1)
        ax_mean.set_title(f"{CLASS_NAMES[c]}", fontsize=9, fontweight="bold")
        ax_mean.set_xticks([])
        ax_mean.set_yticks([])
        if c == 0:
            ax_mean.set_ylabel("Mean Prototype\n($\\mu_c$)", fontweight="bold", fontsize=10)
            
        # Variance heatmap
        ax_std = axes[1, c]
        im_s = ax_std.imshow(std_img, cmap="inferno", vmin=0, vmax=0.5)
        ax_std.set_xticks([])
        ax_std.set_yticks([])
        if c == 0:
            ax_std.set_ylabel("Intra-Class Std\n($\\sigma_c$)", fontweight="bold", fontsize=10)
            
    fig.subplots_adjust(right=0.92)
    cbar_ax_m = fig.add_axes([0.93, 0.54, 0.012, 0.35])
    fig.colorbar(im_m, cax=cbar_ax_m, label="Mean Intensity")
    
    cbar_ax_s = fig.add_axes([0.93, 0.12, 0.012, 0.35])
    fig.colorbar(im_s, cax=cbar_ax_s, label="Pixel Std Dev")
    
    plt.suptitle("Class Prototypes & Intra-Class Pixel Variability\n(Illustrating Bilateral Symmetry and High-Variance Contours)", y=0.99)
    path = os.path.join(OUTPUT_DIR, "fig4_class_prototypes.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")


# ==============================================================================
# Figure 5: 2D Dimensionality Reduction (PCA Feature Space Projection)
# ==============================================================================
def plot_pca_clusters(train_images, train_labels, n_samples=3000):
    # Subsample for faster PCA & clear visual readability
    indices = np.random.RandomState(42).choice(len(train_images), n_samples, replace=False)
    X = train_images[indices].reshape(n_samples, -1) / 255.0
    y = train_labels[indices]
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#808080"
    ]
    
    for c in range(10):
        mask = (y == c)
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[c], label=f"{c}: {CLASS_NAMES[c]}",
            alpha=0.6, s=18, edgecolors="none"
        )
        
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance)", fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance)", fontweight="bold")
    ax.set_title("Fashion-MNIST 2D PCA Cluster Projection\n(Footwear vs. Upper Clothing vs. Trousers Separation)", pad=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, borderaxespad=0.)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_pca_clusters.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")


def main():
    print("=" * 60)
    print("STARTING FASHION-MNIST EDA GRAPH GENERATION")
    print("=" * 60)
    
    (train_img, train_lbl), (val_img, val_lbl), (test_img, test_lbl) = load_raw_data("./data")
    
    print(f"Loaded Train: {len(train_img):,} | Val: {len(val_img):,} | Test: {len(test_img):,}")
    
    print("\n[1/5] Generating Representative Samples Grid...")
    plot_representative_samples(train_img, train_lbl)
    
    print("[2/5] Generating Class Distribution & Partition Split...")
    plot_class_distribution(train_lbl, val_lbl, test_lbl)
    
    print("[3/5] Generating Pixel Intensity Distribution & Sparsity...")
    plot_pixel_distribution(train_img)
    
    print("[4/5] Generating Class Prototypes & Variance Heatmaps...")
    plot_class_prototypes(train_img, train_lbl)
    
    print("[5/5] Generating 2D PCA Cluster Separation...")
    plot_pca_clusters(train_img, train_lbl)
    
    print("\n" + "=" * 60)
    print("ALL 5 EDA FIGURES GENERATED IN: ./eda_figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
