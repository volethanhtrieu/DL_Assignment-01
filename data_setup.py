"""
Fashion-MNIST Data Pipeline Setup for CO3133 Assignment 01 (HCMUT)
Group Goat Specification:
- Benchmarked on Fashion-MNIST (10 classes: 28x28 grayscale)
- Stratified Split: Train (50,000), Validation (10,000), Test (10,000)
- Leakage Prevention: Zero overlap between partitions (Instance-level partition, Seed: 42)
- Normalization: Mean = 0.2860, Std = 0.3530
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

# Exact Fashion-MNIST Class Labels
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Dataset Statistics (Fashion-MNIST benchmark)
MEAN = 0.2860
STD = 0.3530


def get_transforms(augment: bool = False):
    """
    Returns torchvision transforms for Fashion-MNIST.
    
    Args:
        augment (bool): If True, applies data augmentations (RandomHorizontalFlip,
                        RandomAffine) suitable for spatial architectures (CNN, ViT).
    """
    transform_list = []
    
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ])
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    
    return transforms.Compose(transform_list)


def get_stratified_indices(targets, val_samples_per_class: int = 1000, seed: int = 42):
    """
    Creates stratified index splits to ensure exact class balance across Train and Val sets.
    - Train: 5,000 samples per class (50,000 total)
    - Val: 1,000 samples per class (10,000 total)
    """
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    if isinstance(targets, torch.Tensor):
        targets_arr = targets.numpy()
    else:
        targets_arr = np.asarray(targets)
    num_classes = len(np.unique(targets_arr))

    for c in range(num_classes):
        class_indices = np.where(targets_arr == c)[0]
        rng.shuffle(class_indices)
        val_idx = class_indices[:val_samples_per_class]
        train_idx = class_indices[val_samples_per_class:]
        
        val_indices.extend(val_idx)
        train_indices.extend(train_idx)

    # Shuffle combined indices
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def get_fashion_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    augment_train: bool = False,
):
    """
    Downloads Fashion-MNIST, constructs stratified train/val/test splits,
    and returns PyTorch DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    os.makedirs(data_dir, exist_ok=True)

    # Define transforms
    train_transform = get_transforms(augment=augment_train)
    eval_transform = get_transforms(augment=False)

    # Download raw datasets with respective transforms
    raw_train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    raw_val_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=eval_transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    # Perform stratified split
    train_indices, val_indices = get_stratified_indices(
        raw_train_dataset.targets, val_samples_per_class=1000, seed=seed
    )

    train_subset = Subset(raw_train_dataset, train_indices)
    val_subset = Subset(raw_val_dataset, val_indices)

    # PyTorch DataLoaders
    # Pin memory enables faster transfer to NVIDIA RTX 3060 GPU
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, CLASS_NAMES


def verify_dataset(train_loader, val_loader, test_loader):
    """
    Verifies dataset integrity, tensor dimensions, and statistical properties.
    """
    print("=" * 60)
    print("FASHION-MNIST DATA INTEGRITY CHECK")
    print("=" * 60)
    
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    total = n_train + n_val + n_test
    
    print(f"Training Samples:   {n_train:,} (5,000 / class)")
    print(f"Validation Samples: {n_val:,} (1,000 / class)")
    print(f"Test Samples:       {n_test:,} (1,000 / class)")
    print(f"Total Images:       {total:,}")
    print("-" * 60)

    # Inspect a single batch
    images, labels = next(iter(train_loader))
    print(f"Batch Tensor Shape: {images.shape} (B, C, H, W)")
    print(f"Batch Labels Shape: {labels.shape}")
    print(f"Data Type:          {images.dtype}")
    print(f"Pixel Range (Norm): [{images.min().item():.3f}, {images.max().item():.3f}]")
    print(f"Batch Mean:         {images.mean().item():.3f} (approx 0.0 post-normalization)")
    print(f"Batch Std:          {images.std().item():.3f}  (approx 1.0 post-normalization)")
    print("=" * 60)


if __name__ == "__main__":
    # Quick test execution
    print("Initializing Fashion-MNIST Data Pipeline...")
    train_dl, val_dl, test_dl, classes = get_fashion_mnist_dataloaders(
        data_dir="./data",
        batch_size=64,
        num_workers=0,  # 0 for safe execution across platforms
    )
    verify_dataset(train_dl, val_dl, test_dl)
    print("Dataset pipeline is ready for model training!")
