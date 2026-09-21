"""
Model Architectures for CO3133 Assignment 01 (HCMUT)
Milestone 1 (M1) Deliverable — Group Goat

Mandatory Models Implemented:
1. LinearClassifier (Linear / Softmax Regression baseline):
   - Flatten -> Linear(784, 10)
   - Parameters: 7,850
2. MLPClassifier (Multilayer Perceptron):
   - Flatten -> Linear(784, 256) -> ReLU -> Dropout(0.2)
             -> Linear(256, 128) -> ReLU -> Dropout(0.2)
             -> Linear(128, 10)
   - Parameters: 235,146
"""

from typing import List, Optional
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LinearClassifier(nn.Module):
    """
    Linear / Softmax Regression Baseline.

    A single linear transformation mapping flattened 28x28 grayscale image pixels
    directly to 10 class logits: z = W*x + b.

    Specifications:
        - Input:  (B, 1, 28, 28) or (B, 784)
        - Output: (B, 10) unnormalized logits
        - Parameters: 784 * 10 + 10 = 7,850
        - Loss:   torch.nn.CrossEntropyLoss (applies LogSoftmax internally)
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)

        # Initialize weights with standard normal / Xavier
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten image to 1D feature vector: (B, 1, 28, 28) -> (B, 784)
        x = self.flatten(x)
        # Linear projection to class logits: (B, 784) -> (B, 10)
        logits = self.linear(x)
        return logits


class MLPClassifier(nn.Module):
    """
    Multilayer Perceptron (MLP) Classifier.

    A feedforward deep architecture designed according to Group Goat specifications:
    Two hidden layers (256 -> 128), ReLU non-linear activations, and Dropout regularization
    to overcome the linear separability bottleneck.

    Specifications:
        - Input:        (B, 1, 28, 28) or (B, 784)
        - Hidden Dims:  256 -> 128
        - Activations:  ReLU
        - Regularizer:  Dropout (p=0.2)
        - Output:       (B, 10) unnormalized logits
        - Parameters:   (784*256 + 256) + (256*128 + 128) + (128*10 + 10) = 235,146
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.flatten = nn.Flatten()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            linear_layer = nn.Linear(prev_dim, h_dim)
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity="relu")
            nn.init.zeros_(linear_layer.bias)

            layers.extend([
                linear_layer,
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
            prev_dim = h_dim

        # Final classification layer
        out_layer = nn.Linear(prev_dim, num_classes)
        nn.init.xavier_uniform_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 28, 28) -> (B, 784)
        x = self.flatten(x)
        # Forward through MLP layers -> (B, 10) logits
        logits = self.network(x)
        return logits


class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network (CNN) Classifier for Fashion-MNIST.

    A 3-stage convolutional backbone designed according to Group Goat specifications:
    - Block 1: Conv2d(1 -> 32, k=3, p=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2, 2)
    - Block 2: Conv2d(32 -> 64, k=3, p=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2, 2)
    - Block 3: Conv2d(64 -> 128, k=3, p=1) -> BatchNorm2d(128) -> ReLU -> AdaptiveAvgPool2d((1, 1))
    - Dropout(p=0.25)
    - Linear(128 -> 10)

    Total Parameters: ~94,186 (94k).
    Inductive bias: Strong spatial locality and translation equivariance.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, dropout_rate: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B, 1, 28, 28) -> (B, 32, 14, 14)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (B, 32, 14, 14) -> (B, 64, 7, 7)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (B, 64, 7, 7) -> (B, 128, 1, 1)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is flattened (B, 784), reshape to (B, 1, 28, 28)
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        feats = self.features(x)  # (B, 128, 1, 1)
        feats = torch.flatten(feats, 1)  # (B, 128)
        feats = self.dropout(feats)
        logits = self.classifier(feats)  # (B, 10)
        return logits


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to retrieve instantiated model by name.

    Args:
        model_name: 'linear', 'mlp', or 'cnn'.
    """
    name = model_name.lower().strip()
    if name in ["linear", "softmax", "linear_classifier"]:
        return LinearClassifier(**kwargs)
    elif name in ["mlp", "mlp_classifier", "multilayer_perceptron"]:
        return MLPClassifier(**kwargs)
    elif name in ["cnn", "cnn_classifier", "convnet"]:
        return CNNClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'linear', 'mlp', or 'cnn'.")


if __name__ == "__main__":
    print("=" * 70)
    print("M1 MODEL ARCHITECTURES SPECIFICATION CHECK")
    print("=" * 70)

    # 1. Linear Classifier Check
    linear_model = LinearClassifier()
    linear_params = count_parameters(linear_model)
    print(f"1. Linear Classifier: {linear_params:,} parameters (Expected: 7,850)")
    assert linear_params == 7850, f"Expected 7,850 params, got {linear_params}"

    # 2. MLP Classifier Check
    mlp_model = MLPClassifier()
    mlp_params = count_parameters(mlp_model)
    print(f"2. MLP Classifier:    {mlp_params:,} parameters (Expected: 235,146)")
    assert mlp_params == 235146, f"Expected 235,146 params, got {mlp_params}"

    # 3. CNN Classifier Check
    cnn_model = CNNClassifier()
    cnn_params = count_parameters(cnn_model)
    print(f"3. CNN Classifier:    {cnn_params:,} parameters (~94k)")

    # 4. Dummy Forward Pass Shape Verification
    dummy_input = torch.randn(16, 1, 28, 28)
    out_linear = linear_model(dummy_input)
    out_mlp = mlp_model(dummy_input)
    out_cnn = cnn_model(dummy_input)

    print("-" * 70)
    print(f"Input Shape:         {tuple(dummy_input.shape)}")
    print(f"Linear Output Shape: {tuple(out_linear.shape)} -> Expected (16, 10)")
    print(f"MLP Output Shape:    {tuple(out_mlp.shape)} -> Expected (16, 10)")
    print(f"CNN Output Shape:    {tuple(out_cnn.shape)} -> Expected (16, 10)")
    assert out_linear.shape == (16, 10) and out_mlp.shape == (16, 10) and out_cnn.shape == (16, 10)

    print("=" * 70)
    print("All architecture checks passed perfectly!")
    print("=" * 70)
