"""
Model Architectures for CO3133 Assignment 01 (HCMUT)
Milestone 2 (M2 Final) Deliverables — Group Goat

Mandatory Models Implemented (All 5 Architecture Families):
1. LinearClassifier (Linear / Softmax Regression baseline):
   - Flatten -> Linear(784, 10)
   - Parameters: 7,850
2. MLPClassifier (Multilayer Perceptron):
   - Flatten -> Linear(784, 256) -> ReLU -> Dropout(0.2)
             -> Linear(256, 128) -> ReLU -> Dropout(0.2)
             -> Linear(128, 10)
   - Parameters: 235,146
3. CNNClassifier (Custom 3-Stage Convolutional Neural Network):
   - 3x [Conv2d -> BatchNorm -> ReLU -> MaxPool/AdaptivePool] -> Linear(128, 10)
   - Parameters: 94,186
4. LSTMClassifier (Bidirectional Row-Sequence Recurrent Network):
   - Image as sequence of 28 rows (T=28, D=28) -> 2-layer Bi-LSTM (H=64) -> Linear(128, 10)
   - Parameters: 148,746
5. ViTClassifier (Vision Transformer with Patch Embeddings & Multi-Head Self-Attention):
   - 4x4 Patch Projection (49 tokens) + [CLS] Token + 1D PosEnc -> 3x TransformerEncoder -> Linear(64, 10)
   - Parameters: 105,546
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


class LSTMClassifier(nn.Module):
    """
    Recurrent Neural Network (Bi-LSTM / GRU) Classifier for Fashion-MNIST.

    Handbook Section 11.1 Requirement:
    - Sequence Representation: Treats each 28x28 grayscale image as a sequence of
      T = 28 timesteps, where each timestep vector x_t in R^28 represents a single row
      of 28 pixel values scanning top-to-bottom.
    - Architecture: 2-layer Bidirectional LSTM (hidden_size = 64).
      Forward and backward hidden states are concatenated at the final step (dim = 128).
    - Regularization: Dropout (p=0.2) between recurrent layers and before classification.
    - Inductive Bias: Sequential dependency and directional recurrence.

    Parameters: ~148,746 parameters.
    """

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
        bidirectional: bool = True,
        dropout_rate: float = 0.2,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.rnn_type = rnn_type.lower()

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        out_features = hidden_dim * self.num_directions
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(out_features, num_classes),
        )

        # Weight initialization
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Forget gate bias init to 1.0 for LSTM
                if self.rnn_type == "lstm":
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape handling:
        # (B, 1, 28, 28) -> (B, 28, 28): 28 timesteps of 28 row features
        if x.dim() == 4:
            x = x.squeeze(1)
        elif x.dim() == 2:
            x = x.view(-1, 28, 28)

        if self.rnn_type == "lstm":
            out, (h_n, c_n) = self.rnn(x)
        else:
            out, h_n = self.rnn(x)

        # Gather final hidden state representation
        if self.bidirectional:
            # Concatenate forward final state (h_n[-2]) and backward final state (h_n[-1])
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden_dim * 2)
        else:
            feat = h_n[-1]  # (B, hidden_dim)

        logits = self.classifier(feat)
        return logits


class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT) Classifier for Fashion-MNIST.

    Handbook Section 11.1 Requirement:
    - Tokenization / Patch Projection: Divides the 28x28 image into non-overlapping
      P x P patches (P = 4 -> 7x7 = 49 patches). Each patch of 16 pixels is projected
      linearly into a continuous token embedding space of dimension D = 64.
    - Class Token ([CLS]): A learnable token prepended to the patch sequence (L = 49 + 1 = 50 tokens).
    - Positional Encoding: Learnable 1D position embeddings added to preserve 2D topological order.
    - Transformer Encoder: 3-layer Pre-LayerNorm Transformer with Multi-Head Self-Attention (4 heads).
    - Inductive Bias: Extremely low; lacks 2D translation equivariance and locality priors.
      Learns all pairwise patch interactions through full self-attention.

    Parameters: ~105,546 parameters.
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 28,
        patch_size: int = 4,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        num_classes: int = 10,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
        self.num_patches = (img_size // patch_size) ** 2  # (28 // 4)**2 = 49
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 1. Patch projection via strided Conv2d (equiv to linear projection of flattened patches)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Learnable [CLS] token and 1D positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout_rate)

        # 3. Transformer Encoder Blocks (Pre-LayerNorm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Final LayerNorm & Classification Head on [CLS] token
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        B = x.shape[0]

        # Extract patch tokens: (B, 1, 28, 28) -> (B, embed_dim, 7, 7) -> (B, 49, embed_dim)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend [CLS] token: (B, 50, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding and apply dropout
        x = self.pos_drop(x + self.pos_embed)

        # Pass through Transformer Encoder
        x = self.transformer(x)

        # Extract [CLS] representation at index 0
        cls_out = self.norm(x[:, 0])

        # Classification logits: (B, 10)
        logits = self.head(cls_out)
        return logits


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to retrieve instantiated model by name.

    Args:
        model_name: 'linear', 'mlp', 'cnn', 'lstm', 'gru', or 'vit' / 'transformer'.
    """
    name = model_name.lower().strip()
    if name in ["linear", "softmax", "linear_classifier"]:
        return LinearClassifier(**kwargs)
    elif name in ["mlp", "mlp_classifier", "multilayer_perceptron"]:
        return MLPClassifier(**kwargs)
    elif name in ["cnn", "cnn_classifier", "convnet"]:
        return CNNClassifier(**kwargs)
    elif name in ["lstm", "lstm_classifier", "rnn", "bilstm"]:
        return LSTMClassifier(rnn_type="lstm", **kwargs)
    elif name in ["gru", "gru_classifier"]:
        return LSTMClassifier(rnn_type="gru", **kwargs)
    elif name in ["vit", "vit_classifier", "transformer", "vision_transformer"]:
        return ViTClassifier(**kwargs)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from: 'linear', 'mlp', 'cnn', 'lstm', 'gru', 'vit'."
        )


if __name__ == "__main__":
    print("=" * 80)
    print("M2 FINAL: 5 MANDATORY MODEL ARCHITECTURES SPECIFICATION CHECK")
    print("=" * 80)

    # Instantiate all 5 models
    models = {
        "1. Linear / Softmax": LinearClassifier(),
        "2. Multilayer Perceptron": MLPClassifier(),
        "3. Custom 3-Stage CNN": CNNClassifier(),
        "4. Bi-LSTM (Row Sequence)": LSTMClassifier(rnn_type="lstm"),
        "5. Vision Transformer (ViT)": ViTClassifier(),
    }

    dummy_input = torch.randn(16, 1, 28, 28)
    print(f"{'Model Architecture':<30} | {'Parameters':^12} | {'Output Shape':^15} | {'Status'}")
    print("-" * 80)

    for name, model in models.items():
        params = count_parameters(model)
        out = model(dummy_input)
        assert out.shape == (16, 10), f"Expected (16, 10), got {out.shape}"
        print(f"{name:<30} | {params:^12,d} | {str(tuple(out.shape)):^15} | PASS")

    print("=" * 80)
    print("All 5 mandatory model architectures passed specification check perfectly!")
    print("=" * 80)
