# CO3133: Deep Learning and Its Applications (Semester 261)
## Assignment 01: Foundations of Deep Learning Pipelines and Architectures
### Comprehensive Final Submission (Milestone 1 & Milestone 2) — Group Goat

---

## 👤 Project Authorship & Individual Contribution

| Member Name | Student ID | Assignment 01 Contribution (%) | Role & Responsibilities |
| :--- | :---: | :---: | :--- |
| **Hoàng Trọng Huy Minh** | **2452743** | **100% (Sole Contributor)** | **End-to-End Implementation:** Problem formulation, 5-figure EDA, leak-free stratified DataLoader (Seed 42), all 5 model architectures (Linear, MLP, Custom CNN, Bi-LSTM, Vision Transformer), training engine with Early Stopping on Macro-F1, 5-model comparative benchmark, sanity audit suite, 20-page LaTeX report (`Goat_A1_Report.pdf`), interactive GitHub Pages website. |
| Võ Lê Thành Triệu | 2453297 | 0% | No contribution to Assignment 1. |
| Nguyễn Huỳnh Nam Quốc | 2453089 | 0% | No contribution to Assignment 1. |

> **Academic Honor Declaration:** I, **Hoàng Trọng Huy Minh** (Student ID: 2452743), declare that I am the sole author and contributor for Assignment 1. All reported metrics, model weights, and experimental observations reflect genuine, reproducible training runs conducted on my local hardware (NVIDIA GeForce RTX 3060).

---

## 🔗 Project Links & Official Deliverables

- **Live GitHub Pages Webpage:** [https://volethanhtrieu.github.io/Deep-Learning-project-main-page/assignment-01/](https://volethanhtrieu.github.io/Deep-Learning-project-main-page/assignment-01/)
- **Main Project Portal:** [https://volethanhtrieu.github.io/Deep-Learning-project-main-page/](https://volethanhtrieu.github.io/Deep-Learning-project-main-page/)
- **Cumulative Final Technical Report (PDF):** [`Goat_A1_Report.pdf`](Goat_A1_Report.pdf) / [`docs/Report_Assignment_01.pdf`](docs/Report_Assignment_01.pdf)
- **Source Code Repository:** [https://github.com/volethanhtrieu/DL_Assignment-01](https://github.com/volethanhtrieu/DL_Assignment-01)
- **AI Usage Disclosure (9 Mandatory Fields):** [`AI_usage.md`](AI_usage.md)

---

## 📊 Master 5-Model Comparative Benchmark Results

All five models were trained from scratch under identical hardware (NVIDIA GeForce RTX 3060 Laptop GPU), identical stratified data splits ($50{,}000$ train / $10{,}000$ val / $10{,}000$ test, Seed 42), and identical optimization schedules (AdamW, Cosine Annealing, 10 epochs, Early Stopping on validation Macro-F1):

| Model Architecture | Milestone | Trainable Params | Test Loss | Test Accuracy (%) | Test Macro-F1 (%) | Train Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Classifier** (Softmax Regression) | M1 Baseline | **7,850** | 0.4392 | 84.49% | 84.37% | ~209.7s | **0.280 ms / sample** |
| **Multilayer Perceptron (MLP)** (256-128, ReLU, Dropout 0.2) | M1 Baseline | 235,146 | 0.3122 | 88.86% | 88.83% | ~201.6s | 0.540 ms / sample |
| **Custom 3-Stage CNN** (Conv2d, BatchNorm, MaxPool, AvgPool) | M2 Final | 94,186 | **0.2703** | **90.26%** | **90.26%** | ~210.7s | 0.540 ms / sample |
| **Bidirectional LSTM (Bi-LSTM)** (2-Layer, $H=64$, Row-Sequence) | M2 Final | 149,002 | 0.2912 | 89.65% | 89.61% | ~209.2s | 0.522 ms / sample |
| **Vision Transformer (ViT)** (Patch $4 \times 4$, 3 Blocks, 4 Heads) | M2 Final | 105,546 | 0.3293 | 88.07% | 88.05% | ~263.3s | 0.548 ms / sample |

### Key Inductive Bias Insights
1. **CNN achieves peak performance**: With only $94{,}186$ parameters, the Custom CNN surpasses the $235\text{k}$-parameter MLP by $+1.40\%$ accuracy while cutting parameters by $60\%$, validating the decisive value of 2D spatial locality and translation equivariance.
2. **Bi-LSTM captures vertical contours**: Row-by-row scanning maintains vertical memory, achieving $89.65\%$ accuracy and the highest individual class score on *Shirt* ($72.62\%$).
3. **ViT requires inductive bias or scale**: Trained entirely from scratch without ImageNet pretraining on $50\text{k}$ samples, ViT reaches a respectable $88.07\%$, but is constrained by the data hunger of global self-attention on low-resolution grayscale images.
4. **Non-linearity is mandatory**: MLP improves $+4.37\%$ over Linear, proving the non-linear manifold nature of apparel categories.

---

## 📁 Repository Structure

```text
DL_Assignment-01/
├── checkpoints/                 # Serialized PyTorch model weights (*.pth)
│   ├── best_linear.pth          # Best Linear model weights (84.49% Acc)
│   ├── best_mlp.pth             # Best MLP model weights (88.86% Acc)
│   ├── best_cnn.pth             # Best Custom CNN model weights (90.26% Acc)
│   ├── best_lstm.pth            # Best Bi-LSTM model weights (89.65% Acc)
│   └── best_vit.pth             # Best Vision Transformer model weights (88.07% Acc)
├── docs/                        # PDF reports and documentation
│   ├── Goat_A1_Report.pdf       # Submission-named 20-page technical report
│   └── Report_Assignment_01.pdf # Course handbook standardized report
├── eda_figures/                 # 5 Exploratory Data Analysis figures
│   ├── fig1_representative_samples.png
│   ├── fig2_class_distribution.png
│   ├── fig3_pixel_distribution.png
│   ├── fig4_class_prototypes.png
│   └── fig5_pca_clusters.png
├── outputs/                     # Generated experimental artifacts
│   ├── checkpoints/             # Per-epoch training history (JSON & CSV)
│   └── figures/                 # Learning curves, confusion matrices, prediction examples
├── data_setup.py                # Stratified 50k/10k/10k DataLoader (seed=42, zero leakage)
├── models.py                    # PyTorch implementations of all 5 model architectures
├── engine.py                    # Train, validate, fit with EarlyStopping & history tracking
├── metrics.py                   # Accuracy, Macro-F1, Confusion Matrix, parameter counting
├── train.py                     # Unified CLI training script for all 5 models
├── audit_sanity.py              # Automated 4-part sanity and integrity audit script
├── generate_eda_graphs.py       # Standalone EDA visualization generator
├── requirements.txt             # Frozen dependencies
├── AI_usage.md                  # Comprehensive AI usage disclosure (Handbook Sec 5)
├── Goat_A1_Report.tex           # LaTeX source of the 20-page academic report
├── report.pdf                   # Synchronized PDF report
└── README.md                    # Project overview & reproduction instructions
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone the official repository
git clone https://github.com/volethanhtrieu/DL_Assignment-01.git
cd DL_Assignment-01

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Sanity & Integrity Audits
Verify zero data leakage, gradient flow across all 5 models, and checkpoint restoration:
```bash
python audit_sanity.py
```

### 3. Generate Exploratory Data Analysis (EDA) Figures
```bash
python generate_eda_graphs.py
```

### 4. Train All 5 Models (or Single Models)
```bash
# Train all 5 models sequentially:
python train.py --model all --epochs 10 --batch_size 64 --lr 0.001

# Or train individual models:
python train.py --model linear --epochs 10
python train.py --model mlp --epochs 10
python train.py --model cnn --epochs 10
python train.py --model lstm --epochs 10
python train.py --model vit --epochs 10
```
All checkpoints (`.pth`), per-epoch history logs (`.csv` and `.json`), learning curves, and normalized confusion matrices will be saved automatically to `./outputs/` and `./checkpoints/`.
