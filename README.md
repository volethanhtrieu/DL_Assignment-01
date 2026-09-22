# CO3133: Deep Learning and Its Applications (Semester 261)
## Assignment 01: Foundations of Deep Learning Pipelines and Architectures
### Milestone 1 (M1 Draft) — Fashion-MNIST Benchmark Study

---

## 👤 Project Authorship & Individual Contribution

| Member Name | Student ID | Assignment 01 Contribution (%) | Role & Responsibilities |
| :--- | :---: | :---: | :--- |
| **Hoàng Trọng Huy Minh** | **2452743** | **100% (Sole Contributor)** | **End-to-End Implementation:** Problem formulation, EDA (5 figures), data pipeline & stratified DataLoader, Linear and MLP model architectures, training engine with early stopping, evaluation metrics, LaTeX report, and GitHub Pages web page. |
| Võ Lê Thành Triệu | 2453297 | 0% | No contribution to Assignment 1. |
| Nguyễn Huỳnh Nam Quốc | 2453089 | 0% | No contribution to Assignment 1. |

> **Academic Honor Declaration:** I, **Hoàng Trọng Huy Minh** (Student ID: 2452743), declare that I am the sole author and contributor for Assignment 1. All reported numbers, model checkpoints, and experimental observations reflect genuine, verified training runs conducted on my local hardware.

---

## 🔗 Project Links & Deliverables

- **Live GitHub Pages Webpage:** [https://volethanhtrieu.github.io/Deep-Learning-project-main-page/assignment-01/](https://volethanhtrieu.github.io/Deep-Learning-project-main-page/assignment-01/)
- **Main Landing Page:** [https://volethanhtrieu.github.io/Deep-Learning-project-main-page/](https://volethanhtrieu.github.io/Deep-Learning-project-main-page/)
- **Technical Report (PDF):** [`docs/Report_Assignment_01.pdf`](docs/Report_Assignment_01.pdf)
- **Source Code Repository:** [https://github.com/volethanhtrieu/DL_Assignment-01](https://github.com/volethanhtrieu/DL_Assignment-01)

---

## 📊 Summary of M1 Quantitative Results

All models trained for 10 epochs using AdamW ($\text{lr}=10^{-3}$, `weight_decay` $=10^{-4}$),, batch size 64, CosineAnnealingLR scheduler, and Early Stopping with patience 4 on validation macro-F1:

| Model Architecture | Parameters | Test Accuracy (%) | Test Macro-F1 (%) | Train Time | Inference Latency (Batch 64) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Classifier** (Softmax Regression) | 7,850 | 84.49% | 84.37% | ~17.5s | 0.28 ms |
| **Multilayer Perceptron (MLP)** (256-128, ReLU, BN, Dropout 0.2) | 235,146 | 88.86% | 88.83% | ~23.1s | 0.54 ms |
| **Convolutional Neural Network (CNN)** (Custom 3-Stage ConvNet) | 94,186 | **90.26%** | **90.26%** | ~210.7s | 0.54 ms |

---

## 📁 Repository Structure

```text
DL_Assignment-01/
├── checkpoints/                 # Saved model weights
│   ├── best_linear.pth          # Best Linear model checkpoint (Val Macro-F1 = 84.37%)
│   ├── best_mlp.pth             # Best MLP model checkpoint (Val Macro-F1 = 88.83%)
│   └── best_cnn.pth             # Best CNN model checkpoint (Val Macro-F1 = 90.93%)
├── docs/                        # Project documentation & reports
│   └── Report_Assignment_01.pdf # Full Milestone 1 PDF Report
├── eda_figures/                 # 5 Exploratory Data Analysis graphs
│   ├── fig1_representative_samples.png
│   ├── fig2_class_distribution.png
│   ├── fig3_pixel_distribution.png
│   ├── fig4_class_prototypes.png
│   └── fig5_pca_clusters.png
├── outputs/                     # Training curves, confusion matrices, prediction examples
│   ├── checkpoints/             # History files (CSV & JSON)
│   └── figures/                 # Confusion matrices, curves, correct/incorrect samples
├── data_setup.py                # Stratified 50k/10k/10k DataLoader (seed=42, no leakage)
├── models.py                    # LinearClassifier & MLPClassifier definitions
├── engine.py                    # Train, validate, fit with EarlyStopping & history tracking
├── metrics.py                   # Accuracy, Macro-F1, Confusion Matrix, parameter count
├── train.py                     # Main CLI training script
├── generate_eda_graphs.py       # Standalone EDA generator
├── requirements.txt             # Python dependencies
├── AI_usage.md                  # Comprehensive AI usage disclosure (Handbook Sec 5)
└── README.md                    # Project overview & reproduction instructions
```

---

## 🚀 Quickstart & Reproducibility Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/volethanhtrieu/DL_Assignment-01.git
cd DL_Assignment-01

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate EDA Visualizations
```bash
python generate_eda_graphs.py
```

### 3. Run Training Pipeline (M1 Models)
```bash
# Train both Linear Classifier and MLP
python train.py --model all --epochs 10 --batch-size 64 --lr 0.001 --seed 42

# Or train individually:
python train.py --model linear --epochs 10
python train.py --model mlp --epochs 10
```

---

## 🤖 Academic Integrity & AI Usage Disclosure (Section 5)

> 📄 **Standalone Disclosure Document:** Please refer to [`AI_usage.md`](AI_usage.md) for the complete, formal disclosure table satisfying all 9 mandatory fields per tool (representative prompts, student verification logs, and the signed Academic Integrity Declaration).

All interactions with generative AI tools were conducted and verified exclusively by **Hoàng Trọng Huy Minh** (Student ID: **2452743**):

| Tool & Model | Used By | Task / Stage | AI Contribution | Verification & Sources |
| :--- | :--- | :--- | :--- | :--- |
| **Antigravity / Gemini** | Hoàng Trọng Huy Minh | Pipeline Architecture & Training Loop Engine | Assisted with modular code separation (`data_setup.py`, `engine.py`, `metrics.py`), DataLoader memory pinning, and EarlyStopping patience logic. | Verified via official PyTorch 2.4 documentation; tested loss convergence and stratified splits on controlled mini-batches. |
| **Antigravity / Gemini** | Hoàng Trọng Huy Minh | Linear & MLP Architecture Design | Suggested hidden layer dimensions (784 → 256 → 128 → 10), confirmed omitting softmax before CrossEntropyLoss per handbook rubric, and advised on BatchNorm/Dropout placement. | Verified parameter counts (7,850 and 235,146) against theoretical formulas; inspected validation accuracy stability. |
| **Antigravity / Gemini** | Hoàng Trọng Huy Minh | Evaluation Metrics & GitHub Pages UI | Assisted with macro-F1 calculation, CUDA inference latency timing, and structured HTML/CSS with MathJax 3 integration. | Manually verified confusion matrix diagonals against raw test predictions; validated math formula rendering in browser. |
