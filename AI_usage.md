# Academic Integrity & Generative AI Usage Disclosure
## CO3133: Deep Learning and Its Applications — Assignment 01 (Semester 261)
### Group Goat — Milestone 1 (M1 Draft)

---

## 📋 Course Policy Compliance Statement

In compliance with **Section 5 of the Course Project Handbook** (*"Academic Integrity and Generative AI Policies"*), this document transparently logs all generative AI tools used across coding, verification, and documentation for Assignment 1.

All interactions were conducted and verified exclusively by **Hoàng Trọng Huy Minh (Student ID: 2452743)**, who is the sole author and contributor for Assignment 1.

---

## 🤖 Detailed AI Interaction Log (Handbook Section 5.2)

| # | Mandatory Field | Entry 1 | Entry 2 | Entry 3 |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Tool & Model Name** | Google Antigravity (Gemini 3.8 Flash) | Google Antigravity (Gemini 3.8 Flash) | Google Antigravity (Gemini 3.8 Flash) |
| **2** | **Who Used It** | Hoàng Trọng Huy Minh | Hoàng Trọng Huy Minh | Hoàng Trọng Huy Minh |
| **3** | **Development Stage** | Pipeline Setup & Training Loop | Model Architecture & Regularization | Evaluation Metrics & GitHub Pages Web UI |
| **4** | **Purpose / Task** | Structuring modular code (`data_setup.py`, `engine.py`), DataLoader pinning, and EarlyStopping patience logic | Designing Linear, MLP, and 3-stage CNN architectures, parameter calculations, and regularization placement | Computing Macro-F1, measuring inference latency, and building HTML/CSS report with MathJax 3 |
| **5** | **Affected Files / Sections** | `data_setup.py`, `engine.py`, `train.py`, Report Sec 3.2 | `models.py`, `train.py`, Report Sec 3.1 | `metrics.py`, `assignment-01/index.html`, Report Sec 4 |
| **6** | **Representative Prompt Example** | *"How to structure DataLoader with stratified 50k/10k/10k split and implement early stopping monitoring validation macro-F1 in PyTorch?"* | *"Design a 3-stage custom CNN with ~94k parameters using Conv2d, BatchNorm, ReLU, and AdaptiveAvgPool for Fashion-MNIST."* | *"Format confusion matrix, learning curves, and prediction samples into a handbook-compliant HTML layout with MathJax 3."* |
| **7** | **AI Contribution** | Suggested modular file structure, DataLoader `pin_memory=True`, and early stopping with delta threshold. | Recommended layer dimensions (784→256→128→10 for MLP, 32-64-128 for CNN), verified omitting softmax before `CrossEntropyLoss`. | Generated scikit-learn macro-F1 calculation snippet, CUDA synchronization for latency timing, and clean responsive CSS. |
| **8** | **Student Verification** | Tested loss reduction, gradient backprop, and verified zero data leakage between train/val/test splits. | Hand-calculated parameter counts (7,850 for Linear, 235,146 for MLP, 94,186 for CNN); verified tensor output shapes `(B, 10)`. | Manually cross-checked confusion matrix diagonal counts against test predictions; verified LaTeX formula rendering across browsers. |
| **9** | **Sources Used for Verification** | Official PyTorch 2.4 documentation; Course Handbook rubric; scikit-learn official guide. | Deep Learning textbook (Goodfellow et al.); PyTorch `nn.Module` documentation; He/Kaiming initialization papers. | Scikit-learn classification metrics guide; MathJax 3 documentation; W3C CSS standards. |

---

## ✍️ Academic Honor & Integrity Declaration

> **Student Declaration (Section 5.5):**  
> I, **Hoàng Trọng Huy Minh** (Student ID: **2452743**), solemnly declare that:
> 1. I am the sole author and contributor for all code, experiments, analysis, and reports submitted for Assignment 1.
> 2. Generative AI was employed strictly as an intelligent programming assistant and productivity tool under active human supervision and critical evaluation.
> 3. No data, experimental results, model checkpoints, parameter counts, or citations were fabricated or hallucinated. All reported numbers reflect authentic training and evaluation runs conducted on local hardware (NVIDIA GeForce RTX 3060 Laptop GPU).
> 4. I fully understand, can explain, and take 100% academic responsibility for every line of code, formula, and narrative included in this project.
>
> **Student Signature:** *Hoàng Trọng Huy Minh*  
> **Date:** September 22, 2026
