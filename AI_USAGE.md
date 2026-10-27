# Academic Integrity & Generative AI Usage Disclosure
## CO3133: Deep Learning and Its Applications — Assignment 01 (Semester 261)
### Group Goat — Milestone 1 & Milestone 2 (Cumulative Final Report)

---

## 📋 Course Policy Compliance Statement

In compliance with **Section 5 of the Course Project Handbook** (*"Academic Integrity and Generative AI Policies"*), this document transparently logs all generative AI tools used across coding, verification, and documentation for Assignment 1.

All interactions were conducted, scrutinized, and validated exclusively by **Hoàng Trọng Huy Minh (Student ID: 2452743)**, who is the sole author and contributor (100% individual contribution) for Assignment 1.

---

## 🤖 Detailed AI Interaction Log (Handbook Section 5.2)

| # | Mandatory Field | M1: Data Pipeline & Training Loop | M1 & M2: Architecture Design & Regularization | M2: Advanced Evaluation & Reporting |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Tool & Model Name** | Google Antigravity (Gemini 3.8 Flash) | Google Antigravity (Gemini 3.8 Flash) | Google Antigravity (Gemini 3.8 Flash) |
| **2** | **Who Used It** | Hoàng Trọng Huy Minh (2452743) | Hoàng Trọng Huy Minh (2452743) | Hoàng Trọng Huy Minh (2452743) |
| **3** | **Development Stage** | M1: Pipeline Setup & Training Loop | M1: Linear, MLP \| M2: CNN, Bi-LSTM, ViT | M2: Master Benchmarking, LaTeX Report, Web UI |
| **4** | **Purpose / Task** | Structuring modular pipeline (`data_setup.py`, `engine.py`), stratified 50k/10k/10k split, DataLoader pinning, and EarlyStopping patience logic | Designing Linear, MLP, 3-stage CNN, row-wise Bi-LSTM, and patch-based ViT architectures; parameter counting | Computing Macro-F1 across 10 classes, high-precision CUDA timing, and drafting academic 20-page LaTeX report (`Goat_A1_Report.tex`) |
| **5** | **Affected Files / Sections** | `data_setup.py`, `engine.py`, `train.py`, Report Sec 3 & 4 | `models.py`, `train.py`, Report Sec 5 | `metrics.py`, `audit_sanity.py`, `Goat_A1_Report.tex`, `assignment-01/index.html` |
| **6** | **Representative Prompt Example** | *"How to structure DataLoader with stratified 50k/10k/10k split with zero leakage and implement early stopping monitoring validation macro-F1 in PyTorch?"* | *"Implement ViT with 4x4 patch projection, 3 transformer blocks, and a 2-layer Bi-LSTM row-scanning model with ~149k parameters."* | *"Format comprehensive 5-model comparative benchmark table using tabularx and generate confusion matrices and error taxonomies."* |
| **7** | **AI Contribution** | Suggested modular file structure, DataLoader `pin_memory=True`, and early stopping delta logic. | Advised on patch projection via Conv2d, bidirectional hidden state concatenation, and layer norm placement; verified omitting softmax before `CrossEntropyLoss`. | Formatted LaTeX tables using `tabularx`/`adjustbox`, scikit-learn macro-F1 snippet, and CUDA synchronization for latency timing. |
| **8** | **Student Verification** | Verified disjoint train/val index sets (zero overlap), loss convergence, and gradient flow on local GPU. | Hand-calculated exact parameter counts for all 5 models (Linear: 7,850; MLP: 235,146; CNN: 94,186; Bi-LSTM: 149,002; ViT: 105,546); verified tensor output shapes `(B, 10)`. | Verified diagonal counts on 10x10 confusion matrices; audited gradient flow via `audit_sanity.py`; verified LaTeX compilation and cross-references. |
| **9** | **Sources Used for Verification** | Official PyTorch 2.4 documentation; Course Handbook rubric; scikit-learn official guide. | Deep Learning textbook (Goodfellow et al.); PyTorch `nn.Module` documentation; Dosovitskiy et al. (ViT paper); He/Kaiming initialization papers. | Scikit-learn classification metrics guide; LaTeX manual; MiKTeX x64 compiler. |

---

## ✍️ Academic Honor & Integrity Declaration

> **Student Declaration (Section 5.5):**  
> I, **Hoàng Trọng Huy Minh** (Student ID: **2452743**), solemnly declare that:
> 1. I am the sole author and contributor (100% individual contribution) for all code, experiments, analysis, model checkpoints, and reports submitted for Assignment 1 (Milestone 1 and Milestone 2).
> 2. Generative AI was employed strictly as an intelligent programming assistant and productivity tool under active human supervision, critical code review, and empirical verification.
> 3. No data, experimental results, model checkpoints, parameter counts, or citations were fabricated or hallucinated. All reported numbers reflect authentic training and evaluation runs conducted on local hardware (NVIDIA GeForce RTX 3060 Laptop GPU).
> 4. I fully understand, can explain, and take 100% academic responsibility for every line of code, formula, and narrative included in this project.
>
> **Student Signature:** *Hoàng Trọng Huy Minh*  
> **Date:** October 2026
