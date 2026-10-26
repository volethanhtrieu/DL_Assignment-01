# Presentation Defense Script: Assignment 01 (CO3133)
## Title: Foundations of Deep Learning Pipelines & Architectures
### Group: Group Goat | Sole Contributor: Hoàng Trọng Huy Minh (ID: 2452743)
### Course: CO3133: Deep Learning and Its Applications (Semester 261) — HCMUT - VNU-HCM
### Instructor: Assoc. Prof. Lê Thành Sách

---

> **YouTube Video Requirements (Handbook Section 2.3):**
> - **Unified Video Title:** `CO3133-Semester-261 – Group Goat – Assignment 1`
> - **Target Duration:** 8 to 10 minutes.
> - **Visibility:** Unlisted or Public.
> - **Defense Format:** Slide presentation + brief code/demo walkthrough.

---

### [00:00 - 01:00] Slide 1 & 2: Introduction & Authorship
- **Spoken Text:**
  > "Good morning / afternoon, Professor Lê Thành Sách and peers. My name is **Hoàng Trọng Huy Minh**, Student ID **2452743**, representing **Group Goat** in course **CO3133: Deep Learning and Its Applications**, Semester 261 at HCMUT - VNU-HCM.
  >
  > Today, I am defending **Assignment 1**, titled: *'Foundations of Deep Learning Pipelines and Architectures: From Linear Models to Modern Sequence Models — A Comparative Study for Image Classification'*.
  >
  > As declared in our official repository, technical report, and project website, I am the **sole author and contributor (100% contribution)** for Assignment 1, having implemented the entire end-to-end pipeline, the five model architectures, the evaluation harness, and written the final 20-page technical report.
  >
  > Today's presentation is divided into four key sections: First, the problem formulation and leakage-free data pipeline; second, our M1 baseline models; third, the modern spatial, recurrent, and attention architectures from M2; and finally, our master benchmark, systematic error taxonomy, and reproducibility audits."

---

### [01:00 - 02:30] Slide 3, 4 & 5: Problem Formulation, EDA & Pipeline Design
- **Spoken Text:**
  > "Moving to Slide 3: While classic MNIST digits are easily separable with simple linear classifiers, real-world visual perception deals with continuous textures, concave boundaries, and high intra-class variance. We evaluate on **Fashion-MNIST**, comprising 70,000 $28 \times 28$ grayscale images across 10 apparel categories.
  >
  > In Slide 4, our Exploratory Data Analysis reveals crucial patterns: Class prototypes show that while *Trouser* and footwear have distinct geometric footprints, upper-body garments—namely *T-shirt/top*, *Shirt*, *Pullover*, and *Coat*—share almost identical pixel mass distributions. Furthermore, our 2D PCA manifold projection demonstrates significant linear overlap among these garment classes, proving that simple linear separation will inherently hit a ceiling.
  >
  > In Slide 5, we show our unified, leakage-free pipeline. We established a strict stratified partition under Seed 42: exactly 50,000 samples for training, 10,000 for validation, and 10,000 for testing, ensuring zero sample overlap. All models are trained under identical conditions: normalized inputs with mean 0.2860 and standard deviation 0.3530, AdamW optimizer with initial learning rate 0.001, Cosine Annealing scheduler, and Early Stopping with patience of 4 epochs on validation Macro-F1."

---

### [02:30 - 04:00] Slide 6 & 7: Milestone 1 Baselines (Linear vs. MLP)
- **Spoken Text:**
  > "Next, let us look at our M1 baseline models. Slide 6 presents our **Linear Classifier**, or Softmax Regression. Each image is flattened into a 784-dimensional vector and projected through a single weight matrix into 10 class logits. As mandated by the course handbook, we pass raw logits directly to PyTorch's `CrossEntropyLoss` without applying a premature softmax. With just 7,850 parameters, the Linear model reaches an accuracy of **84.49%** and a Macro-F1 of **84.37%**, serving as our lower-bound reference. Because its decision boundaries are hyperplanes, it struggles severely on ambiguous garments.
  >
  > On Slide 7, we introduce non-linearity with our **Multilayer Perceptron (MLP)**. It features two hidden layers of 256 and 128 units with Batch Normalization, ReLU activations, and Dropout of 0.2. With 235,146 parameters, the MLP pushes accuracy to **88.86%**—a decisive $+4.37\%$ improvement over the Linear classifier. This confirms that apparel manifolds are non-linear. However, because the MLP operates on flattened 1D vectors, it completely discards 2D spatial locality and pixel proximity."

---

### [04:00 - 06:00] Slide 8, 9 & 10: Milestone 2 Modern Architectures (CNN, Bi-LSTM, ViT)
- **Spoken Text:**
  > "To address spatial and sequential relationships, Milestone 2 expands our study to three modern architecture families.
  >
  > In Slide 8, we present our **Custom 3-Stage CNN**, designed completely from scratch. It features three hierarchical convolutional stages with $3 \times 3$ kernels, Batch Normalization, ReLU, and Max Pooling, followed by Adaptive Average Pooling and a compact linear head. By incorporating **translation equivariance** and **local receptive fields**, the CNN achieves the highest performance across all models: **90.26% test accuracy** and **90.26% Macro-F1**. Most remarkably, it does so with only **94,186 parameters**—cutting parameter overhead by **60%** compared to the MLP!
  >
  > In Slide 9, we re-conceptualize the 2D image as a sequence in our **Bidirectional LSTM**. An image is treated as a sequence of 28 rows, where each row is a 28-dimensional feature vector. A 2-layer Bi-LSTM with hidden size 64 processes the image from top-to-bottom and bottom-to-top simultaneously. The Bi-LSTM achieves **89.65% accuracy** and, fascinatingly, scores the highest individual F1 on the difficult *Shirt* class (72.62%), because its recurrent state captures vertical structural transitions from collar down to hem.
  >
  > In Slide 10, we implement a **Vision Transformer (ViT)** from scratch. We tokenize each image into 49 non-overlapping $4 \times 4$ patches, project them to dimension 48, add learnable 1D positional encodings, prepend a learnable `[CLS]` token, and process them through 3 Transformer encoder blocks with 4 attention heads each. Without pre-training on large datasets like ImageNet, ViT attains **88.07% accuracy**. While self-attention captures global dependencies, training from scratch on 50,000 samples illustrates the data hunger of Transformer architectures in low-sample regimes."

---

### [06:00 - 07:30] Slide 11 & 12: Master Benchmark & Per-Class Progression
- **Spoken Text:**
  > "Slide 11 presents our master comparison table. We evaluate beyond accuracy alone:
  > - **Linear**: 7,850 params | 84.49% Acc | 84.37% F1 | 0.280 ms latency (fastest edge inference).
  > - **MLP**: 235,146 params | 88.86% Acc | 88.83% F1 | 0.540 ms latency.
  > - **Custom CNN**: 94,186 params | **90.26% Acc** | **90.26% F1** | 0.540 ms latency (optimal accuracy-to-parameter ratio).
  > - **Bi-LSTM**: 149,002 params | 89.65% Acc | 89.61% F1 | 0.522 ms latency.
  > - **ViT**: 105,546 params | 88.07% Acc | 88.05% F1 | 0.548 ms latency.
  >
  > Slide 12 breaks down per-class Macro-F1 progression. Structured classes like *Trouser* (98.54%), *Bag* (98.10%), and *Sandal* (97.15%) achieve near-perfect classification across all models. The primary bottleneck is *Shirt*, improving from 61.25% in Linear up to 72.62% in Bi-LSTM."

---

### [07:30 - 09:00] Slide 13, 14 & 15: Error Taxonomy & Inductive Bias Analysis
- **Spoken Text:**
  > "Slide 13 and 14 detail our systematic error taxonomy and qualitative failure modes:
  > - **Error Type A (Garment Silhouette Ambiguity)**: Responsible for 65% of all errors. Confusions between *Shirt*, *T-shirt*, and *Coat* occur due to identical collar geometry and missing texture depth in grayscale.
  > - **Error Type B (Footwear Transition)**: Minor confusions between *Ankle boot* and *Sneaker* when cropped near the ankle collar.
  > - **Error Type C (Linear Feature Collisions)**: Pullovers versus coats when button/zipper cues are lost in low resolution.
  >
  > In Slide 15, we synthesize our core inductive bias insight: When training from scratch without billion-parameter pre-training, **hard spatial inductive biases (convolution and pooling) decisively outperform weak inductive biases (global self-attention and dense layers)**. The CNN learns local edge detectors and translation invariance inherently, whereas ViT must dedicate model capacity to learn spatial relationships from scratch."

---

### [09:00 - 10:00] Slide 16, 17 & 18: Integrity Audits, Deliverables & Conclusion
- **Spoken Text:**
  > "To ensure complete academic integrity and reproducibility, Slide 16 highlights our automated sanity suite (`audit_sanity.py`), which passed all 4 audits:
  > 1. Zero data leakage between partitions;
  > 2. Healthy non-zero gradient flow across 100% of parameters;
  > 3. Deterministic evaluation behavior under `model.eval()`;
  > 4. Strict checkpoint state dictionary loading.
  >
  > All generative AI tool interactions are fully logged with all 9 mandatory fields in `AI_usage.md` and Section 7 of the report.
  >
  > On Slide 17, all deliverables are live: our code repository, serialized checkpoints, interactive GitHub Pages website, presentation slides, and our 20-page technical report.
  >
  > In conclusion, Assignment 1 demonstrates the critical interplay between inductive bias, architectural capacity, and dataset scale. Thank you very much for your attention, and I welcome any questions regarding the models or experimental findings!"
