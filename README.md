# CCMT Crop Disease — EfficientNet-B2 Evaluation Notebook

Evaluate a **pre-trained EfficientNet-B2** model that classifies crop diseases for **Cassava, Cashew, Maize, and Tomato**.  
The notebook is meant for **testing only** (no training here) and reproduces the same preprocessing used in training.

## Model achieves a test and validation accuracy of 91%
- **Model:** EfficientNet-B2 (from `timm`)
- **Input size:** `380×380`
- **Total classes:** **22** (see full list below)
- **Goal:** Validate real-world performance (accuracy, confusion matrix, error analysis, spot checks on 100 random images).

---

## Class List (22)

**Cassava (5)**  
- bacterial blight, brown spot, green mite, mosaic, healthy

**Cashew (5)**  
- gummosis, red rust, anthracnose, leaf miner, healthy

**Maize (7)**  
- leaf spot, leaf blight, fall armyworm, grasshopper, streak virus, leaf beetle, healthy

**Tomato (5)**  
- leaf curl, leaf blight, septoria leaf spot, verticillium wilt, healthy

> In code, labels are flattened into a single 22-class list (e.g., `"Cassava - bacterial blight"`, `"Maize - leaf spot"`, …).

---

## What’s Inside This Repo

- A Jupyter Notebook that:
  - Loads your **EfficientNet-B2** checkpoint.
  - Uses the **same transforms** as training for evaluation:
    - `Resize(380, 380) → ToTensor() → Normalize(mean=[0.5]*3, std=[0.5]*3)`
  - Evaluates on a **unified test set** (all 22 classes in one folder).
  - Reports:
    - Overall accuracy + per-class precision/recall/F1
    - Confusion Matrix heatmap
    - **100-image random spot check** (ground truth vs prediction)
    - Top-K predictions (optional)
    - Misclassification gallery (optional)

---

