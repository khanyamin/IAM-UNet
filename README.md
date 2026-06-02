# IAM-UNet: A Hybrid State-Space and Multi-Scale Feature Fusion Network for Digital Rock Image Segmentation

<p align="center">
  <img src="architecture.png" alt="IAM-UNet Architecture" width="85%"/>
</p>

<p align="center">
  <a href="https://www.digitalrocksportal.org/projects/317/">
    <img src="https://img.shields.io/badge/Dataset-Digital%20Rocks%20Portal-green" alt="Dataset"/>
  </a>
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

**IAM-UNet** is a hybrid encoder-decoder segmentation framework that integrates 
the **InMamba block** a module containing an **Inception** sub-module for 
multi-scale feature extraction and an **SSM** sub-module for long-range dependency 
modeling and **Attention-guided** skip-connection fusion that suppresses less 
relevant information and highlights useful features across encoder-decoder paths.

---


---

##  Repository Structure

```
IAM-UNet/
├── net.py                  # Full model architecture
├── train.py                # Training script
├── test.py                 # Inference & evaluation script
├── calculate_matrix.py     # Evaluation metrics (Dice, IoU, Precision, Recall)
├── architecture.jpg        # Architecture diagram

```




## ⚙️ Installation

### Requirements

```bash
python >= 3.8
torch >= 1.12
torchvision
timm
Pillow
numpy
mamba_ssm        # Required for SS2D selective scan
```

### Install Dependencies

```bash
pip install torch torchvision timm Pillow numpy
pip install mamba_ssm   # CUDA required; see https://github.com/state-spaces/mamba
```

> **Note:** `mamba_ssm` requires a CUDA-capable GPU. If unavailable, a custom `selective_scan` fallback can be substituted (see `net.py` import block).

---

##  Usage

### 1. Prepare Dataset

Organize your data in the following structure:

```
datasets/
├── train/
│   ├── input/    ← grayscale micro-CT slices (.png / .jpg)
│   └── mask/     ← binary segmentation masks (same filenames)
└── test/
    ├── input/
    └── mask/
```

### 2. Training

```bash
python train.py
```

Key training hyperparameters (configurable in `train.py`):

| Parameter | Value |
|:---|:---:|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Beta₁ / Beta₂ | 0.5 / 0.999 |
| Loss function | MSE |
| Batch size | 8 |
| Epochs | 100 |
| Input size | 256 × 256 |
| Train/Val split | 8:2 |

Checkpoints are saved every 10 epochs and at the end of training to `savemodel/`. Training logs (Loss, Dice, IoU, Precision, Recall) are exported as CSV files.

### 3. Inference & Evaluation

```bash
python test.py
```

Loads `savemodel/net.pth`, runs inference on the test set, saves predicted masks to `savemodel/pred/`, and writes `test_metrics.txt` and per-image CSV.

### 4. Standalone Metric Calculation

```bash
python calculate_matrix.py
```

Computes average Dice, IoU, Precision, and Recall from saved predicted masks vs. ground-truth masks.

---

##  Evaluation Metrics

All metrics are computed with **pore pixels as the positive class** (masks are inverted before calculation, since pores are the minority region).

| Metric | Formula |
|:---|:---|
| **Dice** | 2·TP / (2·TP + FP + FN) |
| **IoU** | TP / (TP + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |


---

##  Experimental Setup

| Setting | Value |
|:---|:---|
| GPU | NVIDIA Tesla V100-PCIE (32 GB) |
| Framework | PyTorch |
| Input size | 256 × 256 pixels |
| Repeated runs | N = 5 (mean ± std reported) |
| Threshold | 0.5 (sigmoid output → binary mask) |

---


---

##  Acknowledgments

This work was supported by the **Deep Earth Probe and Mineral Resources Exploration Major Project, National Science and Technology**, under Grant No. **2025ZD1010908**.

The sandstone dataset is publicly available on the [Digital Rocks Portal](https://www.digitalrocksportal.org/projects/317/)

---

##  Contact

**Md. Yamin Khan**  
Email: [khanyamin687@gmail.com](mailto:khanyamin687@gmail.com)  
School of Resources and Safety Engineering  
Central South University, Changsha, 410083 Hunan, China

For questions or collaborations, please open an issue or contact via the repository.

---

<p align="center">
  <sub>© 2025 IAM-UNet Authors. All rights reserved.</sub>
</p>

