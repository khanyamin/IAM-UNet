# IAM-UNet: A Hybrid Inception-Attention-Mamba U-Net Model for Rock Image Segmentation

<p align="center">
  <img src="architecture.jpg" alt="IAM-UNet Architecture" width="85%"/>
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

> **IAM-UNet** is a hybrid encoder–decoder segmentation framework that integrates **Inception multi-scale convolutions**, **Mamba-based selective state-space modeling (SS2D)**, and **Attention-guided skip-connection fusion** within a unified U-Net topology. It is designed for accurate binary pore–solid segmentation of micro-CT digital rock images, and achieves state-of-the-art performance across pixel-level, morphological, and petrophysical evaluation criteria.

---

## 📄 Paper

**IAM-UNet: A Hybrid Inception-Attention-Mamba U-Net model for Rock Images Segmentation**

Md. Yamin Khan¹, Md Mazedur Rahman², Shams ul Hadi¹, Xibing Li¹, Linqi Huang¹

¹ School of Resources and Safety Engineering, Central South University, Changsha, 410083 Hunan, China  
² School of Computer Science and Technology, Beijing Institute of Technology, Beijing 100081, China

📌 *Submitted to Computers & Geosciences (Elsevier)*

---

## 🔍 Highlights

- 🏆 **Best performance** across all 4 metrics (Dice, IoU, Precision, Recall) vs. 9 baseline models
- 🧠 Novel **Inception–Mamba (IM) block** combining local multi-scale and global long-range feature extraction
- 🎯 **Attention Gate** on every skip connection suppresses irrelevant grain-matrix responses
- 🔬 Validated with **morphological analysis** (S₂, C₂, L₂, CLD) and **petrophysical simulation** (FEM + LBM)
- 💧 Permeability relative error of only **0.22557%** — the lowest among all compared models

---

## 📊 Quantitative Results

### Segmentation Performance (Table 1)

| Model            | Dice       | IoU        | Precision  | Recall     |
|:-----------------|:----------:|:----------:|:----------:|:----------:|
| SegNet           | 0.934792   | 0.878070   | 0.929123   | 0.940940   |
| Dual-SegNet      | 0.961059   | 0.925175   | 0.951874   | 0.970662   |
| ResSegNet        | 0.946157   | 0.898140   | 0.935038   | 0.958014   |
| Dual-ResSegNet   | 0.957314   | 0.918299   | 0.954950   | 0.960024   |
| U-Net            | 0.960587   | 0.924347   | 0.959661   | 0.961755   |
| Dual-U-Net       | 0.954943   | 0.914143   | 0.961542   | 0.949072   |
| ResUNet          | 0.964715   | 0.931962   | 0.968592   | 0.960973   |
| Dual-ResUNet     | 0.961804   | 0.926565   | 0.956075   | 0.967807   |
| DeepLabv3+       | 0.955410   | 0.914840   | 0.952538   | 0.958428   |
| **IAM-UNet**     | **0.990513** | **0.981241** | **0.989511** | **0.991538** |

### Permeability Estimation (Table 2)

| Model            | Permeability (mD) | Relative Error (%) |
|:-----------------|:-----------------:|:-----------------:|
| Ground Truth     | 179.545           | —                 |
| SegNet           | 185.855           | 3.51444           |
| ResUNet          | 178.101           | 0.80426           |
| DeepLabv3+       | 181.618           | 1.15459           |
| **IAM-UNet**     | **179.950**       | **0.22557**       |

---

## 🏗️ Architecture Overview

IAM-UNet follows a symmetric encoder–decoder U-Net topology with three core innovations:

```
Input (1×256×256)
      │
  ┌───▼─────────────────────────────────────┐
  │         ENCODER (4 stages)              │
  │  [InceptionMambaBlock → MaxPool] × 4    │
  └─────────────────────────────────────────┘
      │
  ┌───▼─────────────────────────────────────┐
  │           BOTTLENECK                    │
  │       InceptionMambaBlock               │
  └─────────────────────────────────────────┘
      │
  ┌───▼─────────────────────────────────────┐
  │         DECODER (4 stages)              │
  │  [AttentionGate + Upsample + InceptionBlock] × 4  │
  └─────────────────────────────────────────┘
      │
  Conv1×1 + Sigmoid
      │
Output (1×256×256)   ← binary pore mask
```

### Key Components

| Component | Description |
|:---|:---|
| **InceptionBlock** | 4 parallel branches (1×1, 3×3, 5×5, MaxPool+1×1) for multi-scale local feature extraction |
| **SS2D** | Mamba-style 2D Selective State Space — scans in 4 spatial directions (H, W, and reversed) for linear-complexity global context modeling |
| **ProposedVSSBlock** | Gated VSS: LayerNorm → depthwise conv → SS2D, modulated by a SiLU gating branch + residual |
| **InceptionMambaBlock (IM)** | Sequential fusion of InceptionBlock and ProposedVSSBlock |
| **Attention_block (AG)** | Generates a spatial attention map from decoder gating signal and encoder skip feature to suppress background noise |
| **UpBlock** | Attention gate → channel reduction → bilinear upsample → concat → InceptionBlock |

**Channel progression:** `1 → 64 → 128 → 256 → 512 → 1024` (bottleneck), then symmetric decoder.

---

## 📁 Repository Structure

```
IAM-UNet/
├── net.py                  # Full model architecture
├── train.py                # Training script
├── test.py                 # Inference & evaluation script
├── calculate_matrix.py     # Evaluation metrics (Dice, IoU, Precision, Recall)
├── architecture.jpg        # Architecture diagram
├── datasets/
│   ├── train/
│   │   ├── input/          # Training micro-CT images
│   │   └── mask/           # Training binary masks
│   └── test/
│       ├── input/          # Test micro-CT images
│       └── mask/           # Test binary masks
└── savemodel/              # Saved weights and results (auto-created)
    └── pred/               # Predicted masks (auto-created)
```

---

## 🗂️ Dataset

The dataset consists of **6 sandstone types** obtained from the [Digital Rocks Portal](https://www.digitalrocksportal.org/projects/317/):

| Sandstone | Description |
|:---|:---|
| Berea | Classic benchmark sandstone |
| Bandera Brown | Heterogeneous pore structure |
| Bentheimer | Well-sorted, high-porosity |
| Kirby | Complex pore geometry |
| Leopard | Large grain size variability |
| Parker | Fine-grained low-porosity |

- **Volume size:** 1000 × 1000 × 1000 voxels at 2.25 μm resolution
- **Connected porosity:** 14–27% (mean 20 ± 3%)
- **Permeability:** 9–386 mD (mean 150 ± 40 mD)
- **2D slices used:** 3250 total → **2400 train / 600 validation / 250 test**
- **Image size:** 256 × 256 pixels (grayscale)
- **Masks:** Black = pore, White = solid matrix

> Preprocessing: Non-local means denoising applied to suppress scanner noise while preserving pore boundaries.

---

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

## 🚀 Usage

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
| Train/Val split | 80% / 20% |

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

## 📐 Evaluation Metrics

All metrics are computed with **pore pixels as the positive class** (masks are inverted before calculation, since pores are the minority region).

| Metric | Formula |
|:---|:---|
| **Dice** | 2·TP / (2·TP + FP + FN) |
| **IoU** | TP / (TP + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |

In addition to pixel-level metrics, the paper reports:
- **Morphological validation:** S₂(r), C₂(r), L₂(r), CLD(r), PSD
- **Elastic properties (FEM):** Bulk modulus, Shear modulus, P-wave velocity, S-wave velocity
- **Permeability (LBM):** Absolute permeability with relative error

---

## 🔬 Experimental Setup

| Setting | Value |
|:---|:---|
| GPU | NVIDIA Tesla V100-PCIE (32 GB) |
| Framework | PyTorch |
| Input size | 256 × 256 pixels |
| Repeated runs | N = 5 (mean ± std reported) |
| Threshold | 0.5 (sigmoid output → binary mask) |

---

## 📈 Training Outputs

After training, the following files are saved in `savemodel/`:

```
savemodel/
├── net.pth                      # Final model weights
├── net_10.pth ... net_100.pth   # Epoch checkpoints (every 10 epochs)
├── Train Loss.csv
├── Val Loss.csv
├── Train Dice.csv  / Val Dice.csv
├── Train IoU.csv   / Val IoU.csv
├── Train Precision.csv / Val Precision.csv
├── Train Recall.csv    / Val Recall.csv
├── test_metrics.txt             # Average test metrics
├── test_metrics_per_image.csv   # Per-image test metrics
└── pred/                        # Binary prediction images
```

---

## 🧪 Known Issues & Notes

> ⚠️ **Import name mismatch:** `train.py` and `test.py` import from `matrix`, but the metrics file is named `calculate_matrix.py`. Rename it or add an alias:
> ```bash
> # Option 1: rename the file
> mv calculate_matrix.py matrix.py
> # Option 2: create a soft link / copy
> cp calculate_matrix.py matrix.py
> ```

> ⚠️ **Mamba dependency:** `SS2D` requires `mamba_ssm` with a CUDA GPU. CPU-only environments will raise an `ImportError` at `forward_core()`.

---

## 📚 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{khan2025iamunet,
  title   = {IAM-UNet: A Hybrid Inception-Attention-Mamba U-Net model for Rock Images Segmentation},
  author  = {Khan, Md. Yamin and Rahman, Md Mazedur and Hadi, Shams ul and Li, Xibing and Huang, Linqi},
  journal = {Computers \& Geosciences},
  year    = {2025},
  publisher = {Elsevier}
}
```

---

## 🙏 Acknowledgments

This work was supported by the **Deep Earth Probe and Mineral Resources Exploration Major Project, National Science and Technology**, under Grant No. **2025ZD1010908**.

The sandstone dataset is publicly available on the [Digital Rocks Portal](https://www.digitalrocksportal.org/projects/317/) (Project #317).

The SS2D module is adapted from [VMamba](https://github.com/MzeroMiko/VMamba) (Liu et al., 2024).  
The Attention Gate is adapted from [Attention U-Net](https://arxiv.org/abs/1804.03999) (Oktay et al., 2018).

---

## 📬 Contact

**Md. Yamin Khan**  
School of Resources and Safety Engineering  
Central South University, Changsha, 410083 Hunan, China

For questions or collaborations, please open an issue or contact via the repository.

---

<p align="center">
  <sub>© 2025 IAM-UNet Authors. All rights reserved.</sub>
</p>

