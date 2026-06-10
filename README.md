<h1 id="IAM-UNet">IAM-UNet: A Hybrid State-Space and Multi-Scale Feature Fusion Network for Digital Rock Image Segmentation</h1>
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

### Repository Structure

```
IAM-UNet/
├── net.py                  # Full model architecture
├── train.py                # Training script
├── test.py                 # Inference & evaluation script
├── calculate_matrix.py     # Evaluation metrics (Dice, IoU, Precision, Recall)
├── architecture.jpg        # Architecture diagram

```

### Requirements

<ul>
<li>Python==3.9.21</li>
<li>torch==2.6.0</li>
<li>torchvision</li>
<li>numpy==2.0.2
<li>mamba-ssm==2.3.1</li>
<li>causal-conv1d>=1.4.0</li>
</ul>

---

## Usage

### 1. Prepare Dataset

Organize data in the following structure:

```
datasets/
├── train/
│   ├── input/
│   └── mask/
└── test/
    ├── input/
    └── mask/
```

### 2. Training
<pre><code>python train.py</code></pre>

Key training hyperparameters:

| Parameter | Value |
|:---|:---:|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| β1 / β2 | 0.5 / 0.999 |
| Loss function | MSE |
| Batch size | 8 |
| Epochs | 100 |
| Input image size | 256 × 256 |
| Train/Val split | 8:2 |

### 3. Inference & Evaluation
<pre><code>python test.py</code></pre>

### 4. Standalone Metric Calculation
<pre><code>python calculate_matrix.py</code></pre>

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

---


