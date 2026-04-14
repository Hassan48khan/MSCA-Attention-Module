# MSCA: Multi-Scale Cross-Axis Attention Module

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A novel plug-and-play attention module for CNN backbones.**  
Extends EMA with tri-scale convolutions, cross-axis pooling fusion, and dual-gate recalibration — all without channel dimensionality reduction.

</div>

---

## 🔍 Overview

**MSCA (Multi-Scale Cross-Axis Attention)** builds upon the [EMA attention module](https://github.com/YOLOonMe/EMA-attention-module) and introduces three key innovations:

| Innovation | What it does |
|---|---|
| **Tri-scale parallel convolutions** | 1×1, 3×3, and 5×5 kernels run in parallel with learned fusion weights, capturing short-, mid-, and long-range spatial patterns simultaneously |
| **Cross-axis pooling fusion** | Combines horizontal, vertical, *and* diagonal average pooling via a lightweight 1×1 mixing layer, encoding richer positional context than H+W pooling alone |
| **Dual-gate recalibration** | Applies both a channel-wise sigmoid gate and a spatial sigmoid gate before the cross-spatial dot-product, sharpening attended regions and suppressing noise |

All operations maintain the original channel count — **no dimensionality reduction**.

---

## 🏗 Architecture

```
Input (B, C, H, W)
       │
   ┌───┴──────────────────────────────────────┐
   │         Reshape into G groups             │
   │         (B·G, C/G, H, W)                 │
   └───┬──────────────┬───────────────────────┘
       │              │
  ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
  │ 1×1 Conv│    │ 3×3 Conv│    │ 5×5 Conv│
  │ + Cross │    │         │    │         │
  │ Axis    │    │         │    │         │
  └────┬────┘    └────┬────┘    └────┬────┘
       │              │              │
       └──── Learned weighted sum ───┘
                      │
              ┌───────▼────────┐
              │  Dual-Gate      │
              │  (CH gate +     │
              │   SP gate)      │
              └───────┬────────┘
                      │
          ┌───────────▼───────────┐
          │  Cross-Spatial        │
          │  Dot-Product          │
          │  (pixel-level pairwise│
          │   relationship)       │
          └───────────┬───────────┘
                      │
              Sigmoid reweighting
                      │
               Output (B, C, H, W)
```

### Cross-Axis Pooling Fusion

Unlike EMA which only uses H and W pooling, MSCA adds a **diagonal pooling** branch:

```
H-pool  →  (B·G, C/G, H, 1)
W-pool  →  (B·G, C/G, 1, W)
D-pool  →  (B·G, C/G, 1, W)   ← 45° rotated then H-pooled
           └────── concat + 1×1 mix ──────┘
```

This lets the module encode spatial context along oblique axes — especially useful for object detection on drone imagery and diagonal structure recognition.

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/your-username/MSCA-attention-module.git
cd MSCA-attention-module
pip install torch torchvision
```

### Basic Usage

```python
import torch
from MSCA_attention_module import MSCA

# Create module
attn = MSCA(channels=256, groups=32, use_diag=True)

# Apply to any feature map
x   = torch.randn(2, 256, 32, 32)
out = attn(x)
print(out.shape)  # → torch.Size([2, 256, 32, 32])
```

### Drop into ResNet

```python
import torch.nn as nn
from torchvision.models import resnet50
from MSCA_attention_module import MSCA

class ResNetMSCA(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        base = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.attn     = MSCA(channels=2048, groups=32)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
```

### Drop into YOLOv5 / YOLOv8

Add the following to your model YAML `backbone` or `head` section:

```yaml
# In yolov5s.yaml — replace a C3 block with MSCA
- [-1, 1, MSCA, [256, 32]]   # [channels, groups]
```

Then register in `models/common.py`:

```python
from MSCA_attention_module import MSCA
```

---

## 🔧 API Reference

```python
MSCA(channels, groups=32, use_diag=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `channels` | `int` | — | Input (= output) channel count. Must be divisible by `groups`. |
| `groups` | `int` | `32` | Number of channel groups. Larger = fewer params per group. |
| `use_diag` | `bool` | `True` | Enable diagonal-axis pooling branch. Set `False` to match EMA overhead exactly. |

---

## 📊 Differences vs EMA

| Property | EMA | **MSCA** |
|---|---|---|
| Parallel conv branches | 2 (1×1, 3×3) | **3 (1×1, 3×3, 5×5)** |
| Pooling axes | H, W | **H, W, Diagonal** |
| Axis fusion | Concatenate → split | **Learnable 1×1 mix** |
| Pre-dot-product gate | None | **Dual gate (CH + SP)** |
| Scale fusion weights | Fixed (equal) | **Learned (softmax)** |
| Dimensionality reduction | None | None |

---

## 🧪 Smoke Test

```bash
python MSCA_attention_module.py
```

Expected output:

```
Shape                          Params      Time (ms)  Output OK
-----------------------------------------------------------------
(2,  64, 32, 32)            X,XXX         X.XXms          ✓
(2, 128, 16, 16)            X,XXX         X.XXms          ✓
(2, 256,  8,  8)            X,XXX         X.XXms          ✓
```

---

## 🗂 Repository Structure

```
MSCA-attention-module/
├── MSCA_attention_module.py   # Main module + smoke test
├── README.md
└── LICENSE
```

---

## 📐 Design Principles

1. **No channel dimensionality reduction** — all convolutions operate at full group channel width, avoiding information loss shown to hurt pixel-level regression tasks.
2. **Plug-and-play** — a single `nn.Module` with a shape-preserving `forward()`. Drop into any backbone with one line.
3. **Efficient parallelism** — group reshaping into the batch dimension means all groups process in parallel, matching modern GPU memory layouts.
4. **Tri-scale by default** — the 5×5 kernel adds only marginal parameter overhead per group but meaningfully enlarges the receptive field for large-scale features.

---

## 🔗 Related Work

- **EMA** — [Efficient Multi-Scale Attention Module with Cross-Spatial Learning](https://github.com/YOLOonMe/EMA-attention-module) (Ouyang et al., ICASSP 2023)
- **CA** — [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907) (Hou et al., CVPR 2021)
- **CBAM** — [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) (Woo et al., ECCV 2018)
- **ECA** — [Efficient Channel Attention for Deep CNNs](https://arxiv.org/abs/1910.03151) (Wang et al., CVPR 2020)
- **PSA** — [Polarized Self-Attention](https://arxiv.org/abs/2107.00782) (Liu et al., 2021)

---

## 📄 Citation

If you find MSCA useful in your research, please cite:

```bibtex
@misc{msca2024,
  title   = {MSCA: Multi-Scale Cross-Axis Attention Module},
  author  = {Your Name},
  year    = {2024},
  url     = {https://github.com/your-username/MSCA-attention-module}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
