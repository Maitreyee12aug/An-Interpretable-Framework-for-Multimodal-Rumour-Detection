# Gated Consistency and Uncertainty-Aware Fusion: An Interpretable Framework for Multimodal Rumour Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-The%20Visual%20Computer-green)](DOI_LINK_HERE)
[![DOI](https://zenodo.org/badge/DOI/YOUR_ZENODO_DOI.svg)](YOUR_ZENODO_LINK_HERE)

---

> ⚠️ **Citation Notice:** This repository contains the official implementation
> of the manuscript submitted to The Visual Computer (Springer).
> If you use this code or build upon this work, please cite our paper using the
> BibTeX entry provided at the bottom of this README.

---

## Overview

This is the official PyTorch implementation of:

**"Gated Consistency and Uncertainty-Aware Fusion: An Interpretable Framework for Multimodal Rumour Detection"**

*Maitreyee Ganguly, Paramita Dey*
*Department of Information Technology, Government College of Engineering & Ceramic Technology, Kolkata, India*

We propose an interpretable multimodal rumour detection framework that combines:

- **Cross-Modal Consistency Scoring** via a CLIP-based Cross-Modal Attention GNN (YOLOv8 + spaCy + CLIP)
- **Dynamic Gating Mechanism** that uses the consistency score to modulate feature reliability per sample
- **Uncertainty-Aware Adaptive Fusion** via Monte Carlo Dropout to estimate modality confidence
- **Interpretable outputs** — modality weights, gate values, and predictive uncertainty are all exposed per sample

### Key Results on MiRAGeNews

| Split | Accuracy | Macro F1 |
|---|---|---|
| In-Domain Validation (NYT + MJ) | **98.40%** | **98.40%** |
| OOD: BBC + DALL·E 3 | 86.01% | 86.01% |
| OOD: CNN + DALL·E 3 | 85.36% | 85.36% |
| OOD: BBC + SDXL | 87.35% | 87.03% |
| OOD: CNN + SDXL | 84.58% | 83.53% |

---

## Architecture

```
Raw Image + Raw Caption
        │
        ├─────────────────────────────────────────────┐
        │                                             │
        ▼                                             ▼
┌──────────────────┐                     ┌──────────────────────┐
│  Image Branch    │                     │  Text Branch         │
│  ResNet-50       │                     │  DistilBERT          │
│  (blocks 3,4     │                     │  (last 4 layers      │
│   unfrozen)      │                     │   unfrozen)          │
│  → z_img ∈R^512  │                     │  → z_text ∈R^512     │
│  → u_img ∈R^1    │                     │  → u_text ∈R^1       │
└────────┬─────────┘                     └──────────┬───────────┘
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │  Cross-Modal Consistency    │
          │  YOLOv8 → image regions     │
          │  spaCy  → text segments     │
          │  CLIP   → embeddings        │
          │  GNN    → s_cons ∈ [0,1]    │
          └──────────────┬──────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │  Dynamic Gating Mechanism   │
          │  gate_in = [s_cons‖z_text   │
          │             ‖z_img]         │
          │  → z'_text = z_text ⊙ g_t  │
          │  → z'_img  = z_img  ⊙ g_i  │
          └──────────────┬──────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │  Uncertainty-Aware Fusion   │
          │  [z'_text‖z'_img‖u_t‖u_i]  │
          │  → w_text, w_img (Softmax)  │
          └──────────────┬──────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │  Final Classifier           │
          │  [z'_text‖z'_img‖s_cons]   │
          │  Linear(1025→128)→Linear(2) │
          │  → Real / Fake              │
          └─────────────────────────────┘
```

---

## Repository Structure

```
Interpretable-Multimodal-Rumour-Detection/
│
├── README.md
├── requirements.txt
├── LICENSE
├── config.py                    ← All hyperparameters
│
├── models/
│   ├── __init__.py
│   ├── text_encoder.py          ← Section 4.1: DistilBERT text branch
│   ├── consistency_gnn.py       ← Section 4.3: Cross-Modal Attention GNN
│   ├── dynamic_gate.py          ← Section 4.4: Dynamic Gating Mechanism
│   └── full_model.py            ← Sections 4.1–4.6: Full integrated model
│                                   (includes ImageEncoder, UncertaintyAwareFusion,
│                                    Classifier, mc_inference)
│
├── data/
│   ├── __init__.py
│   ├── preprocess.py            ← Section 3: Data cleaning & .npz saving
│   └── dataset.py               ← Section 3: PyTorch Dataset & DataLoader
│
├── train.py                     ← Section 5.1: Training loop
├── evaluate.py                  ← Section 5.2: In-domain & OOD evaluation
├── visualize.py                 ← Section 5.3: All interpretability figures
└── inference.py                 ← Single sample inference with explanations
```

---

## Requirements

### System Requirements
- Python >= 3.8
- CUDA >= 11.3 (recommended; CPU also supported)
- GPU with >= 8GB VRAM recommended for training

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Interpretable-Multimodal-Rumour-Detection.git
cd Interpretable-Multimodal-Rumour-Detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm
```

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| PyTorch | >= 1.12.0 | Deep learning framework |
| torchvision | >= 0.13.0 | ResNet-50 backbone |
| transformers | >= 4.20.0 | DistilBERT encoder |
| datasets | >= 2.0.0 | MiRAGeNews loading |
| ultralytics | >= 8.0.0 | YOLOv8 region detection |
| CLIP | openai/CLIP | Cross-modal embeddings |
| spaCy | >= 3.4.0 | Text segmentation |
| scikit-learn | >= 1.0.0 | Evaluation metrics |

---

## Dataset

This work uses the **MiRAGeNews** dataset, publicly available on HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("anson-huang/mirage-news")
```

The dataset contains:
- **10,000** training image-caption pairs (NYT + MidJourney)
- **1,250** validation pairs (In-Domain)
- **2,500** out-of-domain test pairs across 4 splits:
  - BBC + DALL·E 3
  - CNN + DALL·E 3
  - BBC + SDXL
  - CNN + SDXL

Labels: `0 = Real`, `1 = Fake (AI-generated image)`

---

## Usage

### Step 1 — Preprocess Data

```bash
python data/preprocess.py
```

This downloads the dataset, cleans captions, resizes images, and saves
`.npz` files to `./preprocessed_miragenews_hf/`.

### Step 2 — Compute Consistency Scores

Consistency scores are computed automatically during training if not
already cached. To pre-compute them explicitly:

```python
from models.consistency_gnn import CrossModalAttentionGNN, CrossModalConsistencyScorer
from data.dataset import load_or_compute_scores
from datasets import load_dataset

dataset = load_dataset("anson-huang/mirage-news")
gnn     = CrossModalAttentionGNN(clip_dim=512, hidden_dim=256)
scorer  = CrossModalConsistencyScorer(gnn_model=gnn, device='cuda')

scores = load_or_compute_scores(
    split_name='train',
    raw_dataset=dataset['train'],
    preprocessed_labels=train_labels,
    scorer=scorer,
)
```

### Step 3 — Train

```bash
python train.py
```

Training config (from `config.py`):
- Optimizer: AdamW, lr=2×10⁻⁴
- Batch size: 32
- Max epochs: 50 with early stopping (patience=5)
- Dropout: 0.3 throughout

Best model is saved to `./checkpoints/best_model.pt`.

### Step 4 — Evaluate

```bash
python evaluate.py
```

Reports Accuracy and Macro F1 on validation split.
Saves confusion matrix to `./plots/`.

### Step 5 — Single Sample Inference

```bash
python inference.py \
    --image ./sample/image.jpg \
    --caption "American troops landed at the National Palace." \
    --checkpoint ./checkpoints/best_model.pt
```

Output:
```
==================================================
  PREDICTION    : FAKE
  Confidence    : 0.9312
  s_cons        : 0.2847
  w_text        : 0.6821
  w_img         : 0.3179
  Uncertainty σ²: 0.000412
==================================================
```

### Step 6 — Visualize Interpretability

```bash
python visualize.py
```

Generates Figures 7–10 from the paper:
- Figure 7: Modality weight distributions by class
- Figure 8: Consistency score vs gate values
- Figure 9: Consistency score vs modality weights
- Figure 10: Qualitative sample analysis

---

## Hyperparameters

All hyperparameters are in `config.py`. 

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁴ |
| Batch Size | 32 |
| Max Epochs | 50 |
| Early Stopping Patience | 5 |
| Dropout Rate | 0.3 |
| MC Dropout Samples (T) | 20 |
| Projection Dimension | 512 |
| GNN Hidden Dim | 256 |
| GNN Attention Heads | 4 |
| Gate Hidden Dim | 512 |
| Classifier Hidden Dim | 128 |
| Max Text Length | 128 tokens |
| Image Size | 224 × 224 |

---

## Citation

If you use this code in your research, please cite our paper. 

---

## Contact

For questions or issues, please open a GitHub Issue or contact:

- **Maitreyee Ganguly** — maitreyee12aug@gmail.com
- **Paramita Dey** — dey.paramita77@gmail.com

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [MiRAGeNews Dataset](https://huggingface.co/datasets/anson-huang/mirage-news)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
