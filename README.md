<div align="center">

# Auto-Segmentation

### *Let AI agents do your segmentation research while you sleep.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

[![Claude Code](https://img.shields.io/badge/Claude%20Code-Agent-blueviolet)](https://claude.ai/code)
[![Codex](https://img.shields.io/badge/Codex-Agent-black)](https://openai.com/codex)
[![Cursor](https://img.shields.io/badge/Cursor-Agent-blue)](https://cursor.sh)

<br>

<table>
<tr><td align="left">

Tired of manually tuning architectures, loss functions, and hyperparameters for every new segmentation task?<br>
Wish you could run 100 experiments overnight and wake up to a better model?<br>
Want an AI that autonomously tries ideas, keeps what works, and discards what doesn't?

</td></tr>
</table>

### Auto-Segmentation does exactly that.

<br>

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for LLM pretraining — **generalized for any image segmentation task**.

Point it at your dataset · pick a metric · let the agent iterate overnight

Medical imaging · remote sensing · natural images · **any segmentation task**

**One config file + one AI agent = autonomous segmentation research**

<br>

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Agent Stages](#-what-the-agent-can-modify-in-trainpy) · [Metrics](#-supported-metrics) · [Examples](#-example-configs)

[**中文**](docs/README_ZH.md) · [**English**](README.md)

</div>

---

## 🚀 How it works

The repo contains only four files that matter:

| File | Purpose | Who modifies |
|------|---------|-------------|
| `prepare.py` | Data loading, evaluation metrics, dataset utilities | ❌ Read-only |
| `train.py` | Model, loss, optimizer, augmentation, training loop | 🤖 AI agent |
| `program.md` | Agent instructions and research strategy | 👨‍💻 Human |
| `config.yaml` | Dataset paths, evaluation metric, time budget, device | 👨‍💻 Human |

The agent modifies `train.py` — everything is fair game: architecture, loss function, augmentation, optimizer, hyperparameters. After each 5-minute training run, the primary metric (Dice, IoU, or HD95) determines whether the change is kept or discarded.

```
┌──────────────────────────────────────────────────────┐
│                   HUMAN SETS UP                       │
│  config.yaml: data path, metric, time budget          │
│  program.md:  research priorities & constraints        │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│                 AGENT LOOP (autonomous)               │
│                                                       │
│  1. Modify train.py (architecture, loss, augment...)  │
│  2. git commit                                        │
│  3. python train.py > run.log 2>&1                    │
│  4. Check primary metric                              │
│  5. Improved? → keep commit                           │
│     Not improved? → git reset                         │
│  6. Log to results.tsv                                │
│  7. GOTO 1                                            │
└──────────────────────────────────────────────────────┘
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Organize your data as:
```
/path/to/dataset/
├── images/          # Input images (PNG, JPG, NIfTI, or NPY)
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
└── masks/           # Segmentation masks (same format, same filenames)
    ├── sample_001.png
    ├── sample_002.png
    └── ...
```

Or generate a synthetic test dataset:
```bash
python prepare.py --synth --synth-dir ./synth_data
```

### 3. Configure

Edit `config.yaml` to point to your dataset:
```yaml
data:
  root: "/path/to/dataset"
  image_dir: "images"
  mask_dir: "masks"
  num_classes: 2          # including background
  image_size: 256
  val_split: 0.2

eval:
  primary_metric: "dice"  # dice | iou | hd95

time_budget: 300          # seconds (5 minutes)
device: "cuda:0"
```

Verify your setup:
```bash
python prepare.py
```

### 4. Run a single training experiment

```bash
python train.py
```

This trains for exactly 5 minutes (or your configured time budget), then evaluates and prints:
```
---
dice:             0.847300
iou:              0.734500
training_seconds: 300.1
peak_vram_mb:     8234.2
num_steps:        1523
num_params:       7,832,134
```

### 5. Run the agent

Point your AI coding agent (Claude Code, Codex, Cursor, etc.) at this repo and prompt:

```
Read program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will create a branch, establish a baseline, and start iterating autonomously.

## Supported data formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| Standard images | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif` | Grayscale or RGB |
| NIfTI | `.nii`, `.nii.gz` | 2D slices or 3D volumes (auto-handled) |
| NumPy | `.npy` | Raw arrays |

Image-mask pairs are matched by filename stem. nnU-Net-style suffixes (`_0000`) are automatically stripped.

## Supported metrics

| Metric | Config key | Direction | Notes |
|--------|-----------|-----------|-------|
| Dice score | `dice` | Higher is better | Per-class mean (excl. background) |
| IoU / Jaccard | `iou` | Higher is better | Per-class mean (excl. background) |
| Hausdorff 95% | `hd95` | Higher is better* | *Returned as negative internally |

## What the agent can modify in train.py

The training script is organized into clearly labeled stages:

| Stage | What it controls | Example changes |
|-------|-----------------|----------------|
| STAGE 1 | Data augmentation | Add elastic deform, cutmix, intensity jitter |
| STAGE 2 | Model architecture | UNet → Attention UNet, ResUNet, FPN, DeepLab |
| STAGE 3 | Loss function | DiceBCE → Focal, Tversky, boundary loss |
| STAGE 4 | Optimizer & scheduler | AdamW → SGD+momentum, cosine → poly decay |
| STAGE 5 | Training loop | Add AMP, EMA, gradient accumulation |

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Keeps scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly N minutes. Experiments are directly comparable regardless of architecture or hyperparameter changes.
- **Config-driven task definition.** Point `config.yaml` at any segmentation dataset — no code changes needed to switch tasks.
- **Format-agnostic data loading.** Supports natural images (PNG/JPG), medical images (NIfTI), and raw arrays (NPY) out of the box.
- **Self-contained.** No external model registries. The baseline UNet is written from scratch in train.py.

## Example configs

<details>
<summary>Medical image segmentation (binary)</summary>

```yaml
data:
  root: "/data/breast_ultrasound"
  image_dir: "images"
  mask_dir: "masks"
  num_classes: 2
  image_size: 256
  val_split: 0.2
eval:
  primary_metric: "dice"
time_budget: 300
device: "cuda:0"
```
</details>

<details>
<summary>Multi-class organ segmentation</summary>

```yaml
data:
  root: "/data/synapse_ct"
  image_dir: "images"
  mask_dir: "masks"
  num_classes: 9
  image_size: 512
  val_split: 0.2
eval:
  primary_metric: "dice"
time_budget: 600    # 10 minutes for larger images
device: "cuda:0"
```
</details>

<details>
<summary>Remote sensing segmentation</summary>

```yaml
data:
  root: "/data/satellite_buildings"
  image_dir: "images"
  mask_dir: "masks"
  num_classes: 2
  image_size: 512
  val_split: 0.15
eval:
  primary_metric: "iou"
time_budget: 300
device: "cuda:0"
```
</details>

## ⚠️ Limitations & How to Extend

This is a **research prototype**, not a production-ready system. However, it's designed to be **easily extensible**. Here are the key limitations and how you can overcome them:

### Architecture & Training
**Limitation:** Basic UNet only — no attention mechanisms, residual blocks, or modern architectural improvements
**🔧 How to fix:** Modify `STAGE 2` in `train.py`. Replace the `UNet` class with:
```python
# Add attention gates to skip connections
# Replace with ResUNet blocks (add residual connections)
# Implement FPN-style decoder
# Try DeepLabV3+ with atrous spatial pyramid pooling
```

**Limitation:** Simple augmentation — only random flips. No elastic deformations, color jittering, CutMix
**🔧 How to fix:** Modify `STAGE 1` in `train.py`. Add to `augment_batch()`:
```python
# Add elastic deformation using scipy.ndimage
# Implement color jittering for RGB images
# Add CutMix or Copy-Paste augmentation
# Try random rotation, scaling, and translation
```

**Limitation:** Basic loss function — only Dice + Cross-Entropy. No focal loss, Tversky loss, boundary-aware loss
**🔧 How to fix:** Modify `STAGE 3` in `train.py`. Replace `DiceBCELoss` with:
```python
# Add focal loss for class imbalance
# Implement Tversky loss with tunable alpha/beta
# Add boundary-weighted loss for edge refinement
# Combine multiple losses with learned weights
```

**Limitation:** Limited metrics — only Dice, IoU, and HD95. No per-class breakdown or confusion matrices
**🔧 How to fix:** Extend the evaluation section in `train.py`:
```python
# Add per-class Dice/IoU computation
# Implement confusion matrix analysis
# Add boundary-specific metrics (boundary Dice, Hausdorff surface)
# Track precision/recall for each class
```

### Training Infrastructure
**Limitation:** No model checkpointing — models are not saved, each experiment starts from scratch
**🔧 How to fix:** Add to `STAGE 5` in `train.py`:
```python
# Save best model: torch.save(model.state_dict(), "best_model.pth")
# Implement periodic checkpointing during training
# Add model loading for continued training or inference
# Export to ONNX for deployment
```

**Limitation:** Fixed time budget — training stops after N minutes regardless of convergence
**🔧 How to fix:** Modify the training loop in `STAGE 5`:
```python
# Add early stopping based on validation metric
# Implement learning rate scheduling with restarts
# Add validation during training (not just at end)
# Use metric plateau detection
```

**Limitation:** Single GPU only — no multi-GPU or distributed training support
**🔧 How to fix:** Wrap model with PyTorch distributed training:
```python
# Add: torch.nn.DataParallel(model) for simple multi-GPU
# Use torch.distributed for serious distributed training
# Implement gradient accumulation for larger effective batch size
# Add mixed precision (AMP) training for memory efficiency
```

**Limitation:** Minimal logging — only terminal output and TSV file. No TensorBoard, W&B, or detailed tracking
**🔧 How to fix:** Add logging in `STAGE 5`:
```python
# Integrate TensorBoard: from torch.utils.tensorboard import SummaryWriter
# Add Weights & Biases logging: import wandb; wandb.init()
# Save training curves and metric histories
# Log sample predictions during training
```

### Data Handling
**Limitation:** Simple preprocessing — basic resize and normalization. No intensity windowing or advanced preprocessing
**🔧 How to fix:** Extend `SegDataset` in `prepare.py`:
```python
# Add intensity windowing for medical images
# Implement histogram equalization or CLAHE
# Add z-score normalization per dataset
# Implement dataset-specific preprocessing hooks
```

**Limitation:** Basic data splitting — simple random split. No cross-validation or stratification
**🔧 How to fix:** Modify the dataset splitting logic:
```python
# Add k-fold cross-validation support
# Implement stratified splitting by class distribution
# Add domain-aware splitting for multi-center data
# Implement ensemble training across different splits
```

**Limitation:** 2D only — no native 3D training support. NIfTI volumes are sliced to 2D
**🔧 How to fix:** Modify data loading and model architecture:
```python
# Load full 3D volumes instead of slices
# Implement 3D UNet with 3D convolutions
# Add sliding window inference for large volumes
# Use memory-efficient 3D training with gradient checkpointing
```

### Research Limitations
**Limitation:** No hyperparameter search — experiments driven by AI agent intuition, not systematic HPO
**🔧 How to fix:** Add systematic search strategies:
```python
# Implement grid search over key hyperparameters
# Add Bayesian optimization for efficient search
# Use Optuna or Hyperopt for automated HPO
# Track hyperparameter importance analysis
```

**Limitation:** No ensembling — no model ensembling, test-time augmentation, or uncertainty estimation
**🔧 How to fix:** Add ensembling in `STAGE 5`:
```python
# Save multiple checkpoints and ensemble predictions
# Implement test-time augmentation (TTA)
# Add Monte Carlo dropout for uncertainty estimation
# Implement model averaging across different random seeds
```

**Limitation:** No interpretability — no attention maps, GradCAM, or prediction visualization
**🔧 How to fix:** Add visualization tools:
```python
# Implement GradCAM for attention visualization
# Add overlay of predictions on original images
# Create attention maps for transformer-based models
# Generate confidence maps and uncertainty visualizations
```

### Production Readiness
**Limitation:** No inference pipeline — no model export, ONNX conversion, or deployment tools
**🔧 How to fix:** Add deployment capabilities:
```python
# Create separate inference script with model loading
# Add ONNX export: torch.onnx.export(model, ...)
# Implement batch inference for production use
# Add REST API or command-line interface for inference
```

**Limitation:** Basic error handling — minimal validation and error recovery
**🔧 How to fix:** Add robust error handling:
```python
# Add input validation and sanity checks
# Implement graceful degradation on errors
# Add comprehensive logging and debugging tools
# Create test suite for validation
```

### The Flexibility Advantage

🎯 **Key Point:** Unlike "black-box" frameworks, Auto-Segmentation puts you in control:
- **Single file to modify** — All changes happen in `train.py`
- **Clear stages** — Each component is labeled and easy to find
- **Git-tracked experiments** — Every modification is recorded and reversible
- **No black magic** — You can see and modify everything

**You're never stuck.** If you hit a limitation:
1. Find the relevant stage in `train.py`
2. Modify the code to add what you need
3. Let the AI agent iterate on your changes
4. Compare results and keep what works

### When This Framework Shines
✅ Rapid prototyping of segmentation ideas
✅ Educational exploration of segmentation techniques
✅ Architecture search and experimentation
✅ Learning how different components affect performance
✅ Building custom segmentation solutions

### When to Use Established Frameworks
❌ Production medical image segmentation
❌ Published research requiring standardized baselines
❌ Competitions requiring state-of-the-art results
❌ Tasks where reliability and reproducibility are critical

## License

MIT License - see [LICENSE](LICENSE) file for details.

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch), which is also licensed under the MIT License.

## Acknowledgments

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original concept of AI-driven autonomous research
- [nanochat](https://github.com/karpathy/nanochat) — the training framework that inspired the single-file approach
- [nn-UNet](https://github.com/MIC-DKFZ/nnUNet) — the state-of-the-art medical image segmentation framework that inspired many of the techniques implemented here
