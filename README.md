<div align="center">

# 🔬 Auto-Seg

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

🧠 &nbsp;Tired of manually tuning architectures, loss functions, and hyperparameters for every new segmentation task?<br>
🌙 &nbsp;Wish you could run 100 experiments overnight and wake up to a better model?<br>
🔄 &nbsp;Want an AI that autonomously tries ideas, keeps what works, and discards what doesn't?

</td></tr>
</table>

### ✨ Auto-Seg does exactly that.

<br>

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for LLM pretraining — **generalized for any image segmentation task**.

Point it at your dataset · pick a metric · let the agent iterate overnight

Medical imaging · remote sensing · natural images · **any segmentation task**

**One config file + one AI agent = autonomous segmentation research**

<br>

[⚡ Quick Start](#-quick-start) · [🚀 How It Works](#-how-it-works) · [🔧 Agent Stages](#-what-the-agent-can-modify-in-trainpy) · [📊 Metrics](#-supported-metrics) · [📝 Examples](#-example-configs)

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

## 📦 Quick start

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

## 📁 Supported data formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| Standard images | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif` | Grayscale or RGB |
| NIfTI | `.nii`, `.nii.gz` | 2D slices or 3D volumes (auto-handled) |
| NumPy | `.npy` | Raw arrays |

Image-mask pairs are matched by filename stem. nnU-Net-style suffixes (`_0000`) are automatically stripped.

## 📊 Supported metrics

| Metric | Config key | Direction | Notes |
|--------|-----------|-----------|-------|
| Dice score | `dice` | Higher is better | Per-class mean (excl. background) |
| IoU / Jaccard | `iou` | Higher is better | Per-class mean (excl. background) |
| Hausdorff 95% | `hd95` | Higher is better* | *Returned as negative internally |

## 🔧 What the agent can modify in train.py

The training script is organized into clearly labeled stages:

| Stage | What it controls | Example changes |
|-------|-----------------|----------------|
| STAGE 1 | Data augmentation | Add elastic deform, cutmix, intensity jitter |
| STAGE 2 | Model architecture | UNet → Attention UNet, ResUNet, FPN, DeepLab |
| STAGE 3 | Loss function | DiceBCE → Focal, Tversky, boundary loss |
| STAGE 4 | Optimizer & scheduler | AdamW → SGD+momentum, cosine → poly decay |
| STAGE 5 | Training loop | Add AMP, EMA, gradient accumulation |

## 🎨 Design choices

- **Single file to modify.** The agent only touches `train.py`. Keeps scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly N minutes. Experiments are directly comparable regardless of architecture or hyperparameter changes.
- **Config-driven task definition.** Point `config.yaml` at any segmentation dataset — no code changes needed to switch tasks.
- **Format-agnostic data loading.** Supports natural images (PNG/JPG), medical images (NIfTI), and raw arrays (NPY) out of the box.
- **Self-contained.** No external model registries. The baseline UNet is written from scratch in train.py.

## 📝 Example configs

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

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch), which is also licensed under the MIT License.

## 🙏 Acknowledgments

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original concept of AI-driven autonomous research
- [nanochat](https://github.com/karpathy/nanochat) — the training framework that inspired the single-file approach
- [nn-UNet](https://github.com/MIC-DKFZ/nnUNet) — the state-of-the-art medical image segmentation framework that inspired many of the techniques implemented here
