<div align="center">

# 🔬 Auto-Segmentation

### *让 AI Agent 在你睡觉的时候做分割研究。*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

[![Claude Code](https://img.shields.io/badge/Claude%20Code-Agent-blueviolet)](https://claude.ai/code)
[![Codex](https://img.shields.io/badge/Codex-Agent-black)](https://openai.com/codex)
[![Cursor](https://img.shields.io/badge/Cursor-Agent-blue)](https://cursor.sh)

<br>

<table>
<tr><td align="left">

🧠 &nbsp;厌倦了为每个新的分割任务手动调架构、损失函数、超参数？<br>
🌙 &nbsp;想让 AI 在你睡觉时跑 100 次实验，醒来就得到更好的模型？<br>
🔄 &nbsp;想要一个能自主尝试想法、保留有效方案、丢弃无效方案的 AI？

</td></tr>
</table>

### ✨ Auto-Segmentation 正是实现这一目标。

<br>

灵感来自 [karpathy/autoresearch](https://github.com/karpathy/autoresearch)（LLM 预训练自主研究项目）—— **现已泛化至任意图像分割任务**。

指向你的数据集 · 选择评估指标 · 让智能体整夜自主迭代

医学影像 · 遥感图像 · 自然图像 · **任意分割任务**

**一个配置文件 + 一个 AI 智能体 = 自主分割研究**

<br>

[⚡ 快速开始](#-快速开始) · [🚀 工作原理](#-工作原理) · [🔧 Agent 可修改的阶段](#-agent-可修改-trainpy-的内容) · [📊 指标](#-支持的指标) · [📝 示例配置](#-示例配置)

[**中文**](README_ZH.md) · [**English**](../README.md)

</div>

---

## 🆚 与 nn-UNet 的对比

| 特性 | Auto-Segmentation | nn-UNet |
|---------|------------------|---------|
| **设计目标** | 自主研究框架，用于实验分割架构和技巧 | 生产级、自动配置的分割系统 |
| **自动化方式** | AI 智能体自主修改代码、训练并迭代改进 | 自动化预处理和配置，但需手动训练模型 |
| **灵活性** | 高度灵活 - 智能体可修改架构、损失函数、数据增强、优化器 | 专注于 U-Net 变体，训练流程固定 |
| **研究重点** | 探索新颖架构、损失函数、训练策略 | 针对已验证的最先进医学图像分割进行优化 |
| **数据集支持** | 任意分割数据集（PNG、NIfTI、NPY），配置简单 | 主要用于医学影像，需特定预处理流程 |
| **时间预算** | 固定训练时间（默认 5 分钟），快速实验 | 完整训练直至收敛（数小时到数天） |
| **评估方式** | 快速迭代，实验结果即时反馈 | 全面评估，包含交叉验证 |
| **适用场景** | 研究原型设计、架构搜索、教育学习 | 生产环境医学图像分割、竞赛项目 |
| **设置复杂度** | 简单 - 配置文件指向数据集即可 | 复杂 - 需要特定的文件夹结构和预处理 |
| **可重现性** | Git 完整记录实验历史 | 基于固定配置，高度可重现 |

**Auto-Segmentation 的核心优势：**
- ✅ **快速实验** - 5 分钟迭代 vs 数小时训练
- ✅ **自主研究** - AI 智能体自动探索设计空间
- ✅ **简单配置** - 无需复杂预处理流程
- ✅ **教育价值** - 通过实验不同方法学习分割技术
- ✅ **高度灵活** - 轻松尝试新架构、损失函数等
- ✅ **格式无关** - 原生支持多种图像格式

**何时选择 nn-UNet：**
- 🏥 生产环境医学图像分割
- 🏆 需要最先进结果的竞赛项目
- 📊 需要基准方法的学术论文
- 🔬 nn-UNet 已成为标准基线的任务

## 🚀 工作原理

整个项目只有四个关键文件：

| 文件 | 作用 | 谁修改 |
|------|------|--------|
| `prepare.py` | 数据加载、评估指标、数据集工具 | ❌ 只读 |
| `train.py` | 模型、损失函数、优化器、数据增强、训练循环 | 🤖 AI 智能体 |
| `program.md` | 智能体指令和研究策略 | 👨‍💻 人类 |
| `config.yaml` | 数据集路径、评估指标、时间预算、设备 | 👨‍💻 人类 |

智能体可以修改 `train.py` 的任何内容：架构、损失函数、数据增强、优化器、超参数。每次训练结束后，主指标（Dice、IoU 或 HD95）决定是否保留本次修改。

```
┌──────────────────────────────────────────────────────┐
│                   人工设置                             │
│  config.yaml: 数据路径、指标、时间预算                   │
│  program.md:  研究优先级和约束条件                      │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│              智能体循环（自主运行）                      │
│                                                       │
│  1. 修改 train.py（架构、损失函数、增强...）             │
│  2. git commit                                        │
│  3. python train.py > run.log 2>&1                    │
│  4. 检查主指标                                         │
│  5. 改善了？→ 保留提交                                 │
│     未改善？→ git reset 回退                           │
│  6. 记录到 results.tsv                                │
│  7. 返回第 1 步                                        │
└──────────────────────────────────────────────────────┘
```

## ⚡ 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将数据组织为以下结构：
```
/path/to/dataset/
├── images/          # 输入图片（PNG、JPG、NIfTI 或 NPY）
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
└── masks/           # 分割标注（相同格式、相同文件名）
    ├── sample_001.png
    ├── sample_002.png
    └── ...
```

或生成合成测试数据集：
```bash
python prepare.py --synth --synth-dir ./synth_data
```

### 3. 配置

编辑 `config.yaml` 指向你的数据集：
```yaml
data:
  root: "/path/to/dataset"
  image_dir: "images"
  mask_dir: "masks"
  num_classes: 2          # 含背景
  image_size: 256
  val_split: 0.2

eval:
  primary_metric: "dice"  # dice | iou | hd95

time_budget: 300          # 秒（默认 5 分钟）
device: "cuda:0"
```

验证配置：
```bash
python prepare.py
```

### 4. 运行一次训练实验

```bash
python train.py
```

训练固定运行你配置的时间预算，然后评估并输出：
```
---
dice:             0.847300
iou:              0.734500
training_seconds: 300.1
peak_vram_mb:     8234.2
num_steps:        1523
num_params:       7,832,134
```

### 5. 启动智能体

将你的 AI 编程助手（Claude Code、Codex、Cursor 等）指向此仓库，然后输入：

```
阅读 program.md，启动新实验！先进行环境设置。
```

智能体将创建分支、建立基线，然后开始自主迭代实验。

## 📁 支持的数据格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 标准图片 | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif` | 灰度或 RGB |
| NIfTI | `.nii`, `.nii.gz` | 2D 切片或 3D 体积（自动处理） |
| NumPy | `.npy` | 原始数组 |

图片-标注对通过文件名匹配。nnU-Net 风格后缀（`_0000`）会自动去除。

## 📊 支持的指标

| 指标 | 配置键 | 方向 | 说明 |
|------|--------|------|------|
| Dice 系数 | `dice` | 越高越好 | 按类别平均（不含背景） |
| IoU / Jaccard | `iou` | 越高越好 | 按类别平均（不含背景） |
| Hausdorff 95% | `hd95` | 越高越好* | *内部返回负值 |

## 🔧 智能体可修改 train.py 的内容

训练脚本被划分为清晰标记的五个阶段：

| 阶段 | 控制内容 | 修改示例 |
|------|----------|----------|
| STAGE 1 | 数据增强 | 添加弹性变形、cutmix、强度抖动 |
| STAGE 2 | 模型架构 | UNet → Attention UNet、ResUNet、FPN、DeepLab |
| STAGE 3 | 损失函数 | DiceBCE → Focal、Tversky、边界损失 |
| STAGE 4 | 优化器和调度器 | AdamW → SGD+momentum、cosine → poly decay |
| STAGE 5 | 训练循环 | 添加 AMP、EMA、梯度累积 |

## 📝 示例配置

<details>
<summary>医学影像分割（二分类）</summary>

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
<summary>多类别器官分割</summary>

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
time_budget: 600    # 大图用 10 分钟
device: "cuda:0"
```
</details>

<details>
<summary>遥感图像分割</summary>

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

## 📄 许可证

MIT 许可证 - 详见 [LICENSE](../LICENSE) 文件。

本项目基于 [karpathy/autoresearch](https://github.com/karpathy/autoresearch)，同样采用 MIT 许可证。

## 🙏 致谢

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — AI 驱动自主研究的原创概念
- [nanochat](https://github.com/karpathy/nanochat) — 启发了单文件设计思想的训练框架
- [nn-UNet](https://github.com/MIC-DKFZ/nnUNet) — 最先进的医学图像分割框架，本项目实现了其中的许多技术
