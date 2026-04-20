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

厌倦了为每个新的分割任务手动调架构、损失函数、超参数？<br>
想让 AI 在你睡觉时跑 100 次实验，醒来就得到更好的模型？<br>
想要一个能自主尝试想法、保留有效方案、丢弃无效方案的 AI？

</td></tr>
</table>

### 🔬 Auto-Segmentation 正为实现这一目标而诞生的。

<br>

灵感来自 [karpathy/autoresearch](https://github.com/karpathy/autoresearch)（LLM 预训练自主研究项目）—— **现已泛化至任意图像分割任务**。

指向你的数据集 · 选择评估指标 · 让智能体整夜自主迭代

医学影像 · 遥感图像 · 自然图像 · **任意分割任务**

**一个配置文件 + 一个 AI 智能体 = 自主分割研究**

<br>

[快速开始](#-快速开始) · [工作原理](#-工作原理) · [可修改阶段](#-智能体可修改-trainpy-的内容) · [指标](#-支持的指标) · [示例配置](#-示例配置)

[**中文**](README_ZH.md) · [**English**](../README.md)

</div>

---

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

## 快速开始

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

## 支持的数据格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 标准图片 | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif` | 灰度或 RGB |
| NIfTI | `.nii`, `.nii.gz` | 2D 切片或 3D 体积（自动处理） |
| NumPy | `.npy` | 原始数组 |

图片-标注对通过文件名匹配。nnU-Net 风格后缀（`_0000`）会自动去除。

## 支持的指标

| 指标 | 配置键 | 方向 | 说明 |
|------|--------|------|------|
| Dice 系数 | `dice` | 越高越好 | 按类别平均（不含背景） |
| IoU / Jaccard | `iou` | 越高越好 | 按类别平均（不含背景） |
| Hausdorff 95% | `hd95` | 越高越好* | *内部返回负值 |

## 智能体可修改 train.py 的内容

训练脚本被划分为清晰标记的五个阶段：

| 阶段 | 控制内容 | 修改示例 |
|------|----------|----------|
| STAGE 1 | 数据增强 | 添加弹性变形、cutmix、强度抖动 |
| STAGE 2 | 模型架构 | UNet → Attention UNet、ResUNet、FPN、DeepLab |
| STAGE 3 | 损失函数 | DiceBCE → Focal、Tversky、边界损失 |
| STAGE 4 | 优化器和调度器 | AdamW → SGD+momentum、cosine → poly decay |
| STAGE 5 | 训练循环 | 添加 AMP、EMA、梯度累积 |

## 示例配置

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

## ⚠️ 局限性与扩展方法

这是一个**研究原型系统**，而非生产级系统。然而，它被设计为**易于扩展**。以下是主要限制及如何克服它们：

### 架构与训练
**限制：** 仅基础 UNet — 无注意力机制、残差连接或现代架构改进
**🔧 解决方法：** 修改 `train.py` 中的 `STAGE 2`。将 `UNet` 类替换为：
```python
# 在跳跃连接中添加注意力门
# 用 ResUNet 块替换（添加残差连接）
# 实现 FPN 风格的解码器
# 尝试带空洞空间金字塔池化的 DeepLabV3+
```

**限制：** 简单数据增强 — 仅随机翻转。无弹性形变、颜色抖动、CutMix
**🔧 解决方法：** 修改 `train.py` 中的 `STAGE 1`。在 `augment_batch()` 中添加：
```python
# 使用 scipy.ndimage 添加弹性形变
# 为 RGB 图像实现颜色抖动
# 添加 CutMix 或复制-粘贴增强
# 尝试随机旋转、缩放和平移
```

**限制：** 基础损失函数 — 仅 Dice + 交叉熵损失。无 Focal 损失、Tversky 损失、边界感知损失
**🔧 解决方法：** 修改 `train.py` 中的 `STAGE 3`。将 `DiceBCELoss` 替换为：
```python
# 添加 Focal 损失处理类别不平衡
# 实现可调 α/β 的 Tversky 损失
# 添加用于边缘细化的边界加权损失
# 组合多个损失并学习权重
```

**限制：** 有限评估指标 — 仅 Dice、IoU 和 HD95。无逐类别分解或混淆矩阵
**🔧 解决方法：** 在 `train.py` 中扩展评估部分：
```python
# 添加逐类别 Dice/IoU 计算
# 实现混淆矩阵分析
# 添加边界特定指标（边界 Dice、Hausdorff 表面）
# 跟踪每个类别的精确率/召回率
```

### 训练基础设施
**限制：** 无模型检查点 — 模型不保存，每次实验从头开始
**🔧 解决方法：** 在 `train.py` 的 `STAGE 5` 中添加：
```python
# 保存最佳模型：torch.save(model.state_dict(), "best_model.pth")
# 实现训练期间的定期检查点
# 添加模型加载以继续训练或推理
# 导出为 ONNX 用于部署
```

**限制：** 固定时间预算 — 训练在 N 分钟后停止，不考虑收敛
**🔧 解决方法：** 修改 `STAGE 5` 中的训练循环：
```python
# 基于验证指标添加早停
# 实现带重启的学习率调度
# 在训练期间添加验证（不仅仅在结束时）
# 使用指标平台检测
```

**限制：** 仅单 GPU — 无多 GPU 或分布式训练支持
**🔧 解决方法：** 用 PyTorch 分布式训练包装模型：
```python
# 添加：torch.nn.DataParallel(model) 进行简单多 GPU
# 使用 torch.distributed 进行真正的分布式训练
# 实现梯度累积以获得更大的有效批量大小
# 添加混合精度（AMP）训练以提高内存效率
```

**限制：** 最小化日志 — 仅终端输出和 TSV 文件。无 TensorBoard、W&B 或详细跟踪
**🔧 解决方法：** 在 `STAGE 5` 中添加日志：
```python
# 集成 TensorBoard：from torch.utils.tensorboard import SummaryWriter
# 添加 Weights & Biases 日志：import wandb; wandb.init()
# 保存训练曲线和指标历史
# 在训练期间记录样本预测
```

### 数据处理
**限制：** 简单预处理 — 基本的调整大小和归一化。无强度窗宽化或高级预处理
**🔧 解决方法：** 在 `prepare.py` 中扩展 `SegDataset`：
```python
# 为医学图像添加强度窗宽化
# 实现直方图均衡化或 CLAHE
# 添加按数据集的 z-score 归一化
# 实现特定数据集的预处理钩子
```

**限制：** 基础数据分割 — 简单随机分割。无交叉验证或分层分割
**🔧 解决方法：** 修改数据集分割逻辑：
```python
# 添加 k 折交叉验证支持
# 按类别分布实现分层分割
# 为多中心数据添加领域感知分割
# 在不同分割上实现集成训练
```

**限制：** 仅 2D 训练 — 无原生 3D 训练支持。NIfTI 体数据被切片为 2D
**🔧 解决方法：** 修改数据加载和模型架构：
```python
# 加载完整 3D 体数据而非切片
# 实现带 3D 卷积的 3D UNet
# 为大体数据添加滑动窗口推理
# 使用梯度检查点进行内存高效的 3D 训练
```

### 研究限制
**限制：** 无超参数搜索 — 实验由智能体直觉驱动，非系统化 HPO
**🔧 解决方法：** 添加系统化搜索策略：
```python
# 对关键超参数实现网格搜索
# 添加贝叶斯优化以进行高效搜索
# 使用 Optuna 或 Hyperopt 进行自动化 HPO
# 跟踪超参数重要性分析
```

**限制：** 无集成方法 — 无模型集成、测试时增强或不确定性估计
**🔧 解决方法：** 在 `STAGE 5` 中添加集成：
```python
# 保存多个检查点并集成预测
# 实现测试时增强（TTA）
# 添加蒙特卡洛 Dropout 以进行不确定性估计
# 实现跨不同随机种子的模型平均
```

**限制：** 无可解释性 — 无注意力图、GradCAM 或预测可视化
**🔧 解决方法：** 添加可视化工具：
```python
# 实现 GradCAM 以进行注意力可视化
# 添加预测在原始图像上的叠加
# 为基于 Transformer 的模型创建注意力图
# 生成置信度图和不确定性可视化
```

### 生产就绪度
**限制：** 无推理管道 — 无模型导出、ONNX 转换或部署工具
**🔧 解决方法：** 添加部署能力：
```python
# 创建带模型加载的独立推理脚本
# 添加 ONNX 导出：torch.onnx.export(model, ...)
# 实现用于生产的批量推理
# 添加用于推理的 REST API 或命令行接口
```

**限制：** 基础错误处理 — 最小化的验证和错误恢复
**🔧 解决方法：** 添加健壮的错误处理：
```python
# 添加输入验证和健全性检查
# 实现错误的优雅降级
# 添加全面的日志记录和调试工具
# 创建验证测试套件
```

### 灵活性优势

🎯 **关键点：** 与"黑盒"框架不同，Auto-Segmentation 让您完全掌控：
- **单一修改文件** — 所有更改都在 `train.py` 中进行
- **清晰的阶段** — 每个组件都有标记且易于查找
- **Git 跟踪实验** — 每个修改都被记录且可撤销
- **无黑魔法** — 您可以查看和修改所有内容

**您永远不会被困住。** 如果遇到限制：
1. 在 `train.py` 中找到相关阶段
2. 修改代码以添加所需功能
3. 让 AI 智能体迭代您的更改
4. 比较结果并保留有效方案

### 本框架适用场景
✅ 分割想法的快速原型验证
✅ 分割技术的教育探索
✅ 架构搜索和实验
✅ 学习不同组件如何影响性能
✅ 构建自定义分割解决方案

### 建议使用成熟框架的场景
❌ 生产环境医学图像分割
❌ 需要标准化基线的已发表研究
❌ 需要最先进结果的竞赛项目
❌ 可靠性和可重现性至关重要的任务

## 许可证

MIT 许可证 - 详见 [LICENSE](../LICENSE) 文件。

本项目基于 [karpathy/autoresearch](https://github.com/karpathy/autoresearch)，同样采用 MIT 许可证。

## 致谢

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — AI 驱动自主研究的原创概念
- [nanochat](https://github.com/karpathy/nanochat) — 启发了单文件设计思想的训练框架
- [nn-UNet](https://github.com/MIC-DKFZ/nnUNet) — 最先进的医学图像分割框架，本项目实现了其中的许多技术
