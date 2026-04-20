"""
auto-seg: Single-file segmentation training script.
This is the ONLY file the AI agent modifies.

It contains the full segmentation pipeline:
  - Data augmentation
  - Model architecture (UNet baseline)
  - Loss function
  - Optimizer & scheduler
  - Training loop

The agent experiments by modifying any of these components.
The only constraint: the code must run without crashing and finish
within the time budget defined in config.yaml.

Usage: python train.py
"""

import os
import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, NUM_CLASSES, IMAGE_SIZE, PRIMARY_METRIC, DEVICE,
    make_dataloader, evaluate, load_config,
)

# ============================================================================
# STAGE 1: DATA AUGMENTATION (agent can modify)
# ============================================================================
# Augmentations are applied on GPU tensors: (C, H, W) image, (H, W) mask.
# Must return (image, mask) with same shapes.

def augment_batch(images, masks):
    """Apply augmentations to a batch of (images, masks) on GPU.
    images: (N, C, H, W) float, masks: (N, H, W) long
    """
    N = images.shape[0]
    for i in range(N):
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            images[i] = images[i].flip(-1)
            masks[i] = masks[i].flip(-1)
        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            images[i] = images[i].flip(-2)
            masks[i] = masks[i].flip(-2)
    return images, masks

# ============================================================================
# STAGE 2: MODEL ARCHITECTURE (agent can modify)
# ============================================================================
# Default: UNet with configurable depth and width.
# The agent is free to replace this with any architecture.

class ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU x2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Standard UNet with configurable depth and base channels.
    
    Args:
        in_channels:  Number of input channels (1 for grayscale, 3 for RGB)
        num_classes:  Number of output classes (including background)
        base_ch:      Base channel count (doubled at each encoder level)
        depth:        Number of encoder/decoder levels
    """
    def __init__(self, in_channels=1, num_classes=2, base_ch=64, depth=4):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        encoder_channels = []
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            encoder_channels.append(out_ch)
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck
        bottleneck_ch = base_ch * (2 ** depth)
        self.bottleneck = ConvBlock(ch, bottleneck_ch)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = encoder_channels[i]
            self.upconvs.append(nn.ConvTranspose2d(ch, skip_ch, 2, stride=2))
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch))
            ch = skip_ch

        # Output
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        # Encoder path
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder path
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Handle size mismatch from non-power-of-2 inputs
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)


def build_model(in_channels):
    """Build the segmentation model. Agent can modify this function."""
    return UNet(
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
        base_ch=BASE_CHANNELS,
        depth=DEPTH,
    )

# ============================================================================
# STAGE 3: LOSS FUNCTION (agent can modify)
# ============================================================================

class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss.
    Works for both binary and multi-class segmentation.
    """
    def __init__(self, dice_weight=1.0, ce_weight=1.0, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:  (N, C, H, W) raw model output
        targets: (N, H, W) class indices
        """
        # Cross-entropy
        ce_loss = F.cross_entropy(logits, targets)

        # Dice loss (per-class, excluding background)
        probs = F.softmax(logits, dim=1)
        dice_loss = 0.0
        num_fg_classes = logits.shape[1] - 1  # exclude background
        for c in range(1, logits.shape[1]):
            p = probs[:, c]
            t = (targets == c).float()
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice_loss += 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        if num_fg_classes > 0:
            dice_loss = dice_loss / num_fg_classes

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def build_loss():
    """Build the loss function. Agent can modify this function."""
    return DiceBCELoss(dice_weight=DICE_WEIGHT, ce_weight=CE_WEIGHT)

# ============================================================================
# STAGE 4: OPTIMIZER & SCHEDULER (agent can modify)
# ============================================================================

def build_optimizer(model):
    """Build optimizer. Agent can modify this function."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, total_steps):
    """Build LR scheduler. Agent can modify this function."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LEARNING_RATE * 0.01,
    )

# ============================================================================
# HYPERPARAMETERS (agent can modify these directly)
# ============================================================================

# Model
DEPTH = 4                  # number of encoder/decoder levels
BASE_CHANNELS = 64         # base channel count (doubled per level)

# Training
BATCH_SIZE = 8             # training batch size
LEARNING_RATE = 1e-3       # initial learning rate
WEIGHT_DECAY = 1e-4        # AdamW weight decay

# Loss
DICE_WEIGHT = 1.0          # weight for Dice loss component
CE_WEIGHT = 1.0            # weight for cross-entropy loss component

# ============================================================================
# STAGE 5: TRAINING LOOP (agent can modify)
# ============================================================================

def main():
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) if torch.cuda.is_available() else None
    device = torch.device(DEVICE)

    # --- Data ---
    train_loader = make_dataloader("train", batch_size=BATCH_SIZE)

    # Detect input channels from first sample
    sample_img, _ = next(iter(train_loader))
    in_channels = sample_img.shape[1]
    print(f"Input channels: {in_channels}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Num classes: {NUM_CLASSES}")
    print(f"Primary metric: {PRIMARY_METRIC}")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Device: {device}")

    # --- Model ---
    model = build_model(in_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # --- Loss, optimizer, scheduler ---
    criterion = build_loss()
    optimizer = build_optimizer(model)

    # Estimate total steps (rough: based on time budget and batch time)
    est_steps = max(100, TIME_BUDGET * 2)  # rough estimate
    scheduler = build_scheduler(optimizer, est_steps)

    # --- Training ---
    print(f"\nTraining started (budget: {TIME_BUDGET}s)")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 60)

    t_start_training = time.time()
    total_training_time = 0.0
    step = 0
    epoch = 0
    smooth_loss = 0.0
    data_iter = iter(train_loader)

    # GC optimization (from original autoresearch)
    gc.collect()
    gc.freeze()
    gc.disable()

    while True:
        epoch += 1
        model.train()

        for images, masks in train_loader:
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Augmentation (on GPU)
            images, masks = augment_batch(images, masks)

            # Forward
            logits = model(images)
            loss = criterion(logits, masks)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.time()
            dt = t1 - t0

            if step > 5:  # skip first few steps (warmup)
                total_training_time += dt

            # Logging
            loss_val = loss.item()

            # Fast fail
            if math.isnan(loss_val) or loss_val > 100:
                print("FAIL: loss exploded")
                exit(1)

            ema = 0.9
            smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
            debiased = smooth_loss / (1 - ema ** (step + 1))
            pct = 100 * min(total_training_time / TIME_BUDGET, 1.0)
            remaining = max(0, TIME_BUDGET - total_training_time)
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} "
                f"| lr: {lr:.2e} | dt: {dt*1000:.0f}ms "
                f"| epoch: {epoch} | remaining: {remaining:.0f}s    ",
                end="", flush=True,
            )

            step += 1

            # Time's up
            if step > 5 and total_training_time >= TIME_BUDGET:
                break

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r
    print("-" * 60)

    # --- Evaluation ---
    print("Evaluating on validation set...")
    metrics = evaluate(model, device)

    primary_val = metrics[PRIMARY_METRIC]

    # --- Summary ---
    t_end = time.time()
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / 1024 / 1024) if device.type == "cuda" else 0

    print("---")
    print(f"{PRIMARY_METRIC}:        {primary_val:.6f}")
    for k, v in metrics.items():
        if k != PRIMARY_METRIC:
            print(f"{k}:        {v:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {num_params:,}")
    print(f"batch_size:       {BATCH_SIZE}")
    print(f"depth:            {DEPTH}")
    print(f"base_channels:    {BASE_CHANNELS}")


if __name__ == "__main__":
    main()
