"""
auto-seg: Generic data loading and evaluation for segmentation tasks.
This file is READ-ONLY. The AI agent must NOT modify this file.

Supports: PNG/JPG, NIfTI (.nii.gz), NPY image formats.
Provides: Dataset, DataLoader factory, evaluation metrics (Dice, IoU, HD95).

Usage:
    python prepare.py              # Verify dataset and print summary
    python prepare.py --synth      # Generate synthetic dataset for testing
"""

import os
import sys
import math
import random
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

_CFG = load_config()

# Export constants for train.py
TIME_BUDGET = _CFG.get("time_budget", 300)
NUM_CLASSES = _CFG["data"]["num_classes"]
IMAGE_SIZE = _CFG["data"]["image_size"]
PRIMARY_METRIC = _CFG["eval"]["primary_metric"]
DEVICE = _CFG.get("device", "cuda:0")
VAL_SPLIT = _CFG["data"]["val_split"]
DATA_SEED = _CFG["data"]["seed"]
EVAL_BATCH_SIZE = _CFG["eval"].get("eval_batch_size", 8)

# ---------------------------------------------------------------------------
# File discovery and format detection
# ---------------------------------------------------------------------------

SUPPORTED_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".nii", ".nii.gz"}

def _get_extension(filename: str) -> str:
    """Get file extension, handling .nii.gz specially."""
    if filename.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(filename).suffix.lower()

def _discover_files(directory: str) -> List[str]:
    """Discover all supported image/mask files in a directory, sorted for reproducibility."""
    files = []
    for f in os.listdir(directory):
        ext = _get_extension(f)
        if ext in SUPPORTED_IMG_EXT:
            files.append(f)
    return sorted(files)

def _match_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """Match image-mask pairs by filename stem (ignoring extensions)."""
    def stem(f):
        name = f
        if name.endswith(".nii.gz"):
            name = name[:-7]
        else:
            name = Path(name).stem
        # Strip nnUNet-style channel suffix (_0000)
        if name.endswith("_0000"):
            name = name[:-5]
        return name

    images = {stem(f): f for f in _discover_files(image_dir)}
    masks = {stem(f): f for f in _discover_files(mask_dir)}
    common = sorted(set(images.keys()) & set(masks.keys()))
    if not common:
        raise ValueError(
            f"No matching image-mask pairs found.\n"
            f"  Image dir: {image_dir} ({len(images)} files)\n"
            f"  Mask dir:  {mask_dir} ({len(masks)} files)\n"
            f"  Image stems (first 5): {list(images.keys())[:5]}\n"
            f"  Mask stems (first 5):  {list(masks.keys())[:5]}"
        )
    return [(images[s], masks[s]) for s in common]

# ---------------------------------------------------------------------------
# Image I/O (supports PNG/JPG, NIfTI, NPY)
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray:
    """Load image as float32 numpy array with shape (H, W) or (H, W, C)."""
    ext = _get_extension(path)
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    elif ext in (".nii", ".nii.gz"):
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK is required for NIfTI files: pip install SimpleITK")
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        # NIfTI: (D, H, W) or (H, W) — squeeze if 3D with D=1
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3:
            # Take middle slice for 3D volumes, or treat as multi-channel
            if arr.shape[0] <= 4:  # likely channels
                arr = arr.transpose(1, 2, 0)  # -> (H, W, C)
            else:  # likely depth
                arr = arr[arr.shape[0] // 2]  # middle slice
        return arr
    else:
        img = Image.open(path)
        return np.array(img, dtype=np.float32)

def _load_mask(path: str) -> np.ndarray:
    """Load mask as int64 numpy array with shape (H, W). Values are class indices."""
    ext = _get_extension(path)
    if ext == ".npy":
        return np.load(path).astype(np.int64)
    elif ext in (".nii", ".nii.gz"):
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK is required for NIfTI files: pip install SimpleITK")
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.int64)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        return arr
    else:
        img = Image.open(path).convert("L")  # grayscale
        arr = np.array(img, dtype=np.int64)
        # Auto-detect: if values are 0 and 255, map 255 -> 1
        unique = np.unique(arr)
        if set(unique).issubset({0, 255}):
            arr = (arr > 127).astype(np.int64)
        return arr

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegDataset(Dataset):
    """Generic segmentation dataset. Returns (image, mask) tensors.
    
    image: (C, H, W) float32, normalized to [0, 1]
    mask:  (H, W) int64, class indices
    
    No augmentation — augmentation belongs in train.py (agent can modify).
    """

    def __init__(self, split: str, transform=None):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"
        cfg = _CFG["data"]
        root = cfg["root"]
        image_dir = os.path.join(root, cfg["image_dir"])
        mask_dir = os.path.join(root, cfg["mask_dir"])
        
        all_pairs = _match_pairs(image_dir, mask_dir)
        
        # Deterministic split
        rng = random.Random(DATA_SEED)
        indices = list(range(len(all_pairs)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * VAL_SPLIT))
        
        if split == "val":
            selected = [indices[i] for i in range(n_val)]
        else:
            selected = [indices[i] for i in range(n_val, len(indices))]
        
        self.pairs = [(all_pairs[i][0], all_pairs[i][1]) for i in selected]
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = cfg["image_size"]
        self.transform = transform  # Optional: applied AFTER tensor conversion
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        image = _load_image(os.path.join(self.image_dir, img_name))
        mask = _load_mask(os.path.join(self.mask_dir, mask_name))
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # (1, H, W)
        elif image.ndim == 3:
            image = image.transpose(2, 0, 1)  # (C, H, W)
        
        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        
        # Resize
        sz = self.image_size
        image = F.interpolate(image.unsqueeze(0), size=(sz, sz), mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(sz, sz), mode="nearest").squeeze(0).squeeze(0).long()
        
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        return image, mask


def make_dataloader(split: str, batch_size: int, shuffle: bool = None, num_workers: int = 4,
                    transform=None) -> DataLoader:
    """Create a DataLoader for the given split."""
    if shuffle is None:
        shuffle = (split == "train")
    dataset = SegDataset(split=split, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
    )

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> float:
    """Compute mean Dice score across non-background classes.
    
    pred:   (N, H, W) int64 — predicted class indices
    target: (N, H, W) int64 — ground truth class indices
    """
    dices = []
    for c in range(1, num_classes):  # skip background
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        dices.append(dice.item())
    return float(np.mean(dices)) if dices else 0.0


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> float:
    """Compute mean IoU across non-background classes."""
    ious = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou.item())
    return float(np.mean(ious)) if ious else 0.0


def hd95_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Compute mean 95th percentile Hausdorff Distance across non-background classes.
    Lower is better, but we negate it so higher = better (consistent with dice/iou).
    Returns negative HD95 (so higher is still better in the keep/discard logic).
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        print("WARNING: scipy required for HD95. Falling back to Dice.")
        return dice_score(pred, target, num_classes)
    
    hds = []
    for c in range(1, num_classes):
        for i in range(pred.shape[0]):
            p = (pred[i] == c).cpu().numpy().astype(bool)
            t = (target[i] == c).cpu().numpy().astype(bool)
            if not p.any() or not t.any():
                continue
            # Surface distances
            dt_p = distance_transform_edt(~p)
            dt_t = distance_transform_edt(~t)
            surf_p = dt_t[p]  # distances from pred surface to target
            surf_t = dt_p[t]  # distances from target surface to pred
            all_dists = np.concatenate([surf_p, surf_t])
            hds.append(np.percentile(all_dists, 95))
    if not hds:
        return 0.0
    return -float(np.mean(hds))  # negative so higher = better


METRIC_FUNCTIONS = {
    "dice": dice_score,
    "iou": iou_score,
    "hd95": hd95_score,
}

# ---------------------------------------------------------------------------
# Evaluation harness (DO NOT MODIFY)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, device=None) -> Dict[str, float]:
    """Run evaluation on the validation set. Returns dict of metrics.
    
    The primary metric is specified in config.yaml.
    For dice/iou: higher is better.
    For hd95: returned as negative, so higher is still better.
    """
    if device is None:
        device = torch.device(DEVICE)
    
    model.eval()
    val_loader = make_dataloader("val", batch_size=EVAL_BATCH_SIZE, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)  # (N, C, H, W)
        preds = logits.argmax(dim=1)  # (N, H, W)
        
        all_preds.append(preds)
        all_targets.append(masks)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    results = {}
    for name, fn in METRIC_FUNCTIONS.items():
        results[name] = fn(all_preds, all_targets, NUM_CLASSES)
    
    return results


# ---------------------------------------------------------------------------
# Synthetic dataset generation (for testing the pipeline)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(root: str, n_train: int = 100, n_val: int = 20,
                                img_size: int = 256, num_classes: int = 2):
    """Generate a synthetic segmentation dataset with random shapes."""
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    
    total = n_train + n_val  # val split handled by SegDataset
    rng = np.random.RandomState(42)
    
    for i in range(total):
        # Create a grayscale image with random ellipses
        img = rng.randint(20, 80, size=(img_size, img_size), dtype=np.uint8)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # Draw 1-3 random ellipses
        n_objects = rng.randint(1, min(4, num_classes))
        for obj_idx in range(n_objects):
            cx, cy = rng.randint(40, img_size - 40, size=2)
            rx, ry = rng.randint(15, 50, size=2)
            class_id = (obj_idx % (num_classes - 1)) + 1
            
            yy, xx = np.ogrid[:img_size, :img_size]
            ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            mask[ellipse] = class_id
            # Make ellipse brighter in image
            img[ellipse] = np.clip(img[ellipse] + rng.randint(80, 150), 0, 255)
        
        # Add noise
        noise = rng.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        Image.fromarray(img, mode="L").save(os.path.join(root, "images", f"sample_{i:04d}.png"))
        Image.fromarray(mask, mode="L").save(os.path.join(root, "masks", f"sample_{i:04d}.png"))
    
    print(f"Synthetic dataset generated at {root}: {total} samples, {num_classes} classes")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dataset or generate synthetic data")
    parser.add_argument("--synth", action="store_true", help="Generate synthetic dataset for testing")
    parser.add_argument("--synth-dir", type=str, default="./synth_data", help="Output dir for synthetic data")
    parser.add_argument("--synth-n", type=int, default=120, help="Total number of synthetic samples")
    args = parser.parse_args()
    
    if args.synth:
        generate_synthetic_dataset(args.synth_dir, n_train=args.synth_n)
        print(f"\nTo use: edit config.yaml and set data.root to '{os.path.abspath(args.synth_dir)}'")
        sys.exit(0)
    
    # Verify existing dataset
    cfg = _CFG["data"]
    print(f"Config: {CONFIG_PATH}")
    print(f"  Data root:       {cfg['root']}")
    print(f"  Image dir:       {cfg['image_dir']}")
    print(f"  Mask dir:        {cfg['mask_dir']}")
    print(f"  Num classes:     {cfg['num_classes']}")
    print(f"  Image size:      {cfg['image_size']}")
    print(f"  Val split:       {cfg['val_split']}")
    print(f"  Primary metric:  {_CFG['eval']['primary_metric']}")
    print(f"  Time budget:     {TIME_BUDGET}s")
    print(f"  Device:          {DEVICE}")
    
    try:
        image_dir = os.path.join(cfg['root'], cfg['image_dir'])
        mask_dir = os.path.join(cfg['root'], cfg['mask_dir'])
        pairs = _match_pairs(image_dir, mask_dir)
        print(f"\n  Found {len(pairs)} image-mask pairs")
        
        # Test loading one pair
        img = _load_image(os.path.join(image_dir, pairs[0][0]))
        msk = _load_mask(os.path.join(mask_dir, pairs[0][1]))
        print(f"  Sample image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.1f}, {img.max():.1f}]")
        print(f"  Sample mask shape:  {msk.shape}, dtype: {msk.dtype}, unique values: {np.unique(msk)}")
        
        # Test dataset
        ds_train = SegDataset("train")
        ds_val = SegDataset("val")
        print(f"\n  Train samples: {len(ds_train)}")
        print(f"  Val samples:   {len(ds_val)}")
        
        img_t, msk_t = ds_train[0]
        print(f"  Tensor image: {img_t.shape}, range: [{img_t.min():.3f}, {img_t.max():.3f}]")
        print(f"  Tensor mask:  {msk_t.shape}, unique: {msk_t.unique().tolist()}")
        
        print("\n✅ Dataset verified successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Run 'python prepare.py --synth' to generate a test dataset first.")
        sys.exit(1)
