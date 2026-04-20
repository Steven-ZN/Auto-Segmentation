# Auto-Segmentation: Agent Instructions for Segmentation Research

This is an experiment to have an AI agent do autonomous segmentation research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr20`). The branch `Auto-Segmentation/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b Auto-Segmentation/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `config.yaml` — task configuration (dataset path, metric, time budget). Do not modify.
   - `prepare.py` — data loading, evaluation metrics, dataset utilities. Do not modify.
   - `train.py` — the file you modify. Model architecture, loss function, optimizer, augmentation, training loop.
4. **Verify data**: Check that `config.yaml` points to a valid dataset. If not, tell the human to run `python prepare.py --synth` to generate a test dataset.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget** (default 5 minutes, configured in `config.yaml`). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - **STAGE 1 - Data Augmentation**: Add/remove/modify augmentation strategies (random crops, elastic transforms, color jitter, mixup, cutmix, etc.)
  - **STAGE 2 - Model Architecture**: Change the architecture entirely (UNet variants, attention mechanisms, residual blocks, FPN, DeepLab, etc.)
  - **STAGE 3 - Loss Function**: Try different losses (focal loss, boundary loss, Tversky loss, combined losses, etc.)
  - **STAGE 4 - Optimizer & Scheduler**: Change optimizer (SGD, AdamW, LAMB, etc.), learning rate schedules, warmup strategies.
  - **STAGE 5 - Training Loop**: Modify the training logic (gradient accumulation, mixed precision, EMA, deep supervision, etc.)
  - **Hyperparameters**: Batch size, learning rate, model depth/width, weight decay, loss weights — all fair game.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and metrics.
- Modify `config.yaml`. It defines the task and is set by the human.
- Install new packages beyond what's in `requirements.txt`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth.

**The goal is simple: get the highest primary metric** (specified in `config.yaml` — Dice, IoU, or HD95). Since the time budget is fixed, you don't need to worry about training time — it's always the same. Everything is fair game: change the architecture, the loss, the optimizer, the augmentation, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful metric gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Research Strategy for Segmentation

When deciding what to try, consider this priority order:

1. **Loss function** — Often gives the biggest gains with minimal code changes. Try:
   - Dice + CE with different weights
   - Focal loss for class imbalance
   - Boundary-aware losses
   - Tversky loss with tuned alpha/beta

2. **Data augmentation** — Second highest impact. Try:
   - Elastic deformation (great for medical images)
   - Random affine transforms
   - Intensity augmentation (brightness, contrast, gamma)
   - CutMix / Copy-Paste augmentation

3. **Architecture modifications** — Higher risk, higher reward. Try:
   - Attention gates in skip connections
   - Residual blocks instead of plain conv blocks
   - Deep supervision (auxiliary losses at decoder stages)
   - Feature Pyramid Network (FPN) style decoder
   - Squeeze-and-excitation blocks

4. **Training tricks** — Often small but consistent gains. Try:
   - Mixed precision training (AMP)
   - Exponential Moving Average (EMA)
   - Gradient accumulation for larger effective batch size
   - Learning rate warmup
   - Stochastic weight averaging (SWA)

5. **Optimizer tuning** — Usually smaller impact but worth trying:
   - SGD with momentum + polynomial LR decay
   - AdamW with cosine annealing
   - Different learning rates for encoder vs decoder

## Output format

Once the script finishes it prints a summary like this:

```
---
dice:             0.847300
iou:              0.734500
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     8234.2
num_steps:        1523
num_params:       7,832,134
batch_size:       8
depth:            4
base_channels:    64
```

You can extract the key metric from the log file:

```
grep "^dice:" run.log        # or whatever the primary metric is
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	metric_value	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. primary metric value achieved (e.g. 0.847300) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 8.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	metric_value	memory_gb	status	description
a1b2c3d	0.847300	8.0	keep	baseline UNet
b2c3d4e	0.862100	8.2	keep	add attention gates
c3d4e5f	0.845000	8.0	discard	focal loss (no improvement)
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^dice:\|^iou:\|^peak_vram_mb:" run.log` (adjust metric name to match your config)
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If the primary metric improved (higher), you "advance" the branch, keeping the git commit
9. If the metric is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~5 minutes total (+ some time for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes, read the latest segmentation papers for inspiration. The loop runs until the human interrupts you, period.
