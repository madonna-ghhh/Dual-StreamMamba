# Dual-Stream Mamba EEG — Per-Patient Seizure Prediction (v2, Anti-Overfit Edition)

A per-patient deep learning model that predicts preictal (pre-seizure) vs. normal EEG segments using a dual-stream Mamba (state-space model) architecture. Version 2 focuses on curbing overfitting on small, largely synthetic per-patient datasets through stochastic weight averaging, a bottleneck classification head, and stronger regularization throughout.

**Dataset:** [Seizure Prediction Dataset] (CHB-MIT processed + augmented)

---

## Overview

Each patient in the CHB-MIT dataset gets their own model — trained, validated, and tested entirely on their own EEG recordings, so **no patient ever sees another patient's data**. The model treats EEG as a dual-stream problem:

- A **temporal stream** scans patches of time for each channel independently (does this channel's signal look preictal over time?)
- A **channel stream** scans across channels for each time patch (does the spatial pattern across electrodes look preictal at this moment?)

These two views are combined with a gated fusion mechanism, and a compressed bottleneck head produces the final Normal / Preictal classification.

## What changed from v1

| Problem identified | Fix applied |
|---|---|
| Early stopping on noisy val loss | **SWA** (Stochastic Weight Averaging) — averages last N checkpoints instead of picking a noisy best |
| Single-class patients crash AUC | **Pre-flight check** skips patients where val or test has only one class |
| ~100% synthetic train data | **Stronger regularization** — higher dropout rates throughout |
| Val loss jumps ±0.05 per epoch | **Smoothed early stopping** — stops on 5-epoch rolling AUC average |
| Head directly on pooled features | **Bottleneck head** — D→D/2→2 forces compression before classification |

## Architecture

```
Input (batch, 2560, C)
  → PatchEmbed Conv2d        → (batch, 32 patches, C, d_model)
  → SpectralBranch (dw conv) → local frequency features
  → N × DualStreamBlock:
       ├─ Temporal Mamba: scans 32 patches per channel
       ├─ Channel Mamba:  scans C channels per patch
       └─ GatedFusion: sigmoid gate + stochastic depth
  → AdaptiveAvgPool          → (batch, d_model)
  → BottleneckHead: Linear(D→D/2) → GELU → Dropout(0.30) → Linear(D/2→2)
```

**Design properties (v2):**
- No attention matrices — gated fusion instead, no memorizable attention maps
- Two *shared* SSMs (temporal + channel) reused across all blocks, regardless of `N_BLOCKS`
- Spectral branch is a lightweight depthwise convolution (<200 params)
- Stochastic depth (DropPath) at 0.15
- Dropout at every stage: embed=0.15, ssm=0.10, gate=0.15, head=0.30
- Label smoothing at 0.15
- SWA in the training loop instead of noisy best-checkpoint selection

## Why these choices?

**SWA instead of best-checkpoint early stopping.** Some patients' validation sets have fewer than 200 samples, so val loss can swing ±0.05 between epochs from mini-batch randomness alone. SWA (Izmailov et al., 2018) averages weights from the second half of training; the averaged vector sits near the center of a flat loss basin, which generalizes better than any single noisy "best" checkpoint.

**Bottleneck head.** A plain `Linear(32→2)` head has enough free parameters to memorize training patterns, especially with a training set that is almost entirely synthetic augmentation. Forcing features through a 16-dimensional bottleneck before classification changes what the head is structurally capable of representing, not just how many parameters it has.

**Higher dropout throughout.** Training data for some patients is close to 100:1 synthetic-to-real, so the model has little natural incentive to generalize. Every dropout rate was raised relative to v1 (see table below).

| Location | v1 | v2 |
|---|---|---|
| Embed dropout | 0.10 | 0.15 |
| SSM dropout | 0.05 | 0.10 |
| Gate dropout | 0.10 | 0.15 |
| Head dropout | — | 0.30 |

**Smoothed early stopping on AUC, not val loss.** Val loss rewards predicting near 0.5 for everything. AUC measures the actual clinical question: does the model rank preictal samples above normal ones? A 5-epoch rolling average with patience=9 requires a sustained plateau (not a single noisy dip) before stopping.

**Pre-flight patient skipping.** Patients with no real training data (`train_aug` missing) or a single-class val/test split (e.g. chb06, chb08, chb12, chb13, chb24) are skipped automatically with a printed reason, rather than silently producing meaningless AUC or crashing.

## Configuration

| Parameter | Value |
|---|---|
| `D_MODEL` | 64 |
| `N_BLOCKS` | 3 |
| `D_STATE` | 16 |
| `EXPAND` | 2 |
| `D_CONV` | 2 |
| `DW_KERNEL` | 5 |
| `BOTTLENECK_DIM` | 16 |
| `PATCH_SIZE` | 80 (2560 / 80 = 32 patches per window) |
| `BATCH_SIZE` | 16 |
| `EPOCHS` | 30 |
| `LR` / `SWA_LR` | 1e-3 / 2e-4 |
| `WEIGHT_DECAY` | 2e-3 |
| `SWA_START_FRAC` | 0.50 |
| `SMOOTH_WINDOW` / `SMOOTH_PATIENCE` | 5 / 9 |

## Training pipeline (per patient)

1. **Pre-flight check** — skip patients with a single-class or <3% preictal val/test split
2. Fit a `StandardScaler` on `train_aug` only, apply to val/test
3. Build a fresh `DualStreamMambaEEG` with a bottleneck head
4. **Phase 1** (epoch 1 → SWA start): cosine LR schedule + smoothed early stopping on val AUC
5. **Phase 2** (SWA start → end): flat SWA learning rate, collects a weight snapshot every epoch
6. **SWA finalization**: averages all Phase 2 snapshots into the final model
7. Evaluate on the held-out `test` split (accuracy, F1, AUC, confusion matrix)

## Requirements

```
torch
mambapy
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

```bash
pip install mambapy
```

## Usage

1. Point `Config.DATA_DIR` at the processed/augmented CHB-MIT `.npz` files (one train/val/test triplet per patient).
2. Run the notebook top to bottom. Each patient is trained, evaluated, and plotted (loss, val F1, val AUC, test confusion matrix) in sequence.
3. Trained models and per-patient test metrics are saved to `/kaggle/working/patient_models/`, zipped to `all_patient_models.zip`, and combined into a single `big_combined_model.pt` dictionary keyed by patient ID.

## Output

- Per-patient checkpoints (`{pid}_model.pt`) containing the state dict, architecture config, and test metrics
- A summary table of accuracy / F1 / AUC per patient plus the dataset-wide average
- Training curves and a test confusion matrix per patient

## Notes

- No patient's data is ever used to train another patient's model — each model is independent.
- AUC is undefined (`n/a`) for any patient whose test split ends up single-class after augmentation/splitting.
