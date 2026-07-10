# Dual-Stream Mamba EEG — Per-Patient Seizure Prediction

Per-patient seizure prediction model built on Mamba (state-space models) instead of transformers. Each patient gets their own model, trained only on their own EEG data.

Dataset: [Seizure Prediction Dataset](https://www.kaggle.com/datasets/hebametwaly/seizure-prediction-dataset) by Heba Metwaly (CHB-MIT, processed + augmented).

## The idea

EEG has two axes worth paying attention to: time (does a channel's signal look different leading up to a seizure?) and space (does the pattern across electrodes look off at a given moment?). So instead of flattening everything into one sequence, I split it into two streams:

- a **temporal Mamba** that scans across time patches for each channel
- a **channel Mamba** that scans across channels for each time patch

Then I fuse them with a simple gate (sigmoid, no attention) and classify through a small bottleneck head.

```
Input (batch, 2560, C)
  → PatchEmbed Conv2d
  → SpectralBranch (depthwise conv, cheap local freq features)
  → N x DualStreamBlock (temporal Mamba + channel Mamba + gated fusion)
  → AdaptiveAvgPool
  → BottleneckHead: Linear(D→D/2) → GELU → Dropout(0.30) → Linear(D/2→2)
```

## Design choices

The training data for some patients is heavily synthetic — as much as 100 augmented samples for every real one — so a lot of the design here is about not letting the model cheat by memorizing:

- **SWA instead of best-checkpoint early stopping.** Some patients' val sets have fewer than 200 samples, so val AUC bounces around a lot epoch to epoch. Picking a single "best" epoch off that is basically picking a lucky seed. Instead, I average the weights from the back half of training — flatter minimum, less noise-chasing.
- **Pre-flight check per patient.** A handful of patients (chb06, chb08, chb12, chb13, chb24) have single-class val/test splits or no real training data at all. Training on them either crashes AUC or gives a model with zero real validation signal, so they get skipped automatically with a printed reason instead of silently producing garbage.
- **Bottleneck classification head.** Linear(32→16) → GELU → Dropout → Linear(16→2), instead of going straight from pooled features to logits. Forcing everything through 16 dimensions means the head can't just memorize individual training samples.
- **Smoothed early stopping on AUC, not loss.** Val loss is easy to game — predicting ~0.5 for everything gives a decent loss. AUC actually measures whether the model ranks preictal samples above normal ones, which is the question that matters clinically. I average it over a 5-epoch window so one noisy epoch doesn't trigger an early stop.
- **Dropout everywhere.** Embed (0.15), SSM (0.10), gate (0.15), head (0.30). Given how synthetic the data is, the model needs real forced regularization since it has no organic incentive to generalize.

## Config

Nothing fancy, all in one `Config` class in the notebook:

- `D_MODEL = 64`, `N_BLOCKS = 3`, `D_STATE = 16`, `EXPAND = 2`, `D_CONV = 2`
- `PATCH_SIZE = 80` → 2560 / 80 = 32 patches per window
- `BOTTLENECK_DIM = 16`
- `BATCH_SIZE = 16`, `EPOCHS = 30`
- `LR = 1e-3`, `SWA_LR = 2e-4`, `WEIGHT_DECAY = 2e-3`
- SWA kicks in at 50% of epochs (`SWA_START_FRAC = 0.50`)
- Early stopping: 5-epoch smoothing window, patience 9

## How training works, per patient

1. Pre-flight check — skip if val/test is single-class or has <3% preictal samples
2. Fit a StandardScaler on `train_aug` only (never touch val/test stats)
3. Fresh model per patient — nothing is shared across patients
4. Phase 1: cosine LR schedule + smoothed early stopping on val AUC
5. Phase 2: flat SWA LR, grab a weight snapshot every epoch
6. Average all Phase 2 snapshots into the final weights
7. Test once, on the held-out test split

No patient ever touches another patient's data — each model is trained and evaluated in complete isolation.

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

## Running it

Point `Config.DATA_DIR` at wherever the processed/augmented `.npz` files live (one per patient, split into train_aug/val/test), then just run the notebook top to bottom. It'll loop through every patient, print params + skip reasons, train, plot loss/F1/AUC curves plus a confusion matrix, and dump results into a summary table at the end.

Models get saved to `patient_models/`, zipped up, and also combined into one `big_combined_model.pt` keyed by patient ID, so you don't have to juggle 15+ separate files.

## Known issues / TODO

- chb06, chb08, chb12, chb13, chb24 still get skipped — no real fix for this without more real (non-augmented) data for those patients
- Haven't tried this on a different dataset yet, so no idea how well it generalizes outside CHB-MIT
- SWA start fraction (0.50) and dropout values were tuned by hand, not swept — probably room to do a proper sweep here

## Notes

AUC shows as `n/a` for any patient whose test split ends up single-class — this is expected, not a bug.
