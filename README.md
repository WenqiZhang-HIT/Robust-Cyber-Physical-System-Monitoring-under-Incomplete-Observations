# Robust-Cyber-Physical-System-Monitoring-under-Incomplete-Observations
# Missing-aware GAT-VAE for Cyber-Physical System Anomaly Detection

Reference implementation accompanying the paper **"Robust Cyber-Physical System
Monitoring under Incomplete Observations via Missing-aware Graph Variational
Inference."** The model treats each sensor/actuator channel as a graph node,
encodes a sliding window of its readings with a Graph Attention Network (GAT),
and learns a graph-level Variational Autoencoder (VAE) that is explicitly
aware of missing observations (via a per-point mask) at both the encoding and
loss-computation stages. Anomalies are scored by combining a masked
reconstruction error with a KL-divergence-based confidence score.

## Features

- **Missing-aware encoding**: a binary mask marks NaN/dropped values; masked
  entries are zeroed in the input and excluded from the reconstruction loss
  and from edge weighting in the graph.
- **Synthetic missingness for training**: `--missing-rate` randomly drops
  additional points during training so the model learns to be robust to
  incomplete observations at test time.
- **Dual anomaly scoring**: a reconstruction-error score and a KL-divergence
  "confidence" score can be combined (`OR`/`AND`/single-score logic, with an
  `auto` mode that searches all four options on the validation split).
- **Configurable evaluation protocol**: standard F1, point-adjusted F1
  (PA-F1), or a recall-constrained mode, each with its own threshold- and
  post-processing-search routine (smoothing, hysteresis, gap filling, minimum
  segment length).
- Mixed-precision training, gradient clipping, `ReduceLROnPlateau`, and early
  stopping.

## Installation

```bash
pip install torch torch_geometric numpy pandas scikit-learn
```

Install a `torch_geometric` build that matches your local `torch`/CUDA
version; see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Datasets

The paper evaluates on **SWaT**, **WADI**, and **BATADAL**, three widely used
ICS/water-treatment testbed datasets.

- **SWaT** and **WADI** are distributed by the **iTrust Centre for Research
  in Cyber Security, Singapore University of Technology and Design**. They
  are *not* included in this repository. Researchers must submit a request
  directly to iTrust (https://itrust.sutd.edu.sg/itrust-labs_datasets/) and
  agree to their data-usage terms to obtain the full datasets.
- **BATADAL** is available from the [BATADAL competition site](https://batadal.net/).

This repository only provides the model and training/evaluation code. Once
you have obtained the raw data under your own agreement with iTrust (or from
the BATADAL site), preprocess it into two CSV files — one for training, one
for testing — each with one column per sensor/actuator and a binary label
column (default name `label`, override with `--label-col`).

## Usage

```bash
python Missing-aware_GAT-VAE_main.py \
    --train-path data_mask/train_swat.csv \
    --test-path  data_mask/swat_test.csv \
    --label-col label \
    --window-size 30 \
    --epochs 30
```

Key arguments (run `--help` for the full list):

| Argument | Default | Description |
|---|---|---|
| `--window-size` | 30 | sliding window length (time steps) |
| `--missing-rate` | 0.05 | synthetic missing-value rate injected during training |
| `--score-alpha` | 0.5 | weight between mean and top-k sensor-wise reconstruction error |
| `--mode` | `standard` | evaluation objective: `standard` (F1), `pa` (point-adjusted F1), `recall` (recall-constrained) |
| `--use-dual-threshold` / `--no-dual-threshold` | dual on | combine reconstruction + confidence scores vs. reconstruction only |
| `--dual-logic` | `auto` | `or` / `and` / `recon` / `confidence` / `auto` (search all) |
| `--auto-calibrate-score-direction` | off | flip a score dimension whose validation AUC is below 0.5 |
| `--keep-train-anomaly` | off | by default, training windows labeled anomalous are dropped (train on normal data only) |
| `--block-size`, `--val-block-every` | 300, 3 | block-wise holdout evaluation, an alternative to the contiguous validation split |

The script trains the model, then reports metrics twice:

1. **Contiguous-split reference**: a single contiguous, label-ratio-matched
   block is carved out of the test set as a validation split
   (`contiguous_threshold_split`) to select the threshold and
   post-processing parameters; the remaining test points are scored with
   those fixed settings.
2. **Block-wise holdout (recommended)**: the test set is chopped into fixed-
   size blocks, and every `val_block_every`-th block is used for
   threshold/post-processing selection instead of one contiguous run, which
   is less sensitive to where the validation block happens to fall.

## Repository contents

- `Missing-aware_GAT-VAE_main.py` — data loading, the `SensorGraphVAE`
  model, training loop, scoring, thresholding/post-processing search, and
  the two evaluation protocols described above, all in a single script.

## Citation

If you use this code, please cite:

> *Robust Cyber-Physical System Monitoring under Incomplete Observations via
> Missing-aware Graph Variational Inference* (citation details to be added
> upon publication).

