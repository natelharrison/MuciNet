# MuciNet

A high-throughput pipeline for quantifying mucin networks in 2D fluorescence microscopy images of mouse corneal epithelium. It segments web/wrinkle-like network structures from TIFF images using a trained deep learning model, then compiles per-image pixel counts and summary plots across experimental groups.

---

## How it works

### Model

The segmentation model is a **MONAI AttentionUNet** operating on 2D single-channel images:

| Parameter | Value |
|---|---|
| Architecture | AttentionUNet (2D) |
| Input / output channels | 1 / 1 |
| Feature channels | 32 → 64 → 128 → 256 → 512 |
| Downsampling strides | 4 × stride-2 |
| Input tile size | 512 × 512 px |

Inference uses MONAI's **SlidingWindowInferer**: the full image is processed as overlapping 512 × 512 tiles (50% overlap, Gaussian blending at tile boundaries). The model outputs a probability map; pixels above 0.5 are classified as network. Connected components smaller than 64 pixels are removed as noise.

### Normalization

**Per-image z-score normalization** is applied both at training time and at inference time:

```
normalized = (pixel_value − image_mean) / image_std
```

This is done independently for each image before it enters the model. It is intentional and critical: fluorescence intensity varies substantially across microscopy sessions, objective lenses, and staining batches, so a global or dataset-wide normalization would cause the model to see very different value ranges across images. Per-image z-score removes this inter-image intensity variation while preserving the local contrast structure the model was trained to detect.

> **Important:** the normalization at inference must exactly match training. Changing it will degrade model performance. Training used MONAI's `NormalizeIntensityd` (which computes per-image z-score over all pixels by default). Inference replicates this with a manual `(x − mean) / std` step in `src/network_detector.py`.

### Training details

The current model (`models/loocv_epoch_0375.pth`) was trained on manually annotated 512 × 512 crops with the following setup:

- **Loss:** TverskyLoss (α = 0.7, β = 0.3) — weighted toward recall to avoid missing sparse network signal
- **Optimizer:** AdamW (lr = 2e-4, weight_decay = 1e-5)
- **Scheduler:** CosineAnnealingWarmRestarts (T₀ = 500, T_mult = 2) — periodic LR spikes to escape local minima
- **Data split:** `GroupShuffleSplit` on source image IDs (80/20) so all 512 × 512 crops from a single annotated image are always in the same split, preventing data leakage
- **Augmentation:** random flips, rotations, affine transforms, Gaussian noise, contrast/intensity jitter

### Output statistics

Per-image network pixel counts are written to CSV files after each run. When a `WT` phenotype folder is present within a trial, pixel counts for all groups in that trial are normalized to the WT mean:

```
normalized_count = image_pixel_count / mean(WT_pixel_counts_for_that_trial)
```

Normalization is per-trial (not global) because illumination and staining intensity can vary between acquisition sessions.

---

## Installation

**Prerequisites:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

```bash
# 1. Clone the repo
git clone <repo_url>
cd MuciNet

# 2. Create and activate the environment (installs CPU PyTorch by default)
conda env create -f environment.yml
conda activate mucinet
```

**Optional — GPU acceleration (CUDA 12.1):**

```bash
conda remove pytorch cpuonly
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

GPU is strongly recommended for large datasets. The pipeline auto-detects CUDA and uses it when available.

---

## Data layout

The pipeline expects images organized as `<ROOT>/<TRIAL>/<PHENOTYPE>/*.tif`. Each trial can have multiple phenotype subfolders; `WT` is treated specially as the normalization baseline.

```
Project_Root/
├── Trial_01/
│   ├── WT/
│   │   ├── sample_A.oib        # raw OIB files (converted automatically)
│   │   └── MAX_sample_A.tif    # max-projection TIFF (created by pipeline)
│   └── KO/
│       ├── sample_B.oib
│       └── MAX_sample_B.tif
└── Trial_02/
    └── ...
```

Rules:
- TIFFs must be inside a phenotype subfolder — files directly under `<ROOT>/<TRIAL>/` are rejected.
- `mucinet-results/` is excluded from scanning automatically.
- Phenotype folder names (e.g. `WT`, `KO`, `Mut`) become the group labels in plots and CSVs.

---

## Running the pipeline

```bash
python run_analysis.py --dir "<Project_Root>"
```

This single command:
1. Converts any `.oib` files to max-projection TIFFs (skips already-converted files)
2. Runs AttentionUNet segmentation on every TIFF
3. Compiles per-image stats into a combined CSV and bar plots

**Options:**

| Flag | Description |
|---|---|
| `--force-convert` | Re-convert OIBs even if `MAX_*.tif` already exists |
| `--model-path <path>` | Use a specific model checkpoint instead of the default |
| `--montage` | Generate a full-resolution global overlay montage |

**Examples:**

```bash
# Standard run
python run_analysis.py --dir "D:/Experiments/Batch_01"

# Force re-conversion of OIBs (e.g. after changing the channel selection)
python run_analysis.py --dir "D:/Experiments/Batch_01" --force-convert

# Use a different model checkpoint
python run_analysis.py --dir "D:/Experiments/Batch_01" --model-path models/my_model.pth

# Generate montage for visual QC
python run_analysis.py --dir "D:/Experiments/Batch_01" --montage
```

### Running steps individually

```bash
# OIB conversion only
python src/convert_oibs.py --dir "<Project_Root>"
python src/convert_oibs.py --dir "<Project_Root>" --force-convert

# Segmentation only (requires TIFFs to already exist)
python src/network_detector.py --dir "<Project_Root>"

# Stats compilation only (requires segmentation outputs to already exist)
python src/compile_stats.py --dir "<Project_Root>"
```

---

## Outputs

All outputs are written under `<Project_Root>/mucinet-results/`, mirroring the input dataset structure.

**Per image:**
- `network_binary/<image>_network_binary.tif` — binary segmentation mask (0/255)
- `network_overlay/<image>_network_overlay.tif` — cyan overlay on grayscale for visual QC
- `metrics/<image>_network_stats.csv` — pixel count plus inference parameters for reproducibility

**Project-level:**
- `combined_metrics.csv` — all per-image stats in one table, with WT-normalized counts
- `per_trial_bar.png` — mean ± SE bar plot grouped by trial and phenotype
- `global_bar.png` — mean ± SE bar plot collapsed across all trials
- `global_montage.png` — full-resolution overlay montage (only with `--montage`)

---

## Training a new model

### 1. Prepare training crops

Annotated images and their masks (suffix `_mask.tif`) should be in the same directory. Running the prep script slices them into 512 × 512 crops with 50% overlap:

```bash
python model_training/prep_train_data.py --data-dir "<annotated_dir>"
```

Output crops are saved to `<annotated_dir>/crops_lean/`. The source image name is encoded in each crop filename (`{source_stem}_{index:04d}.tif`) so the train/val splitter can group them correctly.

### 2. Train

```bash
# With validation split and early stopping (use this to evaluate a new model)
python model_training/monai_train.py --dir "<annotated_dir>/crops_lean"

# Train on all data, no validation (use this for a final production model)
python model_training/monai_train.py --dir "<annotated_dir>/crops_lean" --train-all --max-epochs 600
```

Checkpoints are saved one level above the crops directory. Training logs to WandB (`mucin-segmentation` project).

**Key training flags:**

| Flag | Description |
|---|---|
| `--train-all` | Use all tiles, no validation split or early stopping |
| `--max-epochs N` | Override default epoch limit (2000) |
| `--save-epochs 375 500` | Save checkpoints at specific epochs |
| `--checkpoint-metric` | `mean_tile` / `median_tile` / `nonempty_tile` (default: `mean_tile`) |
| `--no-compile` | Disable `torch.compile` (use if compile is unavailable) |

### 3. Use the new model

Place the `.pth` file in the `models/` directory and point to it with `--model-path`:

```bash
python run_analysis.py --dir "<Project_Root>" --model-path models/new_model.pth
```
