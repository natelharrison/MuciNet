# Network Quantification Pipeline

A high-throughput pipeline for quantifying web/wrinkle like mucin networks in 2D fluorescence microscopy images of mice corneal epithilium.
Metrics include per-image network pixel counts and summary plots. Includes batch QC.

---
## Methodology
- Networks: MONAI AttentionUnet segmentation on chan1 images.

---
## Features
- End-to-end from OIB -> chan1 max-projection -> metrics/plots
- Batch processing
- Per-image stats plus combined CSVs and plots

---
## Installation
1) Clone/download and enter the repo
```
git clone <repo_url>
cd mucinet
```
2) Create env
```
conda env create -f environment.yml
```
3) Activate
```
conda activate mucinet
```
4) (Optional) Enable GPU PyTorch after install
- This environment installs CPU PyTorch by default.
- To switch to GPU (CUDA 12.1), uninstall CPU PyTorch packages and reinstall CUDA-enabled PyTorch:
```
conda remove pytorch cpuonly
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

---
## Data Organization

### Raw OIBs
```
<Project_Root>/
|-- Trial_01/    # original .oib files
|-- Trial_02/
|-- Trial_03/
```

### Analysis-ready layout (optional comparison subfolders under each trial)
```
<Project_Root>/
|-- Trial_01/
|   |-- WT/                 # optional comparison folder (WT is treated specially in plots)
|   |   |-- sampleA.oib
|   |   `-- MAX_chan1_sampleA.tif
|   `-- Mut/                # any other folder name is treated as a comparison group label
|       |-- sampleB.oib
|       `-- MAX_chan1_sampleB.tif
`-- Trial_02/
    `-- ...
```
Notes:
- Folder names are used for grouping; see Outputs and Notes.

---
## Workflow

### 1) Convert OIBs to chan1 max-projection TIFs (optional)
```
python src/convert_oibs.py "<project_root>"
```
- Scans for `.oib` and writes `MAX_chan1_<oib_stem>.tif` next to each `.oib`.
- Use `--force-convert` to overwrite existing `MAX_chan1_*.tif`.

### 2) Run automated analysis
```
python run_analysis.py "<project_root>"
```
- Finds every chan1 MIP TIFF under the project root:
  - `MAX_chan1_*.tif` next to `.oib`, or
  - `MIPs/chan1/*.tif` if you already generated MIPs elsewhere.
- Writes outputs under `<project_root>/mucinet-results/` mirroring the dataset structure.
- If your model is not at `model_training/lean_best.pth`, pass `--model-path`.
- Optional: run conversion first via `--convert-oibs` (plus `--force-convert` if needed).
- Optional: generate a single full-resolution global montage via `--montage`.

### 3) (Optional) Train MONAI segmentation model
```
python model_training/monai_train.py --dir "E:\path\to\training\data"
```
- Expects images and masks in the same directory; mask files default to the `_mask.tif` suffix.
- The model checkpoint saves to the parent directory of `--dir` as `lean_best.pth`.
- Uses early stopping on validation Dice; training can run up to 1200 epochs.

---
## Outputs
- Per-image outputs under `<project_root>/mucinet-results/` mirroring your dataset structure:
  - `network_binary/`, `network_overlay/`, `metrics/`
- Project-level outputs under `<project_root>/mucinet-results/`:
  - `combined_metrics.csv`
  - `per_trial_bar.png`, `global_bar.png`
  - `global_montage.png` (only when `--montage` is used)

---
## Notes
- Grouping: each image is grouped by its trial folder name and its immediate subfolder name.
- Normalization is within each comparison group; ensure WT controls exist per group.
- Inspect QC overlays and montages to confirm detections.
