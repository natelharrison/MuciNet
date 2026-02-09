# Network Quantification Pipeline

A high-throughput pipeline for quantifying filamentous networks in 2D fluorescence microscopy images.
Metrics include per-image network pixel counts and summary plots. Includes batch QC.

---
## Methodology
- Networks: MONAI AttentionUnet segmentation on chan1 images.
- Reproducibility: Run parameters and environment details are recorded in `analysis_runs/<run_id>/run_meta.json`.

---
## Features
- End-to-end from OIB -> MIPs -> metrics/plots
- Batch processing
- Per-image stats plus combined CSVs and plots with run metadata

---
## Installation
1) Clone/download and enter the repo
```
git clone <repo_url>
cd abboud_project
```
2) Create env
```
conda env create -f environment.yml
```
3) Activate
```
conda activate mucin_env
```
4) (Optional) Micro-SAM + GPU
```
conda install -c conda-forge micro_sam           # CPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # GPU (optional)
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

### Analysis-ready layout (per-genotype folders under each trial)
```
<Project_Root>/
|-- Trial_01/
|   |-- Rosa/               # WT example
|   |   |-- MIPs/
|   |   |   |-- chan0/
|   |   |   |-- chan1/
|   |   `-- analysis_results/
|   `-- V1/                 # Mutant example
|       |-- MIPs/
|       |   |-- chan0/
|       |   |-- chan1/
|       `-- analysis_results/
`-- Trial_02/
    `-- ...
```
Notes:
- Folder names are used for grouping; see Outputs and Notes.

---
## Workflow

### 1) Convert OIBs to max-projection TIFs (optional)
```
python src/convert_oibs.py "<project_root>" --skip-existing
```
- Scans for `.oib` and writes max-projections to `MIPs/chan0` and `MIPs/chan1`.
- `--skip-existing` avoids overwriting existing TIFs.

### 2) Run automated analysis
```
python run_analysis.py "<project_root>" --run-id <optional_name>
```
- Finds every `MIPs/chan1/*.tif` (trial/genotype aware).
- Writes per-image outputs to each genotype's `analysis_results/`.
- Writes combined stats/plots/metadata to `<project_root>/analysis_runs/<run_id>/` (timestamp if not set).
- If your model is not at `model_training/lean_best.pth`, pass `--model-path`.
 - Optional one-click: run `run_analysis.bat` and follow the prompts.

### 3) (Optional) Train MONAI segmentation model
```
python model_training/monai_train.py --dir "E:\path\to\training\data"
```
- Expects images and masks in the same directory; mask files default to the `_mask.tif` suffix.
- The model checkpoint saves to the parent directory of `--dir` as `lean_best.pth`.
- Uses early stopping on validation Dice; training can run up to 1200 epochs.

---
## Outputs
- Per genotype `analysis_results/`:
  - `network_binary/`, `network_overlay/`
  - `quantification/` per-image CSVs
- Project-level `analysis_runs/<run_id>/`:
  - `DATA_Networks_Combined.csv`
  - `PLOT_Per_Trial_Breakdown.png`, `PLOT_Global_Summary.png`
  - `run_meta.json` (parameters, platform, run id)

---
## Notes
- Grouping: each image is grouped by its trial folder name and its immediate subfolder name.
- Normalization is within each comparison group; ensure WT controls exist per group.
- Inspect QC overlays and montages to confirm detections.
