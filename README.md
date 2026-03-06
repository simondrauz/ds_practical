# Trajectron++ Integration and Trajectory Metric Analysis

This repository contains:

- An integrated Trajectron++ training/evaluation pipeline (`train_unified.py`).
- A vendored `trajdata` loader implementation (`unified-av-data-loader/src/trajdata`).
- Analysis utilities for agent-centric and scene-centric trajectory characteristics.
- A script to join characteristic metrics onto per-trajectory model metrics.
- Notebooks for setup validation and exploratory analysis on nuScenes.

The main workflow is:

1. Set up environment and dataset paths.
2. Train/evaluate Trajectron++ and export per-trajectory evaluation CSVs.
3. Compute and join characteristic metrics to those CSVs.
4. Analyze results in notebooks.

## Repository Contents

- `train_unified.py`: Main multi-GPU/multi-process training and evaluation entrypoint (via `torchrun`).
- `src/trajectron/`: Trajectron++ model, evaluation, utilities, and metric-analysis modules.
- `scripts/join_characteristic_metrics.py`: Joins characteristic metrics onto `eval_epoch_*.csv`.
- `config/nuScenes.json`: Default training hyperparameter config used by `--conf`.
- `config/analysis_config.yaml`: Notebook analysis configuration.
- `experiments/nuScenes/user_config.py`: User-specific local paths for data/cache.
- `experiments/nuScenes/preprocess_challenge_splits.py`: Generates prediction-challenge split index files.
- `experiments/trajectory_metrics/`: Per-epoch evaluation CSVs from training.
- `experiments/trajectory_metrics_joined/`: Joined CSV/Parquet outputs with characteristic metrics.
- `unified-av-data-loader/`: Vendored `trajdata` package source.
- `notebooks/`:
  - `00_setup_validation.ipynb`
  - `01_data_characterization.ipynb`
  - `02_agent_centered_trajectory_metrics_analysis.ipynb`
  - `03_scene_centered_trajectory_metrics_analysis.ipynb`

## Prerequisites

- Python `3.10.16` recommended (see `runtime.txt`).
- `pip` and virtualenv tooling.
- nuScenes data files (mini and/or trainval).
- GPU recommended for training.

## Installation

From repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# See note in requirements.txt for why this is installed separately.
pip install --no-dependencies l5kit==1.5.0

# Install vendored trajdata and this repo in editable mode.
pip install -e unified-av-data-loader
pip install -e .
```

Quick sanity checks:

```bash
python train_unified.py --help
python scripts/join_characteristic_metrics.py --help
python -c "import trajectron; import trajectron.analysis.characteristic_metrics"
```

## Dataset Setup (nuScenes)

Download nuScenes from https://www.nuscenes.org/download and place files under `data/raw/`.

Typical layout:

```text
data/raw/
├── maps/
├── samples/
├── sweeps/
├── v1.0-mini/
└── v1.0-trainval/   # if using full trainval
```

Cache directory used by trajdata is typically:

```text
data/processed/trajdata_cache/
```

## Configure Local Paths

Two options are supported:

1. Use `--user` with `experiments/nuScenes/user_config.py`.
2. Pass paths directly via CLI (`--trajdata_cache_dir` and `--data_loc_dict`).

If your username is not in `experiments/nuScenes/user_config.py`, add a profile before running with `--user`.

Example direct override:

```bash
--trajdata_cache_dir /abs/path/to/trajdata_cache \
--data_loc_dict '{"nusc_trainval": "/abs/path/to/data/raw"}'
```

## Prediction-Challenge Split Indexes

For `nusc_trainval-*` training/evaluation, the split index files under `experiments/nuScenes/` are required:

- `predchal_train_index.pkl`
- `predchal_train_val_index.pkl`
- `predchal_val_index.pkl`

They are already present in this repo. Regenerate only if needed:

```bash
cd experiments/nuScenes
python preprocess_challenge_splits.py --user <your_user>
cd ../..
```

## Training and Evaluation

`train_unified.py` performs both training and periodic evaluation.  
During evaluation it writes per-trajectory CSVs to:

```text
experiments/trajectory_metrics/<run_name>/eval_epoch_<N>.csv
```

### Example: quick mini sanity run

```bash
torchrun --nproc_per_node=1 train_unified.py \
  --user simon \
  --conf config/nuScenes.json \
  --train_data nusc_mini-mini_train \
  --eval_data nusc_mini-mini_val \
  --history_sec 2.0 \
  --prediction_sec 6.0 \
  --batch_size 64 \
  --eval_batch_size 64 \
  --train_epochs 1 \
  --eval_every 1 \
  --save_every 1 \
  --log_dir experiments/nuScenes/models \
  --log_tag nusc_mini_debug_tpp
```

### Example: trainval-style run

```bash
torchrun --nproc_per_node=1 train_unified.py \
  --user simon \
  --conf config/nuScenes.json \
  --train_data nusc_trainval-train \
  --eval_data nusc_trainval-train_val \
  --history_sec 2.0 \
  --prediction_sec 6.0 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --preprocess_workers 16 \
  --train_epochs 20 \
  --eval_every 1 \
  --save_every 1 \
  --log_dir experiments/nuScenes/models \
  --log_tag nusc_adaptive_tpp
```

Notes:

- `train_unified.py` uses Weights & Biases (`wandb`). Configure as needed (or disable with `WANDB_MODE=disabled`).
- The default config path is `config/nuScenes.json`.

## Compute and Join Characteristic Metrics

After training has produced `eval_epoch_*.csv`, run:

```bash
python scripts/join_characteristic_metrics.py \
  --conf experiments/nuScenes/models/<run_name>/config.json \
  --run_dir experiments/trajectory_metrics/<run_name> \
  --output_root experiments/trajectory_metrics_joined \
  --format csv
```

Optional:

- `--incl_vector_map` to enable `off_road` computation (slower).
- `--trajdata_cache_dir`, `--data_loc_dict`, and other overrides if your runtime paths differ from training config.

Output files are written to:

```text
experiments/trajectory_metrics_joined/<run_name>/eval_epoch_<N>.csv
```

## Notebook Workflow

Suggested order:

1. `notebooks/00_setup_validation.ipynb`
2. `notebooks/01_data_characterization.ipynb`
3. `notebooks/02_agent_centered_trajectory_metrics_analysis.ipynb`
4. `notebooks/03_scene_centered_trajectory_metrics_analysis.ipynb`

`config/analysis_config.yaml` centralizes notebook analysis settings (dataset mode, paths, thresholds, output dirs).

## Common Troubleshooting

- `ModuleNotFoundError: trajectron` or `trajdata`:
  - Re-run editable installs: `pip install -e unified-av-data-loader && pip install -e .`
- Missing config file:
  - Ensure `--conf` points to a valid JSON, or use default `config/nuScenes.json`.
- Join script reports missing eval CSV:
  - Confirm training ran with evaluation enabled and wrote to `experiments/trajectory_metrics/...`.
- Join results look misaligned:
  - Use the same dataset split/history/future/path settings as the training run config.
- nuScenes challenge split errors:
  - Ensure `predchal_*.pkl` files exist in `experiments/nuScenes/`.

## Acknowledgements

This codebase integrates and adapts:

- Trajectron++ (Stanford ASL / NVIDIA contributors)
- trajdata (NVLabs / NVIDIA contributors)

See `LICENSE`, `CITATIONS.bib` (if present in your branch), and package metadata for attribution details.
