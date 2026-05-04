# Trajectron++ nuScenes Interpretation Pipelines

This repository contains an integrated Trajectron++ training/evaluation setup,
trajectory and scene metric joins, and interpretable modelling notebooks for
analysing trajectory-prediction error on nuScenes.

There are two supported analysis paths:

1. **Full trainval path**
   Run one Trajectron++ configuration on the full nuScenes trainval split, join
   evaluation output with trajectory and scene characteristics, fit GAM and
   XGBoost interpretable models, run model inference, then continue into
   performance-regime clustering and cluster inspection.

2. **Mini model-settings sweep path**
   Run a Cartesian sweep of Trajectron++ settings on nuScenes mini, combine only
   the joined outputs from the current sweep, fit GAM and XGBoost interpretable
   models, and stop after `model_inference_analysis.ipynb`. This path does not
   run clustering or cluster inspection.

## Repository Contents

- `train_unified.py`: Trajectron++ training/evaluation entrypoint.
- `run_sweep.py`: sequential model-settings sweep runner for nuScenes mini.
- `scripts/validate_pipeline_paths.py`: capped integration validation for both
  supported paths.
- `src/trajectron/`: Trajectron++ model, evaluation, and utilities.
- `src/data_preparation/join_characteristic_metrics.py`: joins trajectory and
  scene characteristics onto `eval_epoch_*.csv`.
- `src/data_preparation/combine_runs.py`: safely combines explicitly selected
  joined runs.
- `src/data_modelling/`: interpretable modelling helpers and notebooks.
- `config/shared_config.yaml`: shared agent filtering, attention radius, and map
  settings.
- `config/nuScenes.json`: upstream-aligned Trajectron++ nuScenes base config.
- `config/nuScenes_full_trainval.json`: full nuScenes trainval run overlay.
- `config/nuScenes_mini.json`: nuScenes mini run overlay.
- `config/sweep_config.yaml`: example model-settings sweep configuration. It
  inherits the mini run overlay and should be copied only when your machine
  needs local path overrides.
- `results/trajectory_prediction/trajectory_metrics/`: per-epoch evaluation CSVs
  from Trajectron++.
- `results/trajectory_prediction/trajectory_metrics_joined/`: joined
  evaluation/characteristic tables.
- `results/interpretable_model/`: prepared data, model outputs, inference
  outputs, clustering, and inspection artifacts.
- `unified-av-data-loader/`: vendored `trajdata` source.

## Environment

Use the conda environment `adaptive-py310` when available:

```bash
export PYTHONPATH=src:unified-av-data-loader/src
export WANDB_MODE=disabled
export MPLBACKEND=Agg

conda run -n adaptive-py310 python -m pytest tests
conda run -n adaptive-py310 python run_sweep.py --dry_run
```

If you need to install from scratch:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-dependencies l5kit==1.5.0
pip install -e unified-av-data-loader
pip install -e .
```

## Dataset Setup

For nuScenes mini, the repo expects the usual local layout unless overridden:

```text
data/raw/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/

data/processed/trajdata_cache/
```

For full trainval on the current project machine, raw files and cache are on the
LaCie drive:

```text
/Volumes/LaCie 1TB/nuScenes/v1.0-trainval_raw
/Volumes/LaCie 1TB/nuScenes/trajdata_cache
```

`config/nuScenes_full_trainval.json` uses these paths. Override them on the CLI
if your local layout differs:

```bash
--trajdata_cache_dir "/path/to/trajdata_cache" \
--data_loc_dict '{"nusc_trainval": "/path/to/v1.0-trainval_raw"}'
```

Path precedence is:

1. Explicit CLI flags such as `--trajdata_cache_dir` and `--data_loc_dict`.
2. Values in the resolved JSON config passed to `--conf`.
3. `--user` profile paths from `config/experimental_setup/nuScenes/user_config.py`,
   but only when the selected config does not already define the path key.
4. Repo-relative fallback paths.

The dedicated mini and full config overlays already define `trajdata_cache_dir`
and `data_loc_dict`, so `--user simon` or `--user zoe` does not replace those
paths. For shared work across machines, either keep the same repo-local mini
layout, pass path overrides on the CLI, or use a local uncommitted config/sweep
overlay with only path overrides.

The prediction-challenge split index files under
`config/experimental_setup/nuScenes/` are required for trainval split handling
and are already present:

- `predchal_train_index.pkl`
- `predchal_train_val_index.pkl`
- `predchal_val_index.pkl`

The full trainval path described below uses `nusc_trainval-train` and
`nusc_trainval-train_val` without `--restrict_to_predchal`.

## Path 1: Full Trainval Analysis

### 1. Run Trajectron++

Example full trainval command:

```bash
PYTHONPATH=src:unified-av-data-loader/src WANDB_MODE=disabled \
conda run -n adaptive-py310 python -m torch.distributed.run --nproc_per_node=1 \
  train_unified.py \
  --conf config/nuScenes_full_trainval.json
```

`config/nuScenes_full_trainval.json` extends `config/nuScenes.json` and contains
the full trainval split names, LaCie dataset/cache paths, 12 epochs, learning
rate 0.003, and Pedestrian-only evaluation. Override paths on the CLI if your
machine uses a different trainval layout.

Training writes evaluation CSVs to:

```text
results/trajectory_prediction/trajectory_metrics/<run_name>/eval_epoch_<N>.csv
```

`train_unified.py` also supports validation-only batch caps:

```bash
--max_train_batches 5 --max_eval_batches 5
```

Do not use those caps for final reported results.

### 2. Join Evaluation Metrics With Characteristics

Use the saved run config so the join reconstructs the same eval context and the
same persisted attention-radius map:

```bash
PYTHONPATH=src:unified-av-data-loader/src MPLBACKEND=Agg \
conda run -n adaptive-py310 python -m data_preparation.join_characteristic_metrics \
  --conf results/trajectory_prediction/nuScenes/models/<run_name>/config.json \
  --run_dir <run_name> \
  --output_root results/trajectory_prediction/trajectory_metrics_joined \
  --format csv \
  --incl_vector_map \
  --trajdata_cache_dir "/path/to/trajdata_cache" \
  --data_loc_dict '{"nusc_trainval": "/path/to/v1.0-trainval_raw"}' \
  --preprocess_workers 16
```

The join path flags can be omitted when the saved run `config.json` already
contains valid paths for the machine running the join. Include them when moving
outputs across machines or when the saved config points at a different local
layout.

Output:

```text
results/trajectory_prediction/trajectory_metrics_joined/<run_name>/eval_epoch_<N>.csv
```

The joined table includes the per-trajectory prediction metrics, trajectory
characteristics, scene characteristics, and realized model settings:

```text
attention_radius_m
history_sec
prediction_sec
```

### 3. Run Interpretable Modelling and Inspection

Run the notebooks in this order, setting `RUN_NAME` and `EVAL_CSV_NAME` to the
joined trainval output:

1. `src/data_modelling/interpretable_model_data_preparation.ipynb`
2. `src/data_modelling/gam.ipynb`
3. `src/data_modelling/xgboost.ipynb`
4. `src/data_modelling/model_inference_analysis.ipynb` for `MODEL_ID="gam"`
5. `src/data_modelling/model_inference_analysis.ipynb` for `MODEL_ID="xgboost"`
6. `src/data_modelling/feature_effect_performance_regimes.ipynb` for each model
7. `src/data_modelling/feature_effect_pr_cluster_inspection.ipynb` for selected
   non-empty cluster candidates

The full trainval path is the only path that continues into performance-regime
clustering and cluster inspection.

## Path 2: Mini Model-Settings Sweep

The sweep path is for studying how prediction performance varies with the
realized model settings:

```text
attention_radius_m
history_sec
prediction_sec
```

`attention_radius_scale` appears in `config/sweep_config.yaml` only as the
internal multiplier used to produce different realized `attention_radius_m`
values. It is not copied into the combined dataframe and is not used as a GAM or
XGBoost feature.

### 1. Create a Local Sweep Config

`config/sweep_config.yaml` is an example with repo-local paths. Copy it to an
ignored or temporary location and edit the paths if your machine uses a
different data layout:

```bash
mkdir -p results/interpretable_model/local_configs
cp config/sweep_config.yaml results/interpretable_model/local_configs/sweep_config.local.yaml
```

For local mini data in this repo, use:

```yaml
base_args:
  conf: config/nuScenes_mini.json
  log_tag: sweep_tpp

grid:
  history_sec: [2.0, 4.0]
  prediction_sec: [2.0, 4.0, 6.0]
  attention_radius_scale: [0.5, 1.0, 2.0]
```

`config/nuScenes_mini.json` supplies the mini split/path settings, 40 epochs,
learning rate 0.003, and five-epoch eval/save cadence. Downstream analysis can
therefore choose between epochs 25, 30, 35, and 40.
If your mini data is not under `data/raw` and `data/processed/trajdata_cache`,
add `trajdata_cache_dir` and `data_loc_dict` overrides under `base_args` in your
local sweep config.

Example local-path override:

```yaml
base_args:
  conf: config/nuScenes_mini.json
  log_tag: sweep_tpp
  trajdata_cache_dir: /path/to/mini_trajdata_cache
  data_loc_dict: '{"nusc_mini": "/path/to/mini_raw"}'
```

### 2. Run the Sweep

```bash
PYTHONPATH=src:unified-av-data-loader/src WANDB_MODE=disabled MPLBACKEND=Agg \
conda run -n adaptive-py310 python run_sweep.py \
  --sweep_config results/interpretable_model/local_configs/sweep_config.local.yaml \
  --metrics_root results/trajectory_prediction/trajectory_metrics \
  --joined_root results/trajectory_prediction/trajectory_metrics_joined \
  --format csv
```

`run_sweep.py` runs each combination sequentially. For every combination it:

1. scales `config/shared_config.yaml` attention radii by
   `attention_radius_scale`;
2. runs `train_unified.py` on `nusc_mini-mini_train` and `nusc_mini-mini_val`;
3. joins the evaluation output with trajectory and scene characteristics;
4. restores `config/shared_config.yaml`;
5. combines only the joined run directories produced by the current sweep.

The combined output is written to:

```text
results/trajectory_prediction/combined_runs.csv
```

`combine_runs.py` refuses implicit all-runs combines. If you combine manually,
pass explicit run directories:

```bash
PYTHONPATH=src:unified-av-data-loader/src \
conda run -n adaptive-py310 python -m data_preparation.combine_runs \
  --joined_root results/trajectory_prediction/trajectory_metrics_joined \
  --run_dirs <current_sweep_run_1> <current_sweep_run_2> \
  --output results/trajectory_prediction/combined_runs \
  --format csv
```

Use `--all_runs` only when you intentionally want every joined run under the
joined root.

### 3. Bridge the Combined Sweep Table Into the Notebook Contract

The preparation notebook consumes run-like files under
`trajectory_metrics_joined/<run_name>/eval_epoch_*.csv`. After a sweep, bridge
the combined table into that layout:

```bash
SWEEP_RUN_NAME=sweep_combined
SWEEP_EVAL_CSV=eval_epoch_sweep_combined.csv

mkdir -p "results/trajectory_prediction/trajectory_metrics_joined/${SWEEP_RUN_NAME}"
cp results/trajectory_prediction/combined_runs.csv \
  "results/trajectory_prediction/trajectory_metrics_joined/${SWEEP_RUN_NAME}/${SWEEP_EVAL_CSV}"
```

Then set in `src/data_modelling/interpretable_model_data_preparation.ipynb`:

```python
RUN_NAME = "sweep_combined"
EVAL_CSV_NAME = "eval_epoch_sweep_combined.csv"
```

### 4. Run Modelling and Stop at Inference

Run only:

1. `src/data_modelling/interpretable_model_data_preparation.ipynb`
2. `src/data_modelling/gam.ipynb`
3. `src/data_modelling/xgboost.ipynb`
4. `src/data_modelling/model_inference_analysis.ipynb` for `MODEL_ID="gam"`
5. `src/data_modelling/model_inference_analysis.ipynb` for `MODEL_ID="xgboost"`

Do not run `feature_effect_performance_regimes.ipynb` or
`feature_effect_pr_cluster_inspection.ipynb` for the model-settings sweep path.

## Validation Before Expensive Runs

Use the capped harness before full trainval or full sweep runs, especially after
changes to training, eval CSV writing, metric joining, sweep combination, or
modelling notebooks:

```bash
PYTHONPATH=src:unified-av-data-loader/src WANDB_MODE=disabled MPLBACKEND=Agg \
conda run -n adaptive-py310 python scripts/validate_pipeline_paths.py
```

The harness:

- validates imports, conda environment, data roots, caches, split files, and
  trajdata dataset construction;
- samples first/middle/last records from mini and trainval splits;
- runs a capped full-trainval path through cluster inspection;
- runs a capped mini sweep path through model inference only;
- writes executed notebook copies and a JSON summary under
  `results/interpretable_model/notebook_runs/...`;
- restores `config/shared_config.yaml` byte-for-byte after sweep validation.

The latest validation during this audit passed at:

```text
results/interpretable_model/notebook_runs/pipeline_validation_20260503_100713/validation_summary.json
```

## Notes for `dev-model-setting-inclusion` Users

If you previously worked from `dev-model-setting-inclusion`, the intended
workflow is mostly the same, but the contracts are stricter now:

- Keep `attention_radius_scale` in the sweep config as the multiplier that
  creates scaled runs.
- Do not expect `attention_radius_scale` in joined, combined, prepared, GAM, or
  XGBoost feature tables.
- Interpret model-setting effects through `attention_radius_m`, `history_sec`,
  and `prediction_sec`.
- `run_sweep.py` now combines only the joined dirs produced by the current
  sweep. Manual combines require `--run_dirs` or explicit `--all_runs`.
- `config/shared_config.yaml` is temporarily modified during the sweep and
  restored after each run. Avoid killing the process mid-write; if interrupted,
  check this file before continuing.
- The sweep path stops at `model_inference_analysis.ipynb`. Clustering and
  cluster inspection belong to the full trainval path only.
- Regenerate old prepared data, OOF predictions, feature-effect exports, and
  regime outputs before interpretation. Older artifacts may lack stable identity
  columns or persisted attention-radius settings.
- Update local path values before running if needed. `--user` does not override
  path keys that are already present in a selected run config. Do not commit
  machine-specific edits to shared configs; prefer CLI overrides or a local copy
  under `results/` or another ignored location.

## Common Troubleshooting

- `ModuleNotFoundError: trajectron` or `trajdata`:
  re-run editable installs with `pip install -e unified-av-data-loader` and
  `pip install -e .`, or set `PYTHONPATH=src:unified-av-data-loader/src`.
- Missing eval CSVs:
  confirm `train_unified.py` ran with `--eval_every` set and wrote to
  `results/trajectory_prediction/trajectory_metrics/<run_name>/`.
- Join identity/context mismatch:
  use the saved run `config.json` and the same data/cache paths used for
  training. New eval CSVs include identity and context columns so mismatches
  fail loudly.
- Sweep combine includes unexpected rows:
  rerun with current code and explicit current run dirs. Do not use `--all_runs`
  unless that is intentional.
- Notebook results look stale:
  committed notebooks may contain old rendered outputs. Trust freshly executed
  outputs under `results/`, not stale cell output text.

## Acknowledgements

This codebase integrates and adapts:

- Trajectron++ (Stanford ASL / NVIDIA contributors)
- trajdata (NVLabs / NVIDIA contributors)

See `LICENSE`, `CITATIONS.bib` if present, and package metadata for attribution
details.
