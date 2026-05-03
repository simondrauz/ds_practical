# Model Settings Sweep Hand-Off

This note is for Zoe when moving from `dev-model-settings-inclusion` to the
current `dev`/`dev-fixes` workflow. It focuses on how to run and interpret the
model-settings sweep, not on the implementation changes behind it.

## Usage Changes That Matter

- Regenerate sweep artifacts before interpretation. Older joined, combined,
  prepared, OOF, or inference outputs from `dev-model-settings-inclusion` can
  miss the current identity/context columns or persisted attention-radius
  settings.
- Use `attention_radius_scale` only in the sweep config. Downstream tables and
  model features use the realized settings: `attention_radius_m`, `history_sec`,
  and `prediction_sec`.
- `run_sweep.py` now combines only the joined run directories created by the
  current sweep. This avoids accidentally mixing old or unrelated joined runs
  into `combined_runs.csv`.
- Manual combines must be explicitly scoped with `--run_dirs`, unless you
  intentionally pass `--all_runs`.
- The sweep temporarily rewrites `config/shared_config.yaml` while each
  combination is running, then restores it after that combination. Before a new
  sweep, check that the file does not contain a stale scaled attention-radius
  edit from an interrupted run.
- The model-settings path stops after `model_inference_analysis.ipynb`. Do not
  run `feature_effect_performance_regimes.ipynb` or
  `feature_effect_pr_cluster_inspection.ipynb` for this sweep path.

## Preflight

Run these from the repository root.

```bash
git status --short --branch
ps -axo pid,command | rg "run_sweep.py|train_unified.py"
git diff -- config/shared_config.yaml
```

If `run_sweep.py` or `train_unified.py` is still active, wait for it to finish
before restoring or editing `config/shared_config.yaml`. The `git diff` command
should normally print nothing when no sweep is active. If it only shows stale
attention-radius scaling from an interrupted sweep, restore it after confirming
there are no intentional local edits and no active sweep process:

```bash
git restore config/shared_config.yaml
```

## Create A Local Sweep Config

Do not commit machine-specific path edits to `config/sweep_config.yaml`. Copy it
to an ignored local location and edit the paths there.

```bash
mkdir -p results/interpretable_model/local_configs
cp config/sweep_config.yaml results/interpretable_model/local_configs/sweep_config.local.yaml
```

For a repo-local mini setup, the local config should look like this:

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
learning rate 0.003, and five-epoch eval/save cadence, so the analysis can
later use epoch 25, 30, 35, or 40.
If the mini data is not under the repo-local default paths, add
`trajdata_cache_dir` and `data_loc_dict` overrides under `base_args` in the
local sweep config.

Confirm the runner parses the local config and prints the scoped commands before
doing an expensive run:

```bash
PYTHONPATH=src:unified-av-data-loader/src WANDB_MODE=disabled MPLBACKEND=Agg \
conda run -n adaptive-py310 python run_sweep.py \
  --sweep_config results/interpretable_model/local_configs/sweep_config.local.yaml \
  --metrics_root results/trajectory_prediction/trajectory_metrics \
  --joined_root results/trajectory_prediction/trajectory_metrics_joined \
  --format csv \
  --dry_run
```

## Run The Sweep

```bash
PYTHONPATH=src:unified-av-data-loader/src WANDB_MODE=disabled MPLBACKEND=Agg \
conda run -n adaptive-py310 python run_sweep.py \
  --sweep_config results/interpretable_model/local_configs/sweep_config.local.yaml \
  --metrics_root results/trajectory_prediction/trajectory_metrics \
  --joined_root results/trajectory_prediction/trajectory_metrics_joined \
  --format csv
```

Expected sweep output locations:

```text
results/trajectory_prediction/logs/<sweep_run>/config.json
results/trajectory_prediction/trajectory_metrics/<sweep_run>/eval_epoch_<N>.csv
results/trajectory_prediction/trajectory_metrics_joined/<sweep_run>/eval_epoch_<N>.csv
results/trajectory_prediction/combined_runs.csv
```

## Manual Combine, Only If Needed

The normal `run_sweep.py` command already does this for the current sweep. If
you need to recombine by hand, list the current sweep run directories
explicitly:

```bash
PYTHONPATH=src:unified-av-data-loader/src \
conda run -n adaptive-py310 python -m data_preparation.combine_runs \
  --joined_root results/trajectory_prediction/trajectory_metrics_joined \
  --run_dirs <current_sweep_run_1> <current_sweep_run_2> \
  --output results/trajectory_prediction/combined_runs \
  --format csv
```

Use this only when intentionally combining every joined run under the joined
root:

```bash
PYTHONPATH=src:unified-av-data-loader/src \
conda run -n adaptive-py310 python -m data_preparation.combine_runs \
  --joined_root results/trajectory_prediction/trajectory_metrics_joined \
  --all_runs \
  --output results/trajectory_prediction/combined_runs \
  --format csv
```

## Bridge The Combined Table Into The Notebook Layout

The preparation notebook expects a run-like file under
`trajectory_metrics_joined/<run_name>/eval_epoch_*.csv`. Bridge the combined
sweep table into that layout:

```bash
SWEEP_RUN_NAME=sweep_combined
SWEEP_EVAL_CSV=eval_epoch_sweep_combined.csv

mkdir -p "results/trajectory_prediction/trajectory_metrics_joined/${SWEEP_RUN_NAME}"
cp results/trajectory_prediction/combined_runs.csv \
  "results/trajectory_prediction/trajectory_metrics_joined/${SWEEP_RUN_NAME}/${SWEEP_EVAL_CSV}"
```

Then set these notebook variables:

```python
RUN_NAME = "sweep_combined"
EVAL_CSV_NAME = "eval_epoch_sweep_combined.csv"
```

Use that `RUN_NAME` in:

```text
src/data_modelling/interpretable_model_data_preparation.ipynb
src/data_modelling/gam.ipynb
src/data_modelling/xgboost.ipynb
src/data_modelling/model_inference_analysis.ipynb
```

Run `model_inference_analysis.ipynb` once with `MODEL_ID = "gam"` and once with
`MODEL_ID = "xgboost"`.

## Optional Validation Harness

Before a full expensive run, the capped validation harness exercises the trainval
path and the mini sweep path, then restores `config/shared_config.yaml`:

```bash
PYTHONPATH=src:unified-av-data-loader/src WANDB_MODE=disabled MPLBACKEND=Agg \
conda run -n adaptive-py310 python scripts/validate_pipeline_paths.py
```
