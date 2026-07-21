# Multiple-Seed Trainval And Sweep Runs

This workflow manages the four curated trajectory-prediction result sets used by
the downstream notebooks. Three-seed result sets are seed-averaged only after
all model runs and joins are complete. Single-seed result sets keep the normal
joined or combined output path and do not go through seed aggregation.

## Design

- Full trainval: `config/nuScenes_full_trainval.json`, 12 epochs, evaluate and
  save only epoch 12.
- Mini model-settings sweeps: 30 epochs, evaluate and save only epoch 30.
- `config/sweep_config.yaml`: small 18-setting grid for `sweep_small_3seeds`.
- `config/sweep_config_large.yaml`: expanded 64-setting grid for
  `sweep_large_1seed`.
- Canonical seeds: `123` for single-seed result sets and `123`, `456`, `789`
  for three-seed result sets.

The curated launcher supports these result sets:

```text
full_trainval_3seeds
full_trainval_1seed
sweep_small_3seeds
sweep_large_1seed
```

The script uses staged phases:

1. `train`: run all model trainings and persist each run in a manifest.
2. `join`: run `data_preparation.join_characteristic_metrics` for every
   completed training run.
3. `aggregate`: average target metrics across seeds for three-seed result sets.
4. `combine`: expose a direct single-run trainval output or concatenate
   single-seed sweep settings.

The `all` phase runs those steps in order. If training fails, joins and
aggregation do not start. If joining or aggregation fails, completed model
directories, checkpoints, eval CSVs, and the manifest remain on disk.

## Stable Identifiers

Seed aggregation groups within model/eval settings by the trajdata evaluation
index:

```text
data_idx
```

The group key also includes available model/eval settings so rows are averaged
only across seeds within the same setting:

```text
eval_data, history_sec, prediction_sec, restrict_to_predchal, attention_radius_m
```

As a guardrail, the script validates that these semantic trajectory identity
columns are constant within each `(setting, data_idx)` group before writing
averaged outputs:

```text
scene_path, agent_id, scene_ts, agent_type
```

If `data_idx` or any semantic identity check column is missing, or if semantic
identity varies within a `(setting, data_idx)` group, aggregation fails before
writing averaged outputs.

For seed-varying metrics (`ml_ade`, `ml_fde`, `min_ade_5`, `nll_mean`,
`nll_final`), the aggregate output stores the seed mean under the original
metric name and adds `*_seed_std`, `*_seed_min`, and `*_seed_max` columns.

## Run

Run from the repository root. The command below is intentionally autonomous:
wandb is disabled by default inside the script, logs are written under the
selected result-set root, and every completed or adopted training run is
recorded in `manifest.json`.

```bash
conda run -n adaptive-py310 python scripts/run_prediction_result_set.py \
  --experiment sweep_small_3seeds \
  --phase all
```

To resume a failed workflow, rerun the same command. Completed training and join
records are skipped unless `--force` is passed.

For a command preview:

```bash
conda run -n adaptive-py310 python scripts/run_prediction_result_set.py \
  --experiment sweep_large_1seed \
  --phase all \
  --dry_run
```

## Outputs

The manifest and per-command logs are written under stable result-set roots:

```text
results/trajectory_prediction/experiment_sets/<experiment>/manifest.json
results/trajectory_prediction/experiment_sets/<experiment>/logs/*.log
```

Raw model outputs remain in the normal training locations:

```text
results/trajectory_prediction/nuScenes/models/<trainval_run>/
results/trajectory_prediction/logs/<sweep_run>/
results/trajectory_prediction/trajectory_metrics/<run>/eval_epoch_<N>.csv
```

Joined per-seed outputs are written to:

```text
results/trajectory_prediction/trajectory_metrics_joined/<run>/eval_epoch_<N>.csv
```

Notebook-compatible outputs are written as:

```text
results/trajectory_prediction/trajectory_metrics_joined/full_trainval_12ep_3seeds/eval_epoch_12_seed_mean.csv
results/trajectory_prediction/trajectory_metrics_joined/full_trainval_12ep_1seed/eval_epoch_12.csv
results/trajectory_prediction/trajectory_metrics_joined/sweep_small_30ep_3seeds/eval_epoch_30_seed_mean.csv
results/trajectory_prediction/trajectory_metrics_joined/sweep_large_30ep_1seed/eval_epoch_30_combined.csv
```

Use the notebook-compatible `RUN_NAME` / `EVAL_CSV_NAME` values in
`src/data_modelling/interpretable_model_data_preparation.ipynb`.

The full curated index is written to:

```text
results/trajectory_prediction/experiment_sets/index.json
```
