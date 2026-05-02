# Current Pipeline Overview Notes

This note accompanies the two pipeline graphics:

- `docs/current_pipeline_overview.svg`: compact assisting version for use while explaining verbally.
- `docs/current_pipeline_overview_by_itself.svg`: fuller standalone version that can be read without much verbal context.

## Regenerating SVG/PNG/PDF outputs

Run the export script from the repository root:

```bash
python3 docs/export_pipeline_views.py
```

This recreates the SVG files from the script and then creates regular PNG and
single-page PDF exports next to them:

- `docs/current_pipeline_overview.svg`
- `docs/current_pipeline_overview.png`
- `docs/current_pipeline_overview.pdf`
- `docs/current_pipeline_overview_by_itself.svg`
- `docs/current_pipeline_overview_by_itself.png`
- `docs/current_pipeline_overview_by_itself.pdf`

Useful variants:

```bash
python3 docs/export_pipeline_views.py --format svg
python3 docs/export_pipeline_views.py --format png
python3 docs/export_pipeline_views.py --format pdf
python3 docs/export_pipeline_views.py --view by_itself
```

The script uses a local Chrome/Chromium executable for PNG/PDF export. SVG-only
generation does not require Chrome. If Chrome is not found automatically, pass
`--chrome-bin /path/to/chrome`.

## Source refs consulted

- Core analysis branch: `dev-interpretable-model-full-trainval`
- Model-settings extension branch: `origin/dev-model-settings-inclusion`

The user mentioned `dev-interpretable-model-full-train`; the local repository contains `dev-interpretable-model-full-trainval`, which appears to be the matching branch for the full train/validation pipeline.

## Visual conventions

- Neutral boxes: the main pipeline steps and standard-path outputs.
- Blue shared block: scripts/mechanisms used by both the standard path and the settings-sweep path.
- Light grey boxes: data/artifact states rather than notebooks/scripts.
- Purple boxes/arrows: settings-specific inputs and processing, especially retaining settings and combining runs.
- Dashed styling: planned additions, currently multicollinearity handling and ALE plots.
- Reading order: section A runs left-to-right; section B continues right-to-left.

## Branch observations reflected in the diagram

- `train_unified.py` writes per-trajectory evaluation rows to `results/trajectory_prediction/trajectory_metrics/<run>/eval_epoch_*.csv`, keyed by `data_idx`.
- `src/data_preparation/join_characteristic_metrics.py` rebuilds the eval-aligned trajdata dataset, computes trajectory/scene characteristics, and left-joins them to the eval CSV by `data_idx`.
- `src/data_modelling/interpretable_model_data_preparation.ipynb` prepares the joined table for modelling through distribution checks, target-transform decisions, and feature selection.
- `src/data_modelling/xgboost.ipynb` and `src/data_modelling/gam.ipynb` both use nested resampling for tuning-procedure validation, then retune/refit on all data and export manifests for downstream analysis.
- `src/data_modelling/model_inference_analysis.ipynb` loads a manifest-backed model and exports feature-effect tables/rankings for XGBoost SHAP or GAM additive effects.
- `src/data_modelling/feature_effect_performance_regimes.ipynb` splits trajectories by model performance, evaluates raw/reduced representations, scores clustering with DBCV, and exports regime candidates.
- `src/data_modelling/feature_effect_pr_cluster_inspection.ipynb` inspects selected cluster candidates through feature values and signed effect/contribution values.
- `origin/dev-model-settings-inclusion` adds `run_sweep.py`, `config/sweep_config.yaml`, and `src/data_preparation/combine_runs.py` for the settings sweep over `history_sec`, `prediction_sec`, and attention-radius scaling/`attention_radius_m`.
- The settings-sweep analysis is shown as ending at `model_inference_analysis.ipynb`, where setting effects are interpreted while trajectory and scene characteristics act as controls.
