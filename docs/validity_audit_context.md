# Validity Audit Context

Date: 2026-05-03

Branch used for fixes: `dev-fixes`

Baseline reviewed: `dev` after merge commit `d00b843` (`dev-interpretable-models` merged into `dev`)

Environment used for checks: conda env `adaptive-py310` with
`PYTHONPATH=src:unified-av-data-loader/src`

## Purpose

This document is a handoff note for the trajectory-prediction and
interpretable-model validity audit. It records the audit process, what has
already been fixed, and which validity issues are still open. Use it as session
context before continuing work on this branch.

The audit focused on issues that could invalidate, misalign, or make
non-reproducible reported trajectory-prediction and interpretable-model results:

- target leakage into model features or feature selection;
- row/index misalignment between eval CSVs, joined characteristic metrics,
  prepared modelling data, OOF predictions, feature-effect tables, and regime
  analysis notebooks;
- split and reconstruction mismatches between Trajectron++ eval and
  `join_characteristic_metrics.py`;
- sweep settings such as `attention_radius_m`, `history_sec`, and
  `prediction_sec`;
- stale or unrelated sweep outputs being combined;
- hard-coded local paths and stale notebook outputs that could mislead later
  interpretation.

## Audit Process

The audit was treated as a code/data-pipeline review, not a general cleanup.
The audit initially used lightweight commands only. It now also includes a
capped integration harness that exercises the real mini and trainval nuScenes
paths long enough to validate the pipeline contracts without running full
training or full sweeps. The following commands were used during investigation
and fix validation:

```bash
git status --short --branch
PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python -m pytest tests
PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python run_sweep.py --dry_run
PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python -m data_preparation.join_characteristic_metrics --help
PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python -m data_preparation.combine_runs --help
PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python scripts/validate_pipeline_paths.py --skip-gates
git diff --check
```

During the review we inspected:

- `train_unified.py`, especially the per-trajectory eval CSV writer;
- `src/data_preparation/join_characteristic_metrics.py`;
- `run_sweep.py`;
- `src/data_preparation/combine_runs.py`;
- `src/data_modelling/prepared_data.py`;
- `src/data_modelling/feature_effect_performance_regimes_utils.py`;
- modelling notebooks under `src/data_modelling/`;
- tests under `tests/`.

Latest capped integration validation:

- Summary:
  `results/interpretable_model/notebook_runs/pipeline_validation_20260503_100713/validation_summary.json`
- Full trainval path:
  `train_unified.py` on `nusc_trainval-train`/`nusc_trainval-train_val`
  with capped train/eval batches, then join, preparation, GAM, XGBoost, model
  inference, performance-regime clustering, and cluster inspection.
- Mini sweep path:
  eight capped mini runs over
  `history_sec=[2.0, 4.0]`, `prediction_sec=[2.0, 6.0]`, and two scaled
  attention-radius settings, then scoped combine, preparation, GAM, XGBoost,
  and model inference only.
- The combined sweep dataframe intentionally tracks realized model settings
  only: `attention_radius_m`, `history_sec`, and `prediction_sec`.
  `attention_radius_scale` remains an internal sweep multiplier and is not a
  dataframe/model feature.
- `config/shared_config.yaml` was restored byte-for-byte after sweep
  validation.

## Fixed Issues

### Bug #1: Downstream interpretable-model row identity misalignment

Status: fixed and committed in `abc215b` (`Fix interpretable model row identity alignment`).

Problem:

The prepared modelling CSVs and downstream OOF/feature-effect artifacts dropped
the stable eval identity columns and later treated the reloaded CSV row number
as `row_id`. Regime analysis then joined prepared/OOF rows back to joined
trajectory metrics with `row_id == data_idx`. That only works if CSV row order
happens to match trajdata `data_idx` order. In real joined outputs, eval rows
can be shuffled relative to `data_idx`, which could silently attach OOF
predictions, feature effects, performance groups, scene context, and trajectory
characteristics to the wrong trajectory.

Fix:

- Preserved identity metadata through prepared data and model artifacts:
  `run_name`, `eval_csv_name`, and `data_idx`.
- Kept identity metadata out of model feature matrices.
- Updated preparation, GAM, and XGBoost notebooks to carry identity columns.
- Updated feature-effect exports to include identity metadata when available.
- Updated regime assembly to merge prepared rows, joined metrics, and
  feature-effect tables on stable trajectory keys instead of row position.
- Required `data_idx` for non-legacy regime assembly so stale artifacts fail
  loudly instead of silently misaligning.
- Added regression tests for shuffled `data_idx`, duplicate `data_idx` across
  runs, identity preservation, and legacy-artifact rejection.

Important follow-up:

Prepared data, OOF predictions, feature-effect exports, and regime-analysis
outputs generated before this commit should be considered stale and regenerated
before interpretation.

### Eval identity enforcement for trajdata `data_idx` joins

Status: fixed and committed in `6c01dbd`
(`Validate trajdata eval identity before metric joins`).

Problem:

The intended design is that Trajectron++ eval losses and trajectory/scene
characteristics are joined by a trajdata-derived per-trajectory `data_idx`.
This is valid only if `join_characteristic_metrics.py` reconstructs exactly the
same eval dataset used by `train_unified.py`. Before this fix, the join checked
only that `data_idx` was in range. A different split, temporal window,
prediction-challenge filter, cache/data root, or dataset ordering could still
produce a syntactically valid but semantically wrong joined CSV.

Fix:

- `train_unified.py` now writes stronger eval-row identity columns:
  `data_idx`, `scene_path`, `agent_id`, `scene_ts`, and `agent_type`.
- It also writes eval-context columns:
  `eval_data`, `history_sec`, `prediction_sec`, and `restrict_to_predchal`.
- `join_characteristic_metrics.py` still uses `data_idx` as the lookup key into
  the reconstructed trajdata dataset, but validates that the reconstructed row
  matches the stored scene, agent, timestep, agent type, and eval context.
- Partial identity/context columns fail loudly to avoid ambiguous suffixes or
  partial validation.
- Legacy eval CSVs with only `data_idx` remain supported, but cannot receive the
  stronger reconstruction guarantee.
- Added tests for identity matching/mismatch, context mismatch, legacy
  `data_idx`-only CSVs, partial identity/context columns, and overlap pruning.

Important follow-up:

New results should be generated with the updated eval writer. Legacy eval CSVs
can still be joined, but their reconstruction is not fully enforceable.

### Bug #3: Attention-radius sweep settings were not reproducible

Status: fixed and committed in `48b1ed2`
(`Persist attention radii in run configs`).

Problem:

Training and characteristic joining both depended on the live mutable
`config/shared_config.yaml` attention-radius map. The sweep runner temporarily
mutates this YAML for each run and restores it afterward. Because the saved run
`config.json` did not persist the actual scaled radius map, later rejoining or
reproducing a run could silently use whatever radii were currently in
`shared_config.yaml`, not the radii used during evaluation.

Fix:

- Added serialisable attention-radius helpers in
  `src/shared_config/config_loader.py`.
- `train_unified.py` now persists a JSON-compatible `attention_radius` block in
  the saved run config when missing.
- Training builds train/eval datasets from the run-scoped persisted
  `attention_radius` rather than re-reading mutable shared YAML.
- `join_characteristic_metrics.py` prefers the run-scoped `attention_radius`
  from `config.json`, falling back to shared YAML only for legacy configs.
- The joined `attention_radius_m` column now reflects the run-scoped radius
  used to compute characteristics.
- Added tests for canonical serialisation, YAML round-tripping, tuple-key
  reconstruction, persisted-radius preference, and legacy fallback behavior.

Important follow-up:

Runs produced before this fix may lack persisted attention radii. Rejoining
legacy runs still works via fallback, but exact sweep-setting reproduction is
not guaranteed unless the intended shared YAML state is known.

### Bug #2: Sweep combination mixed stale or unrelated joined runs

Status: fixed and committed in `881d345`
(`Scope sweep combines to current run outputs`).

Files changed for this fix:

- `run_sweep.py`
- `src/data_preparation/combine_runs.py`
- `tests/test_run_sweep.py` (new)

Problem:

`run_sweep.py` ran `combine_runs.py` without `--run_dirs`. The default behavior
of `combine_runs.py` was to include every subdirectory under
`results/trajectory_prediction/trajectory_metrics_joined`. If old mini,
trainval, debug, or unrelated sweep outputs were present, the combined dataset
could silently mix results from different experiments and make downstream
model-setting interpretation invalid.

Fix:

- `run_combination()` now returns the joined run directory name for successful
  non-dry-run runs.
- `run_sweep()` records joined run names produced by the current sweep.
- The final combine command is built with explicit `--run_dirs` for those
  current-sweep runs only.
- Dry-run output now shows explicit current-sweep placeholders after
  `--run_dirs`, rather than advertising an unsafe all-runs combine.
- `combine_runs.py` now refuses an implicit all-runs combine. Users must pass
  either `--run_dirs ...` or explicit `--all_runs`.
- Added tests for safe combine-command construction and CLI scoping behavior.

Validation run after this fix:

```bash
PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python -m pytest tests
# 69 passed

PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python run_sweep.py --dry_run
# combine command includes --run_dirs <current_sweep_run_1> ...

PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python -m data_preparation.combine_runs
# exits with an argparse error unless --run_dirs or --all_runs is supplied

git diff --check
# passed
```

### Validation coverage for the two nuScenes pipeline paths

Status: fixed and committed in `9e90189`
(`Validate nuScenes pipeline paths`).

Problem:

The repository had two intended workflows with different endpoints, but there
was no unattended validation that exercised both against the real nuScenes
dataset construction and downstream notebooks:

- full trainval path: Trajectron++ -> joined metrics -> preparation ->
  GAM/XGBoost -> model inference -> performance-regime clustering -> cluster
  inspection;
- mini sweep path: sweep Trajectron++ settings on mini -> scoped joined/combined
  metrics -> preparation -> GAM/XGBoost -> model inference only.

Without a capped integration check, regressions in dataset reconstruction,
notebook contracts, stale sweep combines, shared-config restoration, or
cluster-inspection artifact selection could remain hidden until an expensive
full run.

Fix:

- Added validation-only Trajectron++ caps:
  `--max_train_batches` and `--max_eval_batches`. They default to uncapped
  behavior and are used only when explicitly passed.
- Added `scripts/validate_pipeline_paths.py` as an unattended validation
  harness.
- The harness preflights imports, conda environment, mini/trainval raw data and
  caches, split files, and actual trajdata dataset construction. It samples the
  first/middle/last record from each required split.
- For the full trainval path, it runs one capped train/eval job on
  `nusc_trainval-train` and `nusc_trainval-train_val`, joins with vector-map
  metrics, asserts eval/joined row and column contracts, executes preparation,
  GAM, XGBoost, model inference, clustering, and cluster inspection for both
  models.
- For the mini sweep path, it writes a temporary local sweep config instead of
  relying on the committed example paths in `config/sweep_config.yaml`, runs eight
  capped mini combinations, scopes combine to the fresh joined dirs, bridges the
  combined dataframe into the existing preparation notebook contract, and stops
  after model inference.
- The sweep validation asserts the current analysis contract through realized
  settings: `attention_radius_m`, `history_sec`, and `prediction_sec`. The
  `attention_radius_scale` multiplier is used only to produce scaled runs; it is
  not propagated as a model feature.
- The harness tracks and restores `config/shared_config.yaml` even if a sweep
  run fails.
- Added the notebook/model dependencies needed by the validation path.
- Added a regression test for full-path performance-regime assembly when joined
  metrics lack non-key identity columns such as `run_name` and `eval_csv_name`.

Latest validation after this fix:

```bash
PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python run_sweep.py --dry_run
# passed

PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python scripts/validate_pipeline_paths.py --skip-gates
# passed; summary at results/interpretable_model/notebook_runs/pipeline_validation_20260503_100713/validation_summary.json

PYTHONPATH=src:unified-av-data-loader/src conda run -n adaptive-py310 python -m pytest tests
# 72 passed, 1 warning

git diff --check
# passed
```

## Open Issues Not Addressed Yet

### Bug #4: MI-elbow feature selection is target-informed outside resampling

Status: open, intentionally not addressed now.

Current behavior:

`src/data_modelling/interpretable_model_data_preparation.ipynb` computes mutual
information between candidate features and the target over the full prepared
dataset, detects an MI elbow, and exports only the selected features. The
modelling notebooks then run nested CV on that already-selected feature set.

Why this threatens validity:

If reported model performance is interpreted as out-of-sample performance of
the full modelling procedure, the feature-selection step has already seen the
target values from all outer-fold validation rows. That is target leakage
through supervised feature selection. It can make nested-CV performance
optimistic because the outer validation fold did not remain fully untouched by
the modelling procedure.

Model-setting contract:

The model-setting columns `attention_radius_m`, `history_sec`, and
`prediction_sec` are exported alongside selected trajectory/scene features and
are then picked up as numeric model features by the modelling helpers. This is
now the intended contract for the model-settings sweep analysis: the sweep asks
how prediction performance varies with these realized settings. The
`attention_radius_scale` multiplier is not part of this contract; it is only the
internal sweep knob used to produce different realized `attention_radius_m`
values.

What should be changed or explicitly verified:

- Decide whether MI-elbow selection is a fixed exploratory preprocessing step
  or part of the predictive model-selection procedure.
- For unbiased performance claims, move supervised feature selection inside the
  training folds, ideally as part of the nested-CV pipeline.
- If feature selection remains global for interpretability, report performance
  as conditional on that target-informed feature set and avoid presenting it as
  unbiased full-pipeline generalization.
- Keep tests/helpers that assert target and identity columns are excluded from
  features, and that realized model-setting columns are included when the
  settings-sweep contract is intended.
- Regenerate prepared data and model artifacts after changing the contract.

### Bug #5: Notebooks contain stale outputs and local run/path state

Status: partially addressed. The top-level `README.md` now documents the two
supported paths, local sweep-config handling, and the
`dev-model-setting-inclusion` migration notes. The remaining notebook output
and committed local-path state is still open.

Current behavior:

Several modelling notebooks contain rendered outputs with absolute local paths,
specific old run names, and stale result summaries. Examples observed during
the audit include mini/debug run names, trainval/debug run names, absolute
`/Users/...` result paths, and old exported plot/table paths in notebook output
cells. `config/sweep_config.yaml` now uses repo-local example paths instead of
one machine's absolute cache/data paths. The README still tells users to copy
the sweep config to an ignored local path when their data layout differs, and
now states that `--user` fallback paths do not override path keys already
defined by a selected run config.

Why this threatens validity and reproducibility:

Rendered notebook outputs can be mistaken for current branch results even when
they were produced before the validity fixes above. Hard-coded local paths make
the pipeline non-reproducible across users and can accidentally point a later
session at the wrong raw data, cache, joined metrics, or result directory.

What should be changed or explicitly verified:

- Clear or strip notebook outputs before relying on notebooks as source
  artifacts.
- Parameterize run names, data roots, cache roots, and result roots rather than
  embedding user-specific absolute paths.
- Keep the README as the source of current path-level usage guidance; it now
  distinguishes full trainval from mini sweep, documents path precedence across
  CLI/config/`--user` fallback values, and states that the sweep stops at model
  inference.
- Treat existing rendered notebook outputs as stale until the notebooks are
  rerun from fixed inputs.
- Add a lightweight check that fails on newly committed notebook outputs or
  absolute `/Users/...` paths in source cells.
- Move local path examples into ignored local config files or documented
  templates.
- Regenerate all analysis outputs after the fixed pipeline has produced new
  eval, joined, prepared, OOF, and feature-effect artifacts.

## Residual Risks After Current Fixes

- Full uncapped trainval training and full uncapped sweeps were not rerun during
  this audit. The capped validation harness exercises the real pipeline paths
  and notebook contracts, but it is not a substitute for final full runs.
- Legacy eval CSVs without the new identity/context columns cannot prove that
  `join_characteristic_metrics.py` reconstructed the exact same eval dataset.
- Existing generated artifacts under `results/` may predate the fixes and
  should not be used for final interpretation unless their provenance is known.
- Notebook outputs may display stale paths/results even when source code cells
  have been updated.
- Tests now cover several alignment and reproducibility risks, and the capped
  validation harness covers the real nuScenes path. There is still no tiny
  deterministic fixture that runs eval -> join -> prepare -> OOF/export without
  requiring the local nuScenes data/cache.

## Recommended Next Steps

1. Decide and implement the intended contract for MI-elbow feature selection.
2. Strip or regenerate stale notebook outputs and remove local absolute paths
   from committed notebook state.
3. Run the full uncapped trainval path and full intended sweep path, then
   regenerate current eval CSVs, joined metrics, prepared data, OOF predictions,
   feature-effect tables, and regime outputs with the fixed pipeline.
4. Keep using `scripts/validate_pipeline_paths.py` before expensive full runs,
   especially after changes to Trajectron++ eval writing, metric joining,
   sweep combination, modelling notebooks, or regime/inspection utilities.
5. Add a tiny deterministic end-to-end fixture if local-data-independent CI
   coverage is needed.
