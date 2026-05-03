# Configuration Directory

This directory contains configuration files for reproducible experiments, including setup assets under `experimental_setup/`.

## Shared Runtime Settings

`shared_config.yaml` centralizes trajdata shared runtime settings used across:
- `train_unified.py`
- `src/data_preparation/join_characteristic_metrics.py`
- `notebooks/02_agent_centered_trajectory_metrics_analysis.ipynb`
- `notebooks/03_scene_centered_trajectory_metrics_analysis.ipynb`

`shared_config.yaml` currently contains:
- `vector_map.raster_map_params`
- `agent_type_defaults` (`only_predict`, `no_types`)
- `attention_radius`

`analysis_config.yaml` stores the vector-map inclusion default (`trajdata.incl_vector_map`) for notebooks and joined-metrics analysis.

## Training Runtime Configs (`--conf`)

Training/eval hyperparameters for `train_unified.py` are read from a JSON file passed via `--conf`.

Common files in this folder:
- `nuScenes.json`: baseline Trajectron++ config, kept aligned with the upstream
  NVlabs nuScenes model hyperparameters and free of project runtime keys.
- `nuScenes_full_trainval.json`: full nuScenes trainval run config. It extends
  `nuScenes.json` and adds the trainval split/path/runtime overrides.
- `nuScenes_mini.json`: nuScenes mini run config. It extends `nuScenes.json`
  and adds the mini split/path/runtime overrides.
- `runtime_config.json`: backwards-compatible alias for `nuScenes_mini.json`.

Run config JSON files can use an `extends` key. Relative parent paths are
resolved from the child config file, nested mappings are merged recursively, and
child values override parent values.

### Resolution Order

At runtime, values are resolved in this order:
1. Explicit CLI flags (highest priority).
2. Values from the resolved JSON config file passed to `--conf`.
3. Argparse defaults (only for keys missing from config).

"Explicit CLI flag" means it is present in the command (`--key value` or `--key=value`).
If `--conf` is omitted, parser default is `config/nuScenes_mini.json`.

Use `--eval_only_predict PEDESTRIAN` when training should keep the configured
target mix but evaluation/prediction output should be restricted to Pedestrian
trajectories.

### Example Commands

Config-driven mini run:
```bash
torchrun --nproc_per_node=1 train_unified.py \
  --user simon \
  --conf config/nuScenes_mini.json
```

Config-driven full trainval run:
```bash
torchrun --nproc_per_node=1 train_unified.py \
  --user simon \
  --conf config/nuScenes_full_trainval.json
```

Override one config value explicitly:
```bash
torchrun --nproc_per_node=1 train_unified.py \
  --user simon \
  --conf config/nuScenes_mini.json \
  --batch_size 128
```

## Purpose

Centralize all experiment parameters, paths, and hyperparameters to ensure:
- **Reproducibility**: Same config → same results
- **Collaboration**: Team members use consistent settings
- **Version Control**: Track configuration changes alongside code

## Example How Configuration Files May Look

### `base_config.yaml`
Core settings used across all experiments:
```yaml
# Data paths
data:
  nuscenes_root: "../data/raw"
  processed_dir: "../data/processed"
  metadata_dir: "../data/metadata"

# trajdata settings
trajdata:
  dt: 0.1
  history_sec: 3.2
  future_sec: 4.8
  raster_px_per_m: 2
  raster_size_px: 224

# Random seeds for reproducibility
random_seed:
  numpy: 42
  torch: 42
  python: 42

# Results output
results:
  plots_dir: "../results/evaluation/plots"
  metrics_dir: "../results/evaluation/metrics"
  reports_dir: "../results/evaluation/reports"
```

### `clustering_config.yaml` (Step 2)
Clustering strategy and parameters:
```yaml
# Clustering approach
method: "manual"  # Options: manual, kmeans, hierarchical, dbscan

# Manual cluster definitions (if method=manual)
manual_clusters:
  cluster_0:
    name: "Boston Urban"
    location: ["boston-seaport"]
    agent_types: ["vehicle"]
  cluster_1:
    name: "Singapore Urban"
    location: ["singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"]
    agent_types: ["vehicle"]
  # Add more clusters as determined by Step 1 analysis

# Automated clustering parameters (if method != manual)
automated:
  n_clusters: 4
  features: ["location", "agent_type", "scene_type"]
  # Method-specific params...
```

### `trajectron_config.yaml` (Step 3)
Trajectron++ model configuration:
```yaml
# Model source
model:
  source: "pretrained"  # Options: pretrained, trained_from_scratch, finetuned
  checkpoint_path: null  # Path to weights if using pretrained/finetuned

# Evaluation settings
evaluation:
  prediction_horizon: 4.8  # seconds (must match trajdata future_sec)
  num_samples: 12  # Number of trajectory samples for multi-modal prediction
  batch_size: 32

# Metrics to compute
metrics:
  - "ADE"  # Average Displacement Error
  - "FDE"  # Final Displacement Error
  - "collision_rate"
  - "off_road_rate"

# Per-cluster evaluation
cluster_evaluation:
  enabled: true
  cluster_file: "../data/metadata/cluster_assignments.json"
```

For the overall project structure, see the [main README](../README.md).
