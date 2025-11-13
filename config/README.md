# Configuration Directory

This directory contains configuration files for reproducible experiments.

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
  plots_dir: "../results/plots"
  metrics_dir: "../results/metrics"
  reports_dir: "../results/reports"
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

