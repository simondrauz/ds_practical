# Results Directory

This directory is the single root for generated artifacts.

## Directory Organization
```
results/
├── trajectory_prediction/
│   ├── nuScenes/                 # model checkpoints and run logs
│   ├── trajectory_metrics/        # raw per-epoch eval CSVs
│   └── trajectory_metrics_joined/ # joined evaluation CSV/Parquet outputs
├── evaluation/
│   ├── metrics/                   # analysis metric tables
│   ├── plots/                     # analysis/evaluation plots
│   └── reports/                   # report-style summaries
└── interpretable_model/           # reserved for future interpretable-model outputs
```

For the overall project structure, see the [main README](../README.md).
