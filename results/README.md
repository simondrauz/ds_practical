# Results Directory

Archive all analysis outputs, visualizations, and evaluation metrics produced throughout the centralized Trajectron++ study.

## Directory Organization
```
results/
├── plots/
│   ├── characterization/    # Data exploration and feature analysis
│   ├── clustering/           # Cluster validation and comparison plots
│   └── evaluation/           # Model performance across clusters
├── metrics/                  # Serialized evaluation outputs (JSON, CSV, parquet)
└── reports/                  # Human-readable summaries and findings
```

## Content Guidelines
- **plots/characterization/**: Agent distributions, trajectory statistics, scene complexity, interaction patterns
- **plots/clustering/**: Cluster sizes, feature importance, silhouette scores, manual vs. algorithmic comparisons
- **plots/evaluation/**: Trajectron++ ADE/FDE curves, per-cluster performance, failure case visualizations
- **metrics/**: Structured data exports for reproducibility and downstream analysis
- **reports/**: Markdown/PDF summaries combining qualitative insights with quantitative results

## Next Steps
- Agree on a filename convention that encodes analysis phase, data subset, and timestamp
- Automate export scripts in `src/` to populate this directory consistently
