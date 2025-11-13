# Source Code Directory

Implement reusable Python modules supporting the centralized Trajectron++ analysis pipeline.

## Suggested Structure
- `data/` loaders and dataset abstractions that wrap TrajData and nuScenes assets.
- `features/` feature engineering utilities for agent dynamics and scene context.
- `clustering/` manual split definitions and algorithmic clustering drivers.
- `evaluation/` scripts to train/evaluate Trajectron++ and aggregate metrics.
- `utils/` common helpers (logging, configuration management, reproducibility aids).

## Next Steps
- Define a configuration schema under `config/` and mirror it with parsing utilities here.
- Implement deterministic data splits to ensure reproducibility across experiments.
- Add unit tests (future `tests/` directory) once core modules solidify.
