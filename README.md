# Trajectory Prediction Performance Analysis for Federated Learning

## Project Overview
This repository explores how data characteristics drawn from the nuScenes dataset impact the performance of a Trajectron++ trajectory prediction model. The centralized analysis produced here will guide the design of a future federated learning (FL) setup for trajectory forecasting in diverse traffic scenarios.

## Getting Started
1. Ensure you have Python 3.9+ available in your environment.
2. Create and activate a virtual environment (example uses `venv`):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the project dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Acquire the nuScenes dataset (follow the nuScenes license and download instructions) and store paths under `data/` as needed.

## Project Workflow
- **Step 1 — Data Characterization:** Use the TrajData library to summarize nuScenes agent behaviors, scene context, and interaction cues.
- **Step 2 — Data Clustering:**
  - Manual clusters based on domain heuristics (agent class, interaction level, weather, etc.).
  - Algorithmic clusters derived from feature embeddings (e.g., K-Means on scene and motion descriptors).
- **Step 3 — Performance Evaluation:** Train and evaluate Trajectron++ on the full dataset and on each cluster to identify performance sensitivities.

## Repository Diagram
- `config/` — Configuration templates for experiments, clustering settings, and model hyperparameters.
- `data/` — Local storage hooks, metadata exports, and scripts for dataset ingestion (nuScenes files not tracked).
- `notebooks/` — Exploratory data analysis, clustering prototypes, and evaluation summaries.
- `results/` — Consolidated metrics, plots, and analysis artifacts.
- `src/` — Reusable Python modules for data pipelines, clustering utilities, and evaluation routines.

## Next Steps
- Fill in configuration templates in `config/` to document dataset paths and cluster definitions.
- Begin exploratory analysis in `notebooks/` to validate feature engineering choices.
- Implement reusable data loaders and evaluators under `src/` once the workflow is finalized.
