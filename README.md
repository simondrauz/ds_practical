# Trajectory Prediction Performance Analysis for Federated Learning

## Project Overview
This repository explores how data characteristics drawn from the nuScenes dataset impact the performance of a Trajectron++ trajectory prediction model. The centralized analysis produced here will guide the design of a future federated learning (FL) setup for trajectory forecasting in diverse traffic scenarios.

## Getting Started

### Prerequisites
- Python 3.10+ (tested with 3.10.16; see `runtime.txt`)
- ~500 MB free disk space for nuScenes mini dataset + map expansion
- Git for version control

### Environment Setup
1. Clone this repository and navigate to the project root:
   ```bash
   git clone https://github.com/simondrauz/ds_practical.git
   cd ds_practical
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install project dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional)** Register a Jupyter kernel if not using VS Code:
   ```bash
   python -m ipykernel install --user --name trajectory-fl --display-name "trajectory-fl"
   ```
   **Note:** VS Code auto-detects `.venv` as a kernel—just select `.venv (Python 3.10.16)` from the kernel picker in notebooks.

### Dataset Acquisition
1. Create a nuScenes account at https://www.nuscenes.org/download and accept the license
2. Download the **v1.0-mini** dataset and the **Map expansion pack (v1.3)**
3. Extract both archives and organize as follows:
   ```
   data/raw/
   ├── maps/
   │   ├── expansion/
   │   │   ├── boston-seaport.json
   │   │   ├── singapore-hollandvillage.json
   │   │   ├── singapore-onenorth.json
   │   │   └── singapore-queenstown.json
   │   └── (map image files)
   ├── samples/
   ├── sweeps/
   └── v1.0-mini/
   ```
4. Open `notebooks/00_setup_validation.ipynb` and run all cells to verify the setup

## Project Workflow
- **Step 0 — Setup Validation:** Run `notebooks/00_setup_validation.ipynb` to verify dataset access and TrajData functionality.
- **Step 1 — Data Characterization:** Use the TrajData library to summarize nuScenes agent behaviors, scene context, and interaction cues (upcoming notebook).
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
