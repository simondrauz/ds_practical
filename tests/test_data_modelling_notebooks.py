from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "src" / "data_modelling"
NOTEBOOK_NAMES = [
    "gam.ipynb",
    "interpretable_model_data_preparation.ipynb",
    "model_inference_analysis.ipynb",
    "oof_evaluation.ipynb",
    "shap_performance_regimes.ipynb",
    "xgboost.ipynb",
]
SECTION_MARKERS = (
    "**Purpose:** ",
    "**Inputs:** ",
    "**Outputs:** ",
    "**How to Verify:** ",
)


def test_data_modelling_notebooks_expose_workflow_structure():
    for notebook_name in NOTEBOOK_NAMES:
        notebook = json.loads((NOTEBOOK_DIR / notebook_name).read_text())
        markdown_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        ]

        assert markdown_cells, f"{notebook_name} should contain markdown cells."
        assert markdown_cells[0].startswith("# "), f"{notebook_name} should start with a title cell."

        section_cells = [cell for cell in markdown_cells if cell.startswith("## ")]
        assert section_cells, f"{notebook_name} should contain numbered workflow sections."

        for section in section_cells:
            for marker in SECTION_MARKERS:
                assert marker in section, f"{notebook_name} section is missing {marker}: {section.splitlines()[0]}"
            assert "**Purpose:** " in section and "<br>" in section, f"{notebook_name} should use inline <br> formatting."
            assert "**Inputs:** " in section and "<br>" in section, f"{notebook_name} should use inline <br> formatting."
            assert "**Outputs:** " in section and "<br>" in section, f"{notebook_name} should use inline <br> formatting."


def test_shap_performance_regimes_notebook_references_split_umap_configuration():
    notebook = json.loads((NOTEBOOK_DIR / "shap_performance_regimes.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "cluster_umap_n_neighbors" in source
    assert "cluster_umap_min_dist" in source
    assert "viz_umap_n_neighbors" in source
    assert "viz_umap_min_dist" in source
    assert "trustworthiness_neighbor_values" in source
    assert "UMAP_TRUSTWORTHINESS_PLOT_PATHS" in source
