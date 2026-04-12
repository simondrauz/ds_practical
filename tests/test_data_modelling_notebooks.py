from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "src" / "data_modelling"
NOTEBOOK_NAMES = [
    "gam.ipynb",
    "feature_effect_performance_regimes.ipynb",
    "interpretable_model_data_preparation.ipynb",
    "model_inference_analysis.ipynb",
    "oof_evaluation.ipynb",
    "feature_effect_pr_cluster_inspection.ipynb",
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


def test_feature_effect_performance_regimes_notebook_references_export_first_cluster_workflow():
    notebook = json.loads((NOTEBOOK_DIR / "feature_effect_performance_regimes.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "cluster_umap_n_neighbors" in source
    assert "cluster_umap_min_dist" in source
    assert "viz_umap_n_neighbors" in source
    assert "viz_umap_min_dist" in source
    assert "trustworthiness_neighbor_values" in source
    assert "UMAP_TRUSTWORTHINESS_PLOT_PATHS" in source
    assert "resolve_feature_effect_regime_export_context" in source
    assert "build_feature_effect_regime_export_layout" in source
    assert "build_feature_effect_regime_artifact_names" in source
    assert "write_cluster_exports" in source
    assert "cluster_feature_effect_profiles" in source
    assert "feature_effect_global_ranking" in source
    assert "cluster_catalog" in source
    assert "load_or_initialize_feature_effect_regime_manifest" in source
    assert "merge_feature_effect_regime_artifact_records" in source
    assert "feature_effect_performance_regimes" in source
    assert "manifest.json" in source
    assert "INSPECTION_CONFIG" not in source
    assert "selected_cluster" not in source


def test_feature_effect_pr_cluster_inspection_notebook_references_exported_cluster_artifacts():
    notebook = json.loads((NOTEBOOK_DIR / "feature_effect_pr_cluster_inspection.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "INSPECTION_CONFIG" in source
    assert "MODEL_ID" in source
    assert "RUN_NAME" in source
    assert "EVAL_CSV_NAME" in source
    assert "CLUSTER_SPEC_DIRNAME" in source
    assert "load_run_context" in source
    assert "resolve_feature_effect_regime_export_context" in source
    assert "default_cluster_spec_manifest_path" in source
    assert "cluster_spec_manifest_path" in source
    assert "cluster_ids" in source
    assert "distribution_matrix_max_columns" in source
    assert "resolve_cluster_inspection_config" in source
    assert "load_cluster_inspection_selection" in source
    assert "build_subset_style_map" in source
    assert "build_cluster_inspection_export_layout" in source
    assert "plot_candidate_umap_scatter" in source
    assert "plot_cluster_profile_barplots" in source
    assert "plot_cluster_profile_heatmap" in source
    assert "plot_metric_distribution_panels" in source
    assert "plot_metric_overview_matrix_pages" in source
    assert "target_orig" in source
