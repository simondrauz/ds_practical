# Results Directory

Archive experiment outputs, plots, and summary tables that capture Trajectron++ performance across data clusters.

## Suggested Organization
- `metrics/` serialized evaluation outputs (JSON, CSV, parquet).
- `plots/` figures illustrating performance trends and feature importance.
- `reports/` human-readable summaries combining qualitative insights with quantitative results.

## Next Steps
- Agree on a filename convention that encodes model version, data subset, and evaluation timestamp.
- Automate export scripts in `src/evaluation/` to populate this directory.
