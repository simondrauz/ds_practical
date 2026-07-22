"""Run and curate the four supported trajectory-prediction result sets.

The script is intentionally a thin orchestration layer around the existing
training, joining, seed aggregation, and sweep-combination primitives. It gives
the downstream notebooks stable result-set names without moving or deleting raw
model artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_seeded_experiments import (
    DEFAULT_JOINED_ROOT,
    DEFAULT_METRICS_ROOT,
    DEFAULT_SWEEP_CONFIG,
    DEFAULT_SWEEP_LOG_DIR,
    ROOT,
    SHARED_CONFIG_PATH,
    _build_combinations,
    _json_default,
    _latest_new_subdir,
    _load_config_log_dir,
    _load_json,
    _load_yaml,
    _repo_path,
    _run_command,
    _run_key,
    _safe_float_token,
    _scale_attention_radii,
    _subprocess_env,
    _timestamp,
    _train_command,
    _write_json,
    _write_yaml,
    aggregate_seeded_records,
)

SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from data_preparation.combine_runs import combine as combine_joined_runs  # noqa: E402


EXPERIMENT_ROOT = ROOT / "results" / "trajectory_prediction" / "experiment_sets"
INDEX_PATH = EXPERIMENT_ROOT / "index.json"
MAIN_3SEED_MANIFEST = (
    ROOT
    / "results"
    / "trajectory_prediction"
    / "seeded_experiments"
    / "main_3seed"
    / "manifest.json"
)
LARGE_SWEEP_CONFIG = ROOT / "config" / "sweep_config_large.yaml"
TRAINVAL_CONF = ROOT / "config" / "nuScenes_full_trainval.json"
DEFAULT_PYTHON = sys.executable
SCAN_RELATIVE_ROOTS = [
    Path("logs"),
    Path("nuScenes") / "models",
    Path("trajectory_metrics"),
    Path("trajectory_metrics_joined"),
    Path("seeded_experiments"),
    Path("overnight_runs"),
]

TRAINED_STATUSES = {"trained", "joined", "adopted", "joined_adopted"}
JOINED_STATUSES = {"joined", "joined_adopted"}


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    kind: str
    seeds: tuple[int, ...]
    checkpoint_epoch: int
    notebook_run_name: str
    notebook_eval_csv_name: str
    sweep_config: Path | None = None
    aggregate_across_seeds: bool = False
    combine_sweep_runs: bool = False

    @property
    def output_root(self) -> Path:
        return EXPERIMENT_ROOT / self.experiment_id

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "manifest.json"

    @property
    def notebook_output_path(self) -> Path:
        return DEFAULT_JOINED_ROOT / self.notebook_run_name / self.notebook_eval_csv_name


EXPERIMENTS: dict[str, ExperimentDefinition] = {
    "full_trainval_3seeds": ExperimentDefinition(
        experiment_id="full_trainval_3seeds",
        kind="trainval",
        seeds=(123, 456, 789),
        checkpoint_epoch=12,
        aggregate_across_seeds=True,
        notebook_run_name="full_trainval_12ep_3seeds",
        notebook_eval_csv_name="eval_epoch_12_seed_mean.csv",
    ),
    "full_trainval_1seed": ExperimentDefinition(
        experiment_id="full_trainval_1seed",
        kind="trainval",
        seeds=(123,),
        checkpoint_epoch=12,
        notebook_run_name="full_trainval_12ep_1seed",
        notebook_eval_csv_name="eval_epoch_12.csv",
    ),
    "sweep_small_3seeds": ExperimentDefinition(
        experiment_id="sweep_small_3seeds",
        kind="sweep",
        seeds=(123, 456, 789),
        checkpoint_epoch=30,
        aggregate_across_seeds=True,
        sweep_config=DEFAULT_SWEEP_CONFIG,
        notebook_run_name="sweep_small_30ep_3seeds",
        notebook_eval_csv_name="eval_epoch_30_seed_mean.csv",
    ),
    "sweep_large_1seed": ExperimentDefinition(
        experiment_id="sweep_large_1seed",
        kind="sweep",
        seeds=(123,),
        checkpoint_epoch=30,
        combine_sweep_runs=True,
        sweep_config=LARGE_SWEEP_CONFIG,
        notebook_run_name="sweep_large_30ep_1seed",
        notebook_eval_csv_name="eval_epoch_30_combined.csv",
    ),
}


def get_experiment_definition(experiment_id: str) -> ExperimentDefinition:
    try:
        return EXPERIMENTS[experiment_id]
    except KeyError as exc:
        raise KeyError(f"Unknown experiment: {experiment_id}") from exc


def _sweep_cfg(definition: ExperimentDefinition) -> dict[str, Any]:
    if definition.sweep_config is None:
        return {}
    return _load_yaml(definition.sweep_config)


def experiment_grid(definition: ExperimentDefinition) -> list[dict[str, Any]]:
    if definition.kind != "sweep":
        return []
    return _build_combinations(_sweep_cfg(definition).get("grid", {}))


def _record_key_for_trainval(seed: int) -> str:
    return _run_key("seed", seed)


def _record_key_for_sweep(seed: int, combo: dict[str, Any]) -> str:
    return _run_key(
        "seed",
        seed,
        "history",
        float(combo["history_sec"]),
        "prediction",
        float(combo["prediction_sec"]),
        "radius_scale",
        float(combo["attention_radius_scale"]),
    )


def expected_training_keys(definition: ExperimentDefinition) -> set[str]:
    if definition.kind == "trainval":
        return {_record_key_for_trainval(seed) for seed in definition.seeds}

    keys = set()
    for seed in definition.seeds:
        for combo in experiment_grid(definition):
            keys.add(_record_key_for_sweep(seed, combo))
    return keys


def _base_manifest(definition: ExperimentDefinition) -> dict[str, Any]:
    sweep_cfg = _sweep_cfg(definition)
    return {
        "created_at": _timestamp(),
        "experiment": definition.experiment_id,
        "kind": definition.kind,
        "seeds": list(definition.seeds),
        "checkpoint_epoch": definition.checkpoint_epoch,
        "sweep_config": str(definition.sweep_config) if definition.sweep_config else None,
        "grid": sweep_cfg.get("grid", {}) if definition.kind == "sweep" else {},
        "records": {},
        "phases": {
            "training_complete": False,
            "join_complete": False,
            "combine_complete": False,
            "aggregate_complete": False,
            "output_complete": False,
        },
        "notebook_output": {
            "run_name": definition.notebook_run_name,
            "eval_csv_name": definition.notebook_eval_csv_name,
            "path": str(definition.notebook_output_path),
        },
    }


def _load_manifest(definition: ExperimentDefinition) -> dict[str, Any]:
    return _load_json(definition.manifest_path, _base_manifest(definition))


def _record_path_exists(record: dict[str, Any], field: str) -> bool:
    value = record.get(field)
    return bool(value and Path(value).exists())


def _record_is_trained(record: dict[str, Any] | None, checkpoint_epoch: int) -> bool:
    return bool(
        record
        and record.get("status") in TRAINED_STATUSES
        and int(record.get("checkpoint_epoch", -1)) == checkpoint_epoch
        and _record_path_exists(record, "eval_csv_path")
        and _record_path_exists(record, "checkpoint_path")
    )


def _record_is_joined(record: dict[str, Any] | None) -> bool:
    return bool(record and record.get("status") in JOINED_STATUSES and _record_path_exists(record, "joined_path"))


def _copy_adopted_record(record: dict[str, Any], source_manifest: Path) -> dict[str, Any]:
    adopted = copy.deepcopy(record)
    adopted["source_status"] = record.get("status")
    adopted["status"] = "joined_adopted" if record.get("status") == "joined" else "adopted"
    adopted["adopted_from"] = str(source_manifest)
    return adopted


def adopt_existing_records(
    definition: ExperimentDefinition,
    manifest: dict[str, Any],
    *,
    source_manifest: Path = MAIN_3SEED_MANIFEST,
) -> int:
    """Adopt reusable records from the completed three-seed manifest."""
    if not source_manifest.exists():
        return 0

    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_section = "trainval" if definition.kind == "trainval" else "sweep"
    source_records = source.get(source_section, {}).get("records", {})
    expected = expected_training_keys(definition)
    adopted_count = 0

    for key in sorted(expected):
        if key in manifest["records"]:
            continue
        source_record = source_records.get(key)
        if not source_record:
            continue
        if not _record_is_trained(source_record, definition.checkpoint_epoch):
            continue
        adopted = _copy_adopted_record(source_record, source_manifest)
        manifest["records"][key] = adopted
        adopted_count += 1

    if adopted_count:
        manifest.setdefault("source_manifests", [])
        source_str = str(source_manifest)
        if source_str not in manifest["source_manifests"]:
            manifest["source_manifests"].append(source_str)
    return adopted_count


def _trainval_spec(
    definition: ExperimentDefinition,
    *,
    seed: int,
    logs_dir: Path,
) -> dict[str, Any]:
    log_tag = f"{definition.experiment_id}_seed{seed}_{definition.checkpoint_epoch}ep"
    return {
        "key": _record_key_for_trainval(seed),
        "train_args": {
            "conf": TRAINVAL_CONF,
            "seed": seed,
            "train_epochs": definition.checkpoint_epoch,
            "eval_every": definition.checkpoint_epoch,
            "save_every": definition.checkpoint_epoch,
            "log_tag": log_tag,
        },
        "log_dir": _load_config_log_dir(TRAINVAL_CONF, fallback=DEFAULT_SWEEP_LOG_DIR),
        "log_path": logs_dir / f"trainval_seed{seed}_{definition.checkpoint_epoch}ep.log",
    }


def _sweep_spec(
    definition: ExperimentDefinition,
    *,
    seed: int,
    combo: dict[str, Any],
    logs_dir: Path,
) -> dict[str, Any]:
    sweep_cfg = _sweep_cfg(definition)
    base_args = dict(sweep_cfg.get("base_args", {}))
    sweep_log_dir = _repo_path(base_args.pop("log_dir", DEFAULT_SWEEP_LOG_DIR))
    history_sec = float(combo["history_sec"])
    prediction_sec = float(combo["prediction_sec"])
    radius_scale = float(combo["attention_radius_scale"])
    log_tag = (
        f"{definition.experiment_id}_seed{seed}_"
        f"h{_safe_float_token(history_sec)}_"
        f"p{_safe_float_token(prediction_sec)}_"
        f"r{_safe_float_token(radius_scale)}_{definition.checkpoint_epoch}ep"
    )
    key = _record_key_for_sweep(seed, combo)
    return {
        "key": key,
        "combo": combo,
        "train_args": {
            **base_args,
            "seed": seed,
            "train_epochs": definition.checkpoint_epoch,
            "eval_every": definition.checkpoint_epoch,
            "save_every": definition.checkpoint_epoch,
            "history_sec": history_sec,
            "prediction_sec": prediction_sec,
            "log_dir": sweep_log_dir,
            "log_tag": log_tag,
        },
        "log_dir": sweep_log_dir,
        "log_path": logs_dir / f"sweep_{key}_{definition.checkpoint_epoch}ep.log",
        "radius_scale": radius_scale,
    }


def training_specs(
    definition: ExperimentDefinition,
    *,
    trajdata_cache_dir: Path | None = None,
    data_loc_dict: str | None = None,
) -> list[dict[str, Any]]:
    logs_dir = definition.output_root / "logs"
    if definition.kind == "trainval":
        specs = [
            _trainval_spec(definition, seed=seed, logs_dir=logs_dir)
            for seed in definition.seeds
        ]
    else:
        specs = []
        for seed in definition.seeds:
            for combo in experiment_grid(definition):
                specs.append(_sweep_spec(definition, seed=seed, combo=combo, logs_dir=logs_dir))

    for spec in specs:
        if trajdata_cache_dir is not None:
            spec["train_args"]["trajdata_cache_dir"] = trajdata_cache_dir
        if data_loc_dict is not None:
            spec["train_args"]["data_loc_dict"] = data_loc_dict
    return specs


def _run_training_spec(
    args: argparse.Namespace,
    definition: ExperimentDefinition,
    manifest: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    existing = manifest["records"].get(spec["key"])
    if not args.force and _record_is_trained(existing, definition.checkpoint_epoch):
        print(f"  Skipping existing trained run: {spec['key']}")
        return

    log_dir = Path(spec["log_dir"])
    before_log_dirs = set(log_dir.iterdir()) if log_dir.exists() else set()
    before_metrics_dirs = set(args.metrics_root.iterdir()) if args.metrics_root.exists() else set()
    cmd = _train_command(
        python_executable=args.python_executable,
        nproc_per_node=args.nproc_per_node,
        train_args=spec["train_args"],
    )
    _run_command(cmd, log_path=spec["log_path"], dry_run=args.dry_run)

    if args.dry_run:
        return

    model_dir = _latest_new_subdir(log_dir, before_log_dirs)
    metrics_dir = _latest_new_subdir(args.metrics_root, before_metrics_dirs)
    eval_csv_path = metrics_dir / f"eval_epoch_{definition.checkpoint_epoch}.csv"
    checkpoint_path = model_dir / f"model_registrar-{definition.checkpoint_epoch}.pt"
    conf_path = model_dir / "config.json"

    missing = [path for path in [eval_csv_path, checkpoint_path, conf_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Training finished but expected outputs are missing: "
            + ", ".join(str(path) for path in missing)
        )

    manifest["records"][spec["key"]] = {
        "status": "trained",
        "checkpoint_epoch": definition.checkpoint_epoch,
        "train_args": spec["train_args"],
        "model_dir": str(model_dir),
        "metrics_run_dir": str(metrics_dir),
        "run_name": metrics_dir.name,
        "eval_csv_path": str(eval_csv_path),
        "checkpoint_path": str(checkpoint_path),
        "conf_path": str(conf_path),
        "log_file": str(spec["log_path"]),
    }
    _write_json(manifest, definition.manifest_path)


def _assert_training_complete(definition: ExperimentDefinition, manifest: dict[str, Any]) -> None:
    missing = [
        key
        for key in sorted(expected_training_keys(definition))
        if not _record_is_trained(manifest["records"].get(key), definition.checkpoint_epoch)
    ]
    if missing:
        raise RuntimeError(
            f"Training phase is incomplete for {definition.experiment_id}. "
            f"Missing: {missing[:20]}{' ...' if len(missing) > 20 else ''}"
        )


def _assert_join_complete(definition: ExperimentDefinition, manifest: dict[str, Any]) -> None:
    _assert_training_complete(definition, manifest)
    missing = [
        key
        for key in sorted(expected_training_keys(definition))
        if not _record_is_joined(manifest["records"].get(key))
    ]
    if missing:
        raise RuntimeError(
            f"Join phase is incomplete for {definition.experiment_id}. "
            f"Missing: {missing[:20]}{' ...' if len(missing) > 20 else ''}"
        )


def run_train_phase(args: argparse.Namespace, definition: ExperimentDefinition, manifest: dict[str, Any]) -> None:
    adopted_count = adopt_existing_records(definition, manifest)
    if adopted_count:
        print(f"Adopted {adopted_count} reusable record(s).")
        if not args.dry_run:
            _write_json(manifest, definition.manifest_path)

    original_shared = _load_yaml(SHARED_CONFIG_PATH)
    try:
        for spec in training_specs(
            definition,
            trajdata_cache_dir=args.trajdata_cache_dir,
            data_loc_dict=args.data_loc_dict,
        ):
            if definition.kind == "sweep" and not args.dry_run:
                scaled_cfg = _scale_attention_radii(original_shared, float(spec["radius_scale"]))
                _write_yaml(scaled_cfg, SHARED_CONFIG_PATH)
            _run_training_spec(args, definition, manifest, spec)
    finally:
        if definition.kind == "sweep" and not args.dry_run:
            _write_yaml(original_shared, SHARED_CONFIG_PATH)

    if not args.dry_run:
        _assert_training_complete(definition, manifest)
        manifest["phases"]["training_complete"] = True
        _write_json(manifest, definition.manifest_path)


def run_join_phase(args: argparse.Namespace, definition: ExperimentDefinition, manifest: dict[str, Any]) -> None:
    _assert_training_complete(definition, manifest)
    logs_dir = definition.output_root / "logs"

    for key, record in sorted(manifest["records"].items()):
        if not args.force and _record_is_joined(record):
            print(f"  Skipping existing joined run: {key}")
            continue

        join_cmd = [
            args.python_executable,
            "-m",
            "data_preparation.join_characteristic_metrics",
            "--conf",
            record["conf_path"],
            "--metrics_root",
            str(args.metrics_root),
            "--run_dir",
            record["run_name"],
            "--output_root",
            str(args.joined_root),
            "--format",
            args.format,
        ]
        log_path = logs_dir / f"join_{key}.log"
        _run_command(join_cmd, log_path=log_path, dry_run=args.dry_run)

        if not args.dry_run:
            joined_path = args.joined_root / record["run_name"] / f"eval_epoch_{record['checkpoint_epoch']}.{args.format}"
            if not joined_path.exists():
                raise FileNotFoundError(f"Join finished but expected output is missing: {joined_path}")
            record["status"] = "joined_adopted" if record.get("status") == "adopted" else "joined"
            record["joined_path"] = str(joined_path)
            record["join_log_file"] = str(log_path)
            manifest["records"][key] = record
            _write_json(manifest, definition.manifest_path)

    if not args.dry_run:
        manifest["phases"]["join_complete"] = True
        _write_json(manifest, definition.manifest_path)


def _records_in_expected_order(definition: ExperimentDefinition, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [manifest["records"][key] for key in sorted(expected_training_keys(definition))]


def _write_csv_for_notebook(
    df: pd.DataFrame,
    definition: ExperimentDefinition,
    *,
    preserve_row_run_names: bool,
) -> dict[str, str]:
    out_path = definition.notebook_output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()

    if not preserve_row_run_names:
        if "run_name" in out_df.columns:
            out_df["run_name"] = definition.notebook_run_name
        else:
            out_df.insert(0, "run_name", definition.notebook_run_name)
    elif "run_name" not in out_df.columns:
        out_df.insert(0, "run_name", definition.notebook_run_name)

    if "eval_csv_name" in out_df.columns:
        out_df["eval_csv_name"] = definition.notebook_eval_csv_name
    else:
        insert_at = out_df.columns.get_loc("run_name") + 1 if "run_name" in out_df.columns else 0
        out_df.insert(insert_at, "eval_csv_name", definition.notebook_eval_csv_name)

    out_df.to_csv(out_path, index=False)
    return {
        "run_name": definition.notebook_run_name,
        "eval_csv_name": definition.notebook_eval_csv_name,
        "path": str(out_path),
    }


def run_aggregate_phase(args: argparse.Namespace, definition: ExperimentDefinition, manifest: dict[str, Any]) -> None:
    if not definition.aggregate_across_seeds:
        print(f"Skipping seed aggregation for single-seed experiment: {definition.experiment_id}")
        return

    _assert_join_complete(definition, manifest)
    if args.dry_run:
        print(f"Dry run: would seed-average {len(manifest['records'])} joined record(s).")
        return

    records = _records_in_expected_order(definition, manifest)
    aggregated = aggregate_seeded_records(records, expected_seeds=len(definition.seeds))
    output = _write_csv_for_notebook(
        aggregated,
        definition,
        preserve_row_run_names=False,
    )
    manifest["outputs"] = {"notebook": output}
    manifest["phases"]["aggregate_complete"] = True
    manifest["phases"]["output_complete"] = True
    _write_json(manifest, definition.manifest_path)
    write_index_entry(definition, manifest)
    print(f"  Wrote aggregate notebook output: {output['path']}")


def run_combine_phase(args: argparse.Namespace, definition: ExperimentDefinition, manifest: dict[str, Any]) -> None:
    if definition.aggregate_across_seeds:
        run_aggregate_phase(args, definition, manifest)
        return

    _assert_join_complete(definition, manifest)
    if args.dry_run:
        print(f"Dry run: would write direct/combined notebook output for {len(manifest['records'])} record(s).")
        return

    records = _records_in_expected_order(definition, manifest)

    if definition.kind == "trainval":
        if len(records) != 1:
            raise ValueError(f"Direct trainval output expects one record, got {len(records)}")
        source = Path(records[0]["joined_path"])
        df = pd.read_csv(source)
        output = _write_csv_for_notebook(
            df,
            definition,
            preserve_row_run_names=False,
        )
    else:
        run_names = [record["run_name"] for record in records]
        combined = combine_joined_runs(args.joined_root, run_names)
        output = _write_csv_for_notebook(
            combined,
            definition,
            preserve_row_run_names=True,
        )

    manifest["outputs"] = {"notebook": output}
    manifest["phases"]["combine_complete"] = True
    manifest["phases"]["output_complete"] = True
    _write_json(manifest, definition.manifest_path)
    write_index_entry(definition, manifest)
    print(f"  Wrote notebook output: {output['path']}")


def _source_paths_from_record(record: dict[str, Any]) -> list[Path]:
    fields = [
        "model_dir",
        "metrics_run_dir",
        "eval_csv_path",
        "checkpoint_path",
        "conf_path",
        "log_file",
        "joined_path",
        "join_log_file",
    ]
    return [Path(record[field]) for field in fields if record.get(field)]


def manifest_keep_paths(manifest: dict[str, Any], manifest_path: Path) -> set[Path]:
    keep = {manifest_path, manifest_path.parent}
    for record in manifest.get("records", {}).values():
        keep.update(_source_paths_from_record(record))
    output = manifest.get("outputs", {}).get("notebook", {})
    if output.get("path"):
        path = Path(output["path"])
        keep.add(path)
        keep.add(path.parent)
    return {path.resolve() for path in keep}


def _index_entry(definition: ExperimentDefinition, manifest: dict[str, Any]) -> dict[str, Any]:
    records = list(manifest.get("records", {}).values())
    return {
        "manifest": str(definition.manifest_path),
        "kind": definition.kind,
        "seeds": list(definition.seeds),
        "checkpoint_epoch": definition.checkpoint_epoch,
        "grid_size": len(experiment_grid(definition)),
        "phases": manifest.get("phases", {}),
        "source_runs": sorted(record.get("run_name", "") for record in records),
        "notebook": manifest.get("outputs", {}).get("notebook", manifest.get("notebook_output", {})),
    }


def write_index() -> None:
    index = {"experiments": {}}
    for definition in EXPERIMENTS.values():
        if definition.manifest_path.exists():
            manifest = json.loads(definition.manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = _base_manifest(definition)
            adopt_existing_records(definition, manifest)
        index["experiments"][definition.experiment_id] = _index_entry(definition, manifest)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(index, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def write_index_entry(definition: ExperimentDefinition, manifest: dict[str, Any]) -> None:
    write_index()


def all_curated_keep_paths(index_path: Path = INDEX_PATH) -> set[Path]:
    keep: set[Path] = {EXPERIMENT_ROOT.resolve()}
    if index_path.exists():
        keep.add(index_path.resolve())
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for entry in index.get("experiments", {}).values():
            manifest_path = Path(entry.get("manifest", ""))
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                keep.update(manifest_keep_paths(manifest, manifest_path))

    for definition in EXPERIMENTS.values():
        if definition.manifest_path.exists():
            manifest = json.loads(definition.manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = _base_manifest(definition)
        adopt_existing_records(definition, manifest)
        keep.update(manifest_keep_paths(manifest, definition.manifest_path))
    return {path.resolve() for path in keep}


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _is_kept_candidate(candidate: Path, keep_paths: Iterable[Path]) -> bool:
    candidate_resolved = candidate.resolve()
    for keep in keep_paths:
        keep_resolved = keep.resolve()
        if candidate_resolved == keep_resolved:
            return True
        try:
            keep_resolved.relative_to(candidate_resolved)
            return True
        except ValueError:
            pass
        try:
            candidate_resolved.relative_to(keep_resolved)
            return True
        except ValueError:
            pass
    return False


def build_archive_plan(
    *,
    keep_paths: set[Path],
    archive_root: Path,
    scan_roots: Iterable[Path] | None = None,
    prediction_root: Path | None = None,
) -> list[dict[str, Any]]:
    prediction_root = prediction_root or ROOT / "results" / "trajectory_prediction"
    roots = list(scan_roots) if scan_roots is not None else [prediction_root / rel for rel in SCAN_RELATIVE_ROOTS]
    plan = []
    for scan_root in roots:
        if not scan_root.exists():
            continue
        for candidate in sorted(scan_root.iterdir()):
            if _is_kept_candidate(candidate, keep_paths):
                continue
            rel = candidate.resolve().relative_to(prediction_root.resolve())
            destination = archive_root / "trajectory_prediction" / rel
            plan.append(
                {
                    "source": str(candidate),
                    "destination": str(destination),
                    "bytes": _path_size(candidate),
                    "reason": "not referenced by curated prediction result sets",
                }
            )
    return sorted(plan, key=lambda item: item["source"])


def run_archive_unused(args: argparse.Namespace) -> None:
    archive_root = ROOT / "results" / "_archive" / f"unused_prediction_results_{args.archive_timestamp}"
    keep_paths = all_curated_keep_paths()
    plan = build_archive_plan(keep_paths=keep_paths, archive_root=archive_root)
    manifest = {
        "created_at": _timestamp(),
        "dry_run": args.dry_run,
        "archive_root": str(archive_root),
        "moved": plan,
        "kept_count": len(keep_paths),
    }

    print(f"Archive candidates: {len(plan)}")
    print(f"Archive root: {archive_root}")
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    for item in plan:
        src = Path(item["source"])
        dst = Path(item["destination"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote archive manifest: {archive_root / 'archive_manifest.json'}")


def run_experiment(args: argparse.Namespace) -> None:
    definition = get_experiment_definition(args.experiment)
    manifest = _load_manifest(definition)
    adopt_existing_records(definition, manifest)
    if not args.dry_run:
        _write_json(manifest, definition.manifest_path)

    print(f"Experiment: {definition.experiment_id}")
    print(f"Manifest:   {definition.manifest_path}")
    print(f"Output:     {definition.notebook_output_path}")
    print(f"Seeds:      {list(definition.seeds)}")
    print(f"Grid size:  {len(experiment_grid(definition))}")

    if args.phase in ("all", "train"):
        run_train_phase(args, definition, manifest)
    if args.dry_run and args.phase == "all":
        print("Dry run stops after training command planning because joins need completed outputs.")
        return
    if args.phase in ("all", "join"):
        run_join_phase(args, definition, manifest)
    if args.phase == "aggregate":
        run_aggregate_phase(args, definition, manifest)
    if args.phase == "combine":
        if definition.aggregate_across_seeds:
            run_aggregate_phase(args, definition, manifest)
        else:
            run_combine_phase(args, definition, manifest)
    if args.phase == "all":
        if definition.aggregate_across_seeds:
            run_aggregate_phase(args, definition, manifest)
        else:
            run_combine_phase(args, definition, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=None,
        help="Curated result set to run. Not required with --archive-unused.",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "join", "combine", "aggregate"),
        default="all",
    )
    parser.add_argument("--archive-unused", action="store_true")
    parser.add_argument("--archive-timestamp", default=_timestamp())
    parser.add_argument("--metrics_root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--joined_root", type=Path, default=DEFAULT_JOINED_ROOT)
    parser.add_argument(
        "--trajdata_cache_dir",
        type=Path,
        default=None,
        help="Override the trajdata cache directory passed to every training run.",
    )
    parser.add_argument(
        "--data_loc_dict",
        default=None,
        help=(
            "Override the dataset locations passed to every training run, "
            'e.g. {"nusc_mini":"/data/nuScenes"}.'
        ),
    )
    parser.add_argument("--python_executable", default=DEFAULT_PYTHON)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.metrics_root = _repo_path(args.metrics_root)
    args.joined_root = _repo_path(args.joined_root)
    if args.trajdata_cache_dir is not None:
        args.trajdata_cache_dir = _repo_path(args.trajdata_cache_dir)
    if args.data_loc_dict is not None:
        try:
            parsed_data_locations = json.loads(args.data_loc_dict)
        except json.JSONDecodeError as exc:
            parser.error(f"--data_loc_dict must be valid JSON: {exc}")
        if not isinstance(parsed_data_locations, dict):
            parser.error("--data_loc_dict must decode to a JSON object")
        args.data_loc_dict = json.dumps(parsed_data_locations, separators=(",", ":"))

    if args.archive_unused:
        return args
    if args.experiment is None:
        parser.error("--experiment is required unless --archive-unused is used")
    return args


def main() -> None:
    args = parse_args()
    if args.archive_unused:
        run_archive_unused(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
