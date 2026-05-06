"""Run seeded trainval and mini model-settings experiments.

The workflow is deliberately staged:

1. Train every requested model and persist a manifest after each completed run.
2. Only after all training runs are complete, join characteristic metrics.
3. Only after all joins are complete, average metrics across seeds using stable
   trajectory identifiers.

This keeps model outputs safe if a later join or aggregation step fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
UNIFIED_LOADER_ROOT = ROOT / "unified-av-data-loader" / "src"
SHARED_CONFIG_PATH = ROOT / "config" / "shared_config.yaml"
DEFAULT_SWEEP_CONFIG = ROOT / "config" / "sweep_config.yaml"
DEFAULT_METRICS_ROOT = ROOT / "results" / "trajectory_prediction" / "trajectory_metrics"
DEFAULT_JOINED_ROOT = ROOT / "results" / "trajectory_prediction" / "trajectory_metrics_joined"
DEFAULT_SWEEP_LOG_DIR = ROOT / "results" / "trajectory_prediction" / "logs"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "trajectory_prediction" / "seeded_experiments"
DEFAULT_SEEDS = [123, 456, 789]

TRAJECTORY_INDEX_COL = "data_idx"
TRAJECTORY_IDENTITY_CHECK_COLS = ["scene_path", "agent_id", "scene_ts", "agent_type"]
SETTING_KEY_COLS = [
    "eval_data",
    "history_sec",
    "prediction_sec",
    "restrict_to_predchal",
    "attention_radius_m",
]
TARGET_METRIC_COLS = ["ml_ade", "ml_fde", "min_ade_5", "nll_mean", "nll_final"]

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_training_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a JSON config with the repo's simple `extends` semantics.

    This intentionally avoids importing `shared_config.config_loader` so dry
    runs and unit tests do not require torch/trajdata.
    """
    resolved = path.resolve()
    seen = seen or set()
    if resolved in seen:
        raise ValueError(f"Config inheritance cycle detected: {resolved}")
    seen.add(resolved)

    data = json.loads(resolved.read_text(encoding="utf-8"))
    parent_name = data.pop("extends", None)
    if parent_name is None:
        return data
    parent_path = resolved.parent / parent_name
    parent = _load_training_config(parent_path, seen)
    return _deep_merge(parent, data)


def _repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT / resolved


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(SRC_ROOT), str(UNIFIED_LOADER_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("RANK", "0")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")
    # The local conda stack can load libomp through multiple compiled
    # dependencies. The training process aborts before model fitting without
    # this compatibility flag.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _run_key(*parts: Any) -> str:
    return "__".join(str(part) for part in parts)


def _run_command(
    cmd: list[str],
    *,
    log_path: Path,
    dry_run: bool,
    cwd: Path = ROOT,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log:     {log_path}")
    if dry_run:
        return

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=_subprocess_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=f"See log: {log_path}")


def _new_subdirs(directory: Path, known: set[Path]) -> list[Path]:
    if not directory.exists():
        return []
    return [path for path in directory.iterdir() if path.is_dir() and path not in known]


def _latest_new_subdir(directory: Path, known: set[Path]) -> Path:
    new_dirs = _new_subdirs(directory, known)
    if not new_dirs:
        raise FileNotFoundError(f"No new directory appeared under {directory}")
    return max(new_dirs, key=lambda path: path.stat().st_mtime)


def _scale_attention_radii(shared_cfg: dict[str, Any], scale: float) -> dict[str, Any]:
    cfg = copy.deepcopy(shared_cfg)
    attn = cfg.get("attention_radius", {})
    if "default" in attn:
        attn["default"] = round(float(attn["default"]) * scale, 4)
    for targets in attn.get("pairs", {}).values():
        for dst in targets:
            targets[dst] = round(float(targets[dst]) * scale, 4)
    return cfg


def _build_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    import itertools

    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[grid[key] for key in keys])]


def _load_config_log_dir(conf_path: Path, fallback: Path) -> Path:
    hyperparams = _load_training_config(conf_path)
    raw_log_dir = hyperparams.get("log_dir")
    if raw_log_dir is None:
        return fallback
    return _repo_path(raw_log_dir)


def _base_manifest(args: argparse.Namespace, sweep_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": _timestamp(),
        "seeds": args.seeds,
        "trainval": {"records": {}},
        "sweep": {
            "sweep_config": str(args.sweep_config),
            "grid": sweep_cfg.get("grid", {}),
            "records": {},
        },
        "paths": {
            "metrics_root": str(args.metrics_root),
            "joined_root": str(args.joined_root),
            "output_root": str(args.output_root),
        },
        "phases": {"training_complete": False, "join_complete": False, "aggregate_complete": False},
    }


def _record_is_trained(record: dict[str, Any] | None, checkpoint_epoch: int) -> bool:
    if not record or record.get("status") not in {"trained", "joined"}:
        return False
    eval_path = Path(record.get("eval_csv_path", ""))
    checkpoint_path = Path(record.get("checkpoint_path", ""))
    return eval_path.exists() and checkpoint_path.exists() and int(record.get("checkpoint_epoch", -1)) == checkpoint_epoch


def _train_command(
    *,
    python_executable: str,
    nproc_per_node: int,
    train_args: dict[str, Any],
) -> list[str]:
    if nproc_per_node == 1:
        cmd = [python_executable, str(ROOT / "train_unified.py")]
        for key, value in train_args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
                continue
            cmd += [f"--{key}", str(value)]
        return cmd

    cmd = [
        python_executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        str(ROOT / "train_unified.py"),
    ]
    for key, value in train_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd += [f"--{key}", str(value)]
    return cmd


def _run_one_training(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    section: str,
    key: str,
    train_args: dict[str, Any],
    checkpoint_epoch: int,
    log_dir: Path,
    log_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    existing_record = manifest[section]["records"].get(key)
    if not args.force and _record_is_trained(existing_record, checkpoint_epoch):
        print(f"  Skipping existing trained run: {section}/{key}")
        return existing_record

    before_log_dirs = set(log_dir.iterdir()) if log_dir.exists() else set()
    before_metrics_dirs = set(args.metrics_root.iterdir()) if args.metrics_root.exists() else set()
    cmd = _train_command(
        python_executable=args.python_executable,
        nproc_per_node=args.nproc_per_node,
        train_args=train_args,
    )
    _run_command(cmd, log_path=log_path, dry_run=dry_run)

    if dry_run:
        record = {
            "status": "dry_run",
            "checkpoint_epoch": checkpoint_epoch,
            "train_args": train_args,
            "log_file": str(log_path),
        }
        return record
    else:
        model_dir = _latest_new_subdir(log_dir, before_log_dirs)
        metrics_dir = _latest_new_subdir(args.metrics_root, before_metrics_dirs)
        eval_csv_path = metrics_dir / f"eval_epoch_{checkpoint_epoch}.csv"
        checkpoint_path = model_dir / f"model_registrar-{checkpoint_epoch}.pt"
        conf_path = model_dir / "config.json"

        missing_outputs = [
            path
            for path in [eval_csv_path, checkpoint_path, conf_path]
            if not path.exists()
        ]
        if missing_outputs:
            raise FileNotFoundError(
                "Training finished but expected outputs are missing: "
                + ", ".join(str(path) for path in missing_outputs)
            )

        record = {
            "status": "trained",
            "checkpoint_epoch": checkpoint_epoch,
            "train_args": train_args,
            "model_dir": str(model_dir),
            "metrics_run_dir": str(metrics_dir),
            "run_name": metrics_dir.name,
            "eval_csv_path": str(eval_csv_path),
            "checkpoint_path": str(checkpoint_path),
            "conf_path": str(conf_path),
            "log_file": str(log_path),
        }

    manifest[section]["records"][key] = record
    _write_json(manifest, manifest_path)
    return record


def _run_train_phase(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path) -> None:
    sweep_cfg = _load_yaml(args.sweep_config)
    combos = _build_combinations(sweep_cfg.get("grid", {}))
    original_shared = _load_yaml(SHARED_CONFIG_PATH)

    trainval_log_dir = _load_config_log_dir(args.trainval_conf, fallback=DEFAULT_SWEEP_LOG_DIR)
    sweep_log_dir = _repo_path(sweep_cfg.get("base_args", {}).get("log_dir", DEFAULT_SWEEP_LOG_DIR))
    logs_dir = args.output_root / "logs"

    print(f"Training trainval runs: {len(args.seeds)} seed(s)")
    for seed in args.seeds:
        log_tag = f"{args.trainval_log_tag}_seed{seed}_{args.trainval_epochs}ep"
        key = _run_key("seed", seed)
        train_args = {
            "conf": args.trainval_conf,
            "seed": seed,
            "train_epochs": args.trainval_epochs,
            "eval_every": args.trainval_epochs,
            "save_every": args.trainval_epochs,
            "log_tag": log_tag,
        }
        _run_one_training(
            args=args,
            manifest=manifest,
            manifest_path=manifest_path,
            section="trainval",
            key=key,
            train_args=train_args,
            checkpoint_epoch=args.trainval_epochs,
            log_dir=trainval_log_dir,
            log_path=logs_dir / f"trainval_seed{seed}_{args.trainval_epochs}ep.log",
            dry_run=args.dry_run,
        )

    print(f"Training sweep runs: {len(args.seeds)} seed(s) x {len(combos)} setting(s)")
    base_args = dict(sweep_cfg.get("base_args", {}))
    base_args.pop("log_dir", None)
    try:
        for seed in args.seeds:
            for combo_idx, combo in enumerate(combos, start=1):
                scale = float(combo["attention_radius_scale"])
                scaled_cfg = _scale_attention_radii(original_shared, scale)
                if not args.dry_run:
                    _write_yaml(scaled_cfg, SHARED_CONFIG_PATH)

                history_sec = float(combo["history_sec"])
                prediction_sec = float(combo["prediction_sec"])
                log_tag = (
                    f"{args.sweep_log_tag}_seed{seed}_"
                    f"h{_safe_float_token(history_sec)}_"
                    f"p{_safe_float_token(prediction_sec)}_"
                    f"r{_safe_float_token(scale)}_{args.sweep_epochs}ep"
                )
                key = _run_key(
                    "seed",
                    seed,
                    "history",
                    history_sec,
                    "prediction",
                    prediction_sec,
                    "radius_scale",
                    scale,
                )
                train_args = {
                    **base_args,
                    "seed": seed,
                    "train_epochs": args.sweep_epochs,
                    "eval_every": args.sweep_epochs,
                    "save_every": args.sweep_epochs,
                    "history_sec": history_sec,
                    "prediction_sec": prediction_sec,
                    "log_dir": sweep_log_dir,
                    "log_tag": log_tag,
                }
                print(
                    f"  Sweep setting {combo_idx}/{len(combos)} seed={seed}: "
                    f"history={history_sec}, prediction={prediction_sec}, radius_scale={scale}"
                )
                _run_one_training(
                    args=args,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    section="sweep",
                    key=key,
                    train_args=train_args,
                    checkpoint_epoch=args.sweep_epochs,
                    log_dir=sweep_log_dir,
                    log_path=logs_dir / f"sweep_{key}_{args.sweep_epochs}ep.log",
                    dry_run=args.dry_run,
                )
    finally:
        if not args.dry_run:
            _write_yaml(original_shared, SHARED_CONFIG_PATH)

    if not args.dry_run:
        _assert_training_complete(args, manifest)
        manifest["phases"]["training_complete"] = True
        _write_json(manifest, manifest_path)


def _expected_trainval_keys(args: argparse.Namespace) -> set[str]:
    return {_run_key("seed", seed) for seed in args.seeds}


def _expected_sweep_keys(args: argparse.Namespace) -> set[str]:
    sweep_cfg = _load_yaml(args.sweep_config)
    keys = set()
    for seed in args.seeds:
        for combo in _build_combinations(sweep_cfg.get("grid", {})):
            keys.add(
                _run_key(
                    "seed",
                    seed,
                    "history",
                    float(combo["history_sec"]),
                    "prediction",
                    float(combo["prediction_sec"]),
                    "radius_scale",
                    float(combo["attention_radius_scale"]),
                )
            )
    return keys


def _assert_training_complete(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    missing = []
    for key in sorted(_expected_trainval_keys(args)):
        if not _record_is_trained(manifest["trainval"]["records"].get(key), args.trainval_epochs):
            missing.append(f"trainval/{key}")
    for key in sorted(_expected_sweep_keys(args)):
        if not _record_is_trained(manifest["sweep"]["records"].get(key), args.sweep_epochs):
            missing.append(f"sweep/{key}")
    if missing:
        raise RuntimeError(
            "Training phase is incomplete; joins and aggregation will not run. Missing: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )


def _joined_file_for_record(args: argparse.Namespace, record: dict[str, Any]) -> Path:
    return args.joined_root / record["run_name"] / f"eval_epoch_{record['checkpoint_epoch']}.{args.format}"


def _record_is_joined(args: argparse.Namespace, record: dict[str, Any] | None) -> bool:
    return bool(record and record.get("status") == "joined" and Path(record.get("joined_path", "")).exists())


def _run_join_phase(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path) -> None:
    _assert_training_complete(args, manifest)
    logs_dir = args.output_root / "logs"

    for section in ["trainval", "sweep"]:
        for key, record in sorted(manifest[section]["records"].items()):
            if not args.force and _record_is_joined(args, record):
                print(f"  Skipping existing joined run: {section}/{key}")
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
            log_path = logs_dir / f"join_{section}_{key}.log"
            _run_command(join_cmd, log_path=log_path, dry_run=args.dry_run)
            if not args.dry_run:
                joined_path = _joined_file_for_record(args, record)
                if not joined_path.exists():
                    raise FileNotFoundError(f"Join finished but expected output is missing: {joined_path}")
                record["status"] = "joined"
                record["joined_path"] = str(joined_path)
                record["join_log_file"] = str(log_path)
                manifest[section]["records"][key] = record
                _write_json(manifest, manifest_path)

    if not args.dry_run:
        manifest["phases"]["join_complete"] = True
        _write_json(manifest, manifest_path)


def _missing_aggregation_cols(df: pd.DataFrame) -> list[str]:
    required_cols = [TRAJECTORY_INDEX_COL] + TRAJECTORY_IDENTITY_CHECK_COLS
    return [col for col in required_cols if col not in df.columns]


def _resolve_group_cols(df: pd.DataFrame) -> list[str]:
    missing = _missing_aggregation_cols(df)
    if missing:
        raise KeyError(
            "Cannot aggregate across seeds because trajectory aggregation "
            f"columns are missing: {missing}"
        )

    setting_cols = [col for col in SETTING_KEY_COLS if col in df.columns]
    return setting_cols + [TRAJECTORY_INDEX_COL]


def _validate_identity_constant_within_groups(
    df: pd.DataFrame,
    group_cols: list[str],
) -> None:
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    mismatched_cols = []
    for col in TRAJECTORY_IDENTITY_CHECK_COLS:
        nunique = grouped[col].nunique(dropna=False)
        bad_groups = nunique[nunique > 1]
        if not bad_groups.empty:
            sample_key = bad_groups.index[0]
            mismatched_cols.append({"column": col, "sample_group": sample_key})

    if mismatched_cols:
        raise ValueError(
            "data_idx is not stable across seeds for at least one setting. "
            "Semantic trajectory identity columns vary within the aggregation key. "
            f"Mismatches: {mismatched_cols[:5]}"
        )


def _read_joined_records(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for record in records:
        joined_path = Path(record.get("joined_path", ""))
        if not joined_path.exists():
            raise FileNotFoundError(f"Joined output not found: {joined_path}")
        df = pd.read_csv(joined_path)
        df.insert(0, "seed", int(record["train_args"]["seed"]))
        if "run_name" not in df.columns:
            df.insert(1, "run_name", record["run_name"])
        if "eval_csv_name" not in df.columns:
            insert_at = df.columns.get_loc("run_name") + 1
            df.insert(insert_at, "eval_csv_name", joined_path.name)
        df["checkpoint_epoch"] = int(record["checkpoint_epoch"])
        frames.append(df)
    if not frames:
        raise ValueError("No joined records were provided for aggregation.")
    return pd.concat(frames, ignore_index=True)


def aggregate_seeded_records(
    records: Iterable[dict[str, Any]],
    *,
    expected_seeds: int,
    allow_incomplete_seed_groups: bool = False,
) -> pd.DataFrame:
    """Average seed-varying eval metrics using stable trajectory identifiers."""
    df = _read_joined_records(records)
    group_cols = _resolve_group_cols(df)
    _validate_identity_constant_within_groups(df, group_cols)

    duplicate_count = int(df.duplicated(subset=["seed"] + group_cols).sum())
    if duplicate_count:
        raise ValueError(
            "Joined data are not unique per seed and stable trajectory key. "
            f"Duplicate rows: {duplicate_count}. Key: {['seed'] + group_cols}"
        )

    metric_cols = [col for col in TARGET_METRIC_COLS if col in df.columns]
    first_cols = [
        col
        for col in df.columns
        if col not in set(group_cols + metric_cols + ["seed", "run_name"])
    ]

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    seed_counts = grouped["seed"].nunique().rename("n_seeds").reset_index()
    incomplete = seed_counts[seed_counts["n_seeds"] != expected_seeds]
    if not incomplete.empty and not allow_incomplete_seed_groups:
        raise ValueError(
            f"{len(incomplete)} trajectory groups do not have all {expected_seeds} seeds. "
            "Use --allow_incomplete_seed_groups if this is intentional."
        )

    pieces = [seed_counts]
    if first_cols:
        pieces.append(grouped[first_cols].first().reset_index(drop=True))

    for metric in metric_cols:
        stats = grouped[metric].agg(["mean", "std", "min", "max"]).reset_index(drop=True)
        pieces.append(
            pd.DataFrame(
                {
                    metric: stats["mean"],
                    f"{metric}_seed_std": stats["std"].fillna(0.0),
                    f"{metric}_seed_min": stats["min"],
                    f"{metric}_seed_max": stats["max"],
                }
            )
        )

    run_names = grouped["run_name"].agg(lambda values: "|".join(sorted(set(map(str, values))))).reset_index(drop=True)
    seeds = grouped["seed"].agg(lambda values: "|".join(map(str, sorted(set(map(int, values)))))).reset_index(drop=True)
    pieces.append(pd.DataFrame({"source_run_names": run_names, "seed_values": seeds}))

    return pd.concat(pieces, axis=1)


def _write_aggregate_outputs(
    args: argparse.Namespace,
    *,
    section: str,
    checkpoint_epoch: int,
    aggregated: pd.DataFrame,
) -> dict[str, str]:
    aggregate_dir = args.output_root / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = aggregate_dir / f"{section}_seed_averaged_epoch_{checkpoint_epoch}.{args.format}"
    aggregate_for_write = aggregated.copy()
    run_name = (
        args.aggregate_trainval_run_name
        if section == "trainval"
        else args.aggregate_sweep_run_name
    )
    run_eval_name = f"eval_epoch_{checkpoint_epoch}_seed_mean.{args.format}"
    if "run_name" in aggregate_for_write.columns:
        aggregate_for_write["run_name"] = run_name
    else:
        aggregate_for_write.insert(0, "run_name", run_name)
    if "eval_csv_name" in aggregate_for_write.columns:
        aggregate_for_write["eval_csv_name"] = run_eval_name
    else:
        aggregate_for_write.insert(1, "eval_csv_name", run_eval_name)

    if args.format == "parquet":
        aggregate_for_write.to_parquet(aggregate_path, index=False)
    else:
        aggregate_for_write.to_csv(aggregate_path, index=False)

    run_layout_dir = args.joined_root / run_name
    run_layout_dir.mkdir(parents=True, exist_ok=True)
    run_layout_path = run_layout_dir / run_eval_name
    if args.format == "parquet":
        aggregate_for_write.to_parquet(run_layout_path, index=False)
    else:
        aggregate_for_write.to_csv(run_layout_path, index=False)

    return {
        "aggregate_path": str(aggregate_path),
        "run_layout_dir": str(run_layout_dir),
        "run_layout_path": str(run_layout_path),
        "run_name": run_name,
        "eval_csv_name": run_eval_name,
    }


def _run_aggregate_phase(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path) -> None:
    if not manifest["phases"].get("join_complete") and not args.dry_run:
        raise RuntimeError("Join phase is not complete; refusing to aggregate.")

    outputs: dict[str, Any] = {}
    for section, checkpoint_epoch in [
        ("trainval", args.trainval_epochs),
        ("sweep", args.sweep_epochs),
    ]:
        records = list(manifest[section]["records"].values())
        if not args.dry_run:
            for record in records:
                if not _record_is_joined(args, record):
                    raise RuntimeError(f"Record is not joined yet: {section}/{record.get('run_name')}")
        print(f"Aggregating {section} records across {len(args.seeds)} seed(s)")
        if args.dry_run:
            continue
        aggregated = aggregate_seeded_records(
            records,
            expected_seeds=len(args.seeds),
            allow_incomplete_seed_groups=args.allow_incomplete_seed_groups,
        )
        outputs[section] = _write_aggregate_outputs(
            args,
            section=section,
            checkpoint_epoch=checkpoint_epoch,
            aggregated=aggregated,
        )
        print(f"  Wrote {section} aggregate: {outputs[section]['aggregate_path']}")
        print(f"  Wrote notebook run layout: {outputs[section]['run_layout_path']}")

    if not args.dry_run:
        manifest["aggregates"] = outputs
        manifest["phases"]["aggregate_complete"] = True
        _write_json(manifest, manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run three-seed trainval and mini model-settings experiments, then "
            "join and seed-average outputs after all model runs finish."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "join", "aggregate"),
        default="all",
        help="Workflow phase to run. 'all' enforces train before join before aggregate.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--trainval_conf", type=Path, default=ROOT / "config" / "nuScenes_full_trainval.json")
    parser.add_argument("--sweep_config", type=Path, default=DEFAULT_SWEEP_CONFIG)
    parser.add_argument("--metrics_root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--joined_root", type=Path, default=DEFAULT_JOINED_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT / _timestamp())
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--trainval_epochs", type=int, default=12)
    parser.add_argument("--sweep_epochs", type=int, default=30)
    parser.add_argument("--trainval_log_tag", default="seeded_trainval")
    parser.add_argument("--sweep_log_tag", default="seeded_sweep")
    parser.add_argument("--aggregate_trainval_run_name", default=None)
    parser.add_argument("--aggregate_sweep_run_name", default=None)
    parser.add_argument("--allow_incomplete_seed_groups", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun phases even when manifest outputs exist.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and planned outputs without running them.")
    args = parser.parse_args()

    args.trainval_conf = _repo_path(args.trainval_conf)
    args.sweep_config = _repo_path(args.sweep_config)
    args.metrics_root = _repo_path(args.metrics_root)
    args.joined_root = _repo_path(args.joined_root)
    args.output_root = _repo_path(args.output_root)
    args.manifest = _repo_path(args.manifest) if args.manifest is not None else args.output_root / "manifest.json"
    if args.aggregate_trainval_run_name is None:
        args.aggregate_trainval_run_name = (
            f"seeded_trainval_{args.trainval_epochs}ep_{len(args.seeds)}seeds"
        )
    if args.aggregate_sweep_run_name is None:
        args.aggregate_sweep_run_name = (
            f"seeded_sweep_{args.sweep_epochs}ep_{len(args.seeds)}seeds"
        )
    return args


def main() -> None:
    args = parse_args()
    sweep_cfg = _load_yaml(args.sweep_config)
    manifest = _load_json(args.manifest, _base_manifest(args, sweep_cfg))

    print(f"Manifest: {args.manifest}")
    print(f"Output root: {args.output_root}")
    print(f"Seeds: {args.seeds}")

    if args.phase in ("all", "train"):
        _run_train_phase(args, manifest, args.manifest)
    if args.dry_run and args.phase == "all":
        print("Dry run stops after training command planning because join paths are created by completed runs.")
        print("Done.")
        return
    if args.phase in ("all", "join"):
        _run_join_phase(args, manifest, args.manifest)
    if args.phase in ("all", "aggregate"):
        _run_aggregate_phase(args, manifest, args.manifest)

    print("Done.")


if __name__ == "__main__":
    main()
