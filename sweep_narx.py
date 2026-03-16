"""Random hyperparameter sweep for the paper-style NARX model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Any

from config import build_arg_parser, config_from_args
from data.scenario_bundle import build_scenario_bundle, save_scenario_bundle
from utils.io_utils import save_json
from utils.logging_utils import setup_logger


DEFAULT_ACTIVATIONS = ["linear", "relu", "tanh", "sigmoid", "snake"]
DEFAULT_OPTIMIZERS = ["adam", "adagrad", "sgd", "yogi"]


def build_sweep_arg_parser() -> argparse.ArgumentParser:
    """Build the sweep-specific CLI parser."""
    parser = argparse.ArgumentParser(
        description="Paper-style hyperparameter sweep for the NARX residual predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep_output_dir", default="outputs/sweeps/narx_table1")
    parser.add_argument("--sweep_name", default="narx_table1")
    parser.add_argument("--sweep_num_trials", type=int, default=24)
    parser.add_argument("--sweep_seed", type=int, default=42)
    parser.add_argument("--sweep_resume", action="store_true")
    parser.add_argument("--sweep_top_k", type=int, default=5)
    parser.add_argument(
        "--sweep_objective",
        choices=["corrected_3d_rmse", "improvement_percent"],
        default="corrected_3d_rmse",
        help="Objective used to rank trials.",
    )
    parser.add_argument("--sweep_activations", nargs="+", default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--sweep_optimizers", nargs="+", default=DEFAULT_OPTIMIZERS)
    parser.add_argument("--sweep_min_layers", type=int, default=1)
    parser.add_argument("--sweep_max_layers", type=int, default=10)
    parser.add_argument("--sweep_min_nodes", type=int, default=4)
    parser.add_argument("--sweep_max_nodes", type=int, default=512)
    parser.add_argument("--sweep_min_prediction_length_sec", type=float, default=0.1)
    parser.add_argument("--sweep_max_prediction_length_sec", type=float, default=240.0)
    parser.add_argument(
        "--sweep_prediction_quantum_sec",
        type=float,
        default=None,
        help="Discrete sampling quantum for prediction length. If omitted, it is inferred from dt_train and dt_full.",
    )
    return parser


def infer_prediction_quantum_sec(dt_train_sec: float, dt_full_sec: float) -> float:
    """Infer the smallest common quantum that is compatible with both sampling intervals."""
    scale = 1_000_000
    train_ticks = int(round(dt_train_sec * scale))
    full_ticks = int(round(dt_full_sec * scale))
    if train_ticks <= 0 or full_ticks <= 0:
        raise ValueError("Sampling intervals must be positive.")
    quantum_ticks = math.lcm(train_ticks, full_ticks)
    return quantum_ticks / scale


def sample_log_uniform_int(rng: random.Random, low: int, high: int) -> int:
    """Sample an integer from a log-uniform distribution."""
    if low <= 0 or high <= 0:
        raise ValueError("Log-uniform bounds must be positive.")
    value = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
    return max(low, min(value, high))


def sample_prediction_length_sec(
    rng: random.Random,
    low_sec: float,
    high_sec: float,
    quantum_sec: float,
) -> float:
    """Sample a prediction length in seconds as a discrete multiple of quantum_sec."""
    min_step = int(math.ceil(low_sec / quantum_sec - 1e-12))
    max_step = int(math.floor(high_sec / quantum_sec + 1e-12))
    if min_step > max_step:
        raise ValueError("No valid prediction lengths exist for the requested range and quantum.")
    sampled_step = rng.randint(min_step, max_step)
    return round(sampled_step * quantum_sec, 9)


def generate_trial_specs(args, base_dt_train_sec: float, base_dt_full_sec: float) -> list[dict[str, Any]]:
    """Generate the random trial specifications."""
    rng = random.Random(args.sweep_seed)
    quantum_sec = args.sweep_prediction_quantum_sec or infer_prediction_quantum_sec(base_dt_train_sec, base_dt_full_sec)
    trials: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    max_attempts = max(args.sweep_num_trials * 50, 200)

    while len(trials) < args.sweep_num_trials:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Failed to sample enough unique hyperparameter combinations for the sweep.")

        trial = {
            "trial_index": len(trials) + 1,
            "narx_activation": rng.choice(args.sweep_activations),
            "narx_num_hidden_layers": rng.randint(args.sweep_min_layers, args.sweep_max_layers),
            "narx_hidden_dim": sample_log_uniform_int(rng, args.sweep_min_nodes, args.sweep_max_nodes),
            "narx_prediction_length_sec": sample_prediction_length_sec(
                rng,
                args.sweep_min_prediction_length_sec,
                args.sweep_max_prediction_length_sec,
                quantum_sec,
            ),
            "optimizer_name": rng.choice(args.sweep_optimizers),
        }
        signature = tuple(trial[key] for key in sorted(trial.keys()) if key != "trial_index")
        if signature in seen:
            continue
        seen.add(signature)
        trials.append(trial)
    return trials


def validate_base_main_args(main_arg_tokens: list[str]) -> argparse.Namespace:
    """Validate the main.py argument payload that will be reused in each trial."""
    parser = build_arg_parser()
    return parser.parse_args(main_arg_tokens)


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load an existing sweep manifest."""
    with manifest_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return list(payload["trials"])


def save_manifest(
    manifest_path: Path,
    trials: list[dict[str, Any]],
    sweep_args: argparse.Namespace,
    base_main_args: list[str],
    base_main_namespace: argparse.Namespace,
    shared_bundle_path: str | None = None,
) -> None:
    """Save the sweep manifest for resume/reproducibility."""
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sweep_args": vars(sweep_args),
        "base_main_args": base_main_args,
        "base_main_config": vars(base_main_namespace),
        "shared_precomputed_bundle_path": shared_bundle_path,
        "trials": trials,
    }
    save_json(payload, str(manifest_path))


def extract_metrics(metrics_path: Path) -> dict[str, Any]:
    """Extract the headline metrics from metrics.json."""
    with metrics_path.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)
    forecast = metrics["forecast_segment"]
    return {
        "baseline_3d_rmse_m": forecast["baseline_3d_position_rmse_m"],
        "corrected_3d_rmse_m": forecast["corrected_3d_position_rmse_m"],
        "improvement_percent": forecast["improvement_percent_3d_rmse"],
        "baseline_final_error_m": forecast["baseline_final_position_error_m"],
        "corrected_final_error_m": forecast["corrected_final_position_error_m"],
    }


def objective_value(objective_name: str, metrics: dict[str, Any]) -> float:
    """Compute the sortable objective value."""
    if objective_name == "corrected_3d_rmse":
        return float(metrics["corrected_3d_rmse_m"])
    if objective_name == "improvement_percent":
        return -float(metrics["improvement_percent"])
    raise ValueError(f"Unsupported objective: {objective_name}")


def trial_command(
    repo_root: Path,
    base_main_args: list[str],
    precomputed_bundle_path: Path,
    trial_dir: Path,
    trial: dict[str, Any],
) -> list[str]:
    """Build the subprocess command for a single trial."""
    return [
        sys.executable,
        str(repo_root / "main.py"),
        *base_main_args,
        "--model_name",
        "narx",
        "--precomputed_bundle_path",
        str(precomputed_bundle_path),
        "--output_dir",
        str(trial_dir),
        "--narx_activation",
        str(trial["narx_activation"]),
        "--narx_num_hidden_layers",
        str(trial["narx_num_hidden_layers"]),
        "--narx_hidden_dim",
        str(trial["narx_hidden_dim"]),
        "--narx_prediction_length",
        str(trial["narx_prediction_length_sec"]),
        "--optimizer_name",
        str(trial["optimizer_name"]),
    ]


def resolve_shared_bundle_path(sweep_root: Path, base_main_namespace: argparse.Namespace) -> Path:
    """Resolve the bundle path used across all sweep trials."""
    if base_main_namespace.precomputed_bundle_path:
        return Path(base_main_namespace.precomputed_bundle_path).expanduser().resolve()
    return (sweep_root / "shared" / "scenario_bundle.npz").resolve()


def ensure_shared_bundle(
    base_main_namespace: argparse.Namespace,
    shared_bundle_path: Path,
) -> Path:
    """Build the shared propagation/residual bundle once, or reuse an existing copy."""
    if shared_bundle_path.exists():
        print(f"[precompute] reusing shared scenario bundle: {shared_bundle_path}")
        return shared_bundle_path

    shared_root = shared_bundle_path.parent
    shared_root.mkdir(parents=True, exist_ok=True)
    shared_cache_dir = shared_root / "cache"
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(shared_root), logger_name="orbit_sweep_precompute")
    base_config = config_from_args(base_main_namespace)
    logger.info("Building shared scenario bundle for sweep reuse.")
    bundle = build_scenario_bundle(config=base_config, cache_dir=str(shared_cache_dir), logger=logger)
    save_scenario_bundle(bundle, str(shared_bundle_path))
    logger.info("Saved shared scenario bundle to %s", shared_bundle_path)
    print(f"[precompute] saved shared scenario bundle: {shared_bundle_path}")
    return shared_bundle_path


def save_results_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Save the sweep summary CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_trial(
    repo_root: Path,
    base_main_args: list[str],
    precomputed_bundle_path: Path,
    trial_root: Path,
    trial: dict[str, Any],
    objective_name: str,
) -> dict[str, Any]:
    """Run one trial and return the summary row."""
    trial_id = f"trial_{trial['trial_index']:03d}"
    trial_dir = trial_root / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    command = trial_command(
        repo_root=repo_root,
        base_main_args=base_main_args,
        precomputed_bundle_path=precomputed_bundle_path,
        trial_dir=trial_dir,
        trial=trial,
    )
    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    duration_sec = time.time() - started_at

    (trial_dir / "sweep_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (trial_dir / "sweep_stderr.log").write_text(completed.stderr, encoding="utf-8")

    row: dict[str, Any] = {
        "trial_id": trial_id,
        "status": "FAILED",
        "returncode": completed.returncode,
        "duration_sec": round(duration_sec, 3),
        "output_dir": str(trial_dir),
        "command": shlex.join(command),
        **trial,
    }

    metrics_path = trial_dir / "metrics" / "metrics.json"
    if completed.returncode == 0 and metrics_path.exists():
        metrics = extract_metrics(metrics_path)
        row.update(metrics)
        row["objective_value"] = objective_value(objective_name, metrics)
        row["status"] = "SUCCESS"
    else:
        stderr_tail = completed.stderr.strip().splitlines()
        row["error_tail"] = stderr_tail[-1] if stderr_tail else "Unknown error"
    save_json(row, str(trial_dir / "sweep_trial_summary.json"))
    return row


def sort_rows(rows: list[dict[str, Any]], objective_name: str) -> list[dict[str, Any]]:
    """Sort sweep rows with successful trials first."""
    successes = [row for row in rows if row.get("status") == "SUCCESS"]
    failures = [row for row in rows if row.get("status") != "SUCCESS"]
    successes.sort(key=lambda row: row["objective_value"])
    failures.sort(key=lambda row: row["trial_id"])
    return successes + failures


def main() -> None:
    sweep_parser = build_sweep_arg_parser()
    sweep_args, main_arg_tokens = sweep_parser.parse_known_args()
    if not main_arg_tokens:
        raise SystemExit("You must provide the normal main.py experiment arguments after the sweep arguments.")

    base_main_namespace = validate_base_main_args(main_arg_tokens)
    repo_root = Path(__file__).resolve().parent
    sweep_root = (repo_root / sweep_args.sweep_output_dir).resolve()
    trial_root = sweep_root / "trials"
    sweep_root.mkdir(parents=True, exist_ok=True)
    trial_root.mkdir(parents=True, exist_ok=True)

    manifest_path = sweep_root / "manifest.json"
    results_json_path = sweep_root / "results.json"
    results_csv_path = sweep_root / "results.csv"
    best_json_path = sweep_root / "best_trial.json"
    shared_bundle_path = resolve_shared_bundle_path(sweep_root=sweep_root, base_main_namespace=base_main_namespace)

    if manifest_path.exists():
        if not sweep_args.sweep_resume:
            raise SystemExit(f"Manifest already exists at {manifest_path}. Use --sweep_resume to continue this sweep.")
        trials = load_manifest(manifest_path)
        if len(trials) != sweep_args.sweep_num_trials:
            raise SystemExit(
                f"Existing manifest contains {len(trials)} trials but --sweep_num_trials={sweep_args.sweep_num_trials}. "
                "Keep them consistent when resuming."
            )
    else:
        trials = generate_trial_specs(
            args=sweep_args,
            base_dt_train_sec=base_main_namespace.dt_train,
            base_dt_full_sec=base_main_namespace.dt_full,
        )
        save_manifest(
            manifest_path=manifest_path,
            trials=trials,
            sweep_args=sweep_args,
            base_main_args=main_arg_tokens,
            base_main_namespace=base_main_namespace,
            shared_bundle_path=str(shared_bundle_path),
        )

    ensure_shared_bundle(
        base_main_namespace=base_main_namespace,
        shared_bundle_path=shared_bundle_path,
    )

    existing_rows: dict[str, dict[str, Any]] = {}
    if results_json_path.exists() and sweep_args.sweep_resume:
        with results_json_path.open("r", encoding="utf-8") as fp:
            previous_rows = json.load(fp)
        existing_rows = {row["trial_id"]: row for row in previous_rows}

    all_rows: list[dict[str, Any]] = []
    for trial in trials:
        trial_id = f"trial_{trial['trial_index']:03d}"
        if sweep_args.sweep_resume and trial_id in existing_rows and existing_rows[trial_id].get("status") == "SUCCESS":
            print(f"[resume] skipping completed {trial_id}")
            all_rows.append(existing_rows[trial_id])
            continue

        print(
            f"[run] {trial_id} | activation={trial['narx_activation']} | layers={trial['narx_num_hidden_layers']} "
            f"| nodes={trial['narx_hidden_dim']} | pred_len={trial['narx_prediction_length_sec']} s | opt={trial['optimizer_name']}"
        )
        row = run_trial(
            repo_root=repo_root,
            base_main_args=main_arg_tokens,
            precomputed_bundle_path=shared_bundle_path,
            trial_root=trial_root,
            trial=trial,
            objective_name=sweep_args.sweep_objective,
        )
        all_rows.append(row)
        ordered_rows = sort_rows(all_rows, sweep_args.sweep_objective)
        save_json(ordered_rows, str(results_json_path))
        save_results_csv(ordered_rows, results_csv_path)

    ordered_rows = sort_rows(all_rows, sweep_args.sweep_objective)
    save_json(ordered_rows, str(results_json_path))
    save_results_csv(ordered_rows, results_csv_path)

    best_successes = [row for row in ordered_rows if row.get("status") == "SUCCESS"]
    best_payload = {
        "objective": sweep_args.sweep_objective,
        "num_trials": len(trials),
        "completed_successes": len(best_successes),
        "top_k": best_successes[: max(sweep_args.sweep_top_k, 1)],
    }
    save_json(best_payload, str(best_json_path))

    print(f"[done] summary saved to {results_csv_path}")
    if best_successes:
        best = best_successes[0]
        print(
            "[best] "
            f"{best['trial_id']} | corrected_3d_rmse={best['corrected_3d_rmse_m']:.6f} m | "
            f"improvement={best['improvement_percent']:.3f} % | output_dir={best['output_dir']}"
        )
    else:
        print("[best] no successful trials")


if __name__ == "__main__":
    main()
