#!/usr/bin/env python
"""
W&B Sweep Agent for Model Editing Experiments

This script is designed to be called by wandb sweep agents to run hyperparameter
search experiments for the ViT model editing pipeline.

It uses subprocess to call main.py, ensuring consistency with param_search.py
and proper ASTRA layer selection.

Usage:
    # 1. Create sweep (returns sweep_id)
    wandb sweep configs/sweep_pathmnist.yaml

    # 2. Run agent (single GPU)
    wandb agent <username>/<project>/<sweep_id>

    # 3. Multi-GPU parallel (each terminal runs one agent)
    CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id> &
    CUDA_VISIBLE_DEVICES=1 wandb agent <sweep_id> &

    # 4. Limit number of runs
    wandb agent <sweep_id> --count 10
"""

import csv
import subprocess
import time
from pathlib import Path

# Import wandb BEFORE modifying sys.path, because the project root contains
# a wandb/ directory (created by wandb for run logs) that would shadow the package.
import wandb

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"


def format_threshold(threshold: float) -> str:
    """Format threshold for directory naming (e.g., 1e-2 -> '1e-2')."""
    if threshold >= 1:
        return str(int(threshold))
    exp = int(f"{threshold:.0e}".split("e")[1])
    mantissa = threshold / (10 ** exp)
    if mantissa == 1:
        return f"1e{exp}"
    return f"{mantissa:.0f}e{exp}"


def build_run_name(config) -> str:
    """Build structured run name from wandb config."""
    dataset = config.dataset
    proj = config.projection_samples
    edit = config.max_edits
    thresh = format_threshold(config.nullspace_threshold)
    layer_desc = f"astra{config.num_edit_layers}"
    return f"{dataset}/proj{proj}_edit{edit}_{layer_desc}_thresh{thresh}"


def parse_metric_csv(csv_path: Path) -> dict:
    """Parse a metric CSV file (format: metric,value,notes)."""
    metrics = {}
    if not csv_path.exists():
        return metrics
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get('metric', '')
            value = row.get('value', '')
            try:
                metrics[metric] = float(value)
            except (ValueError, TypeError):
                metrics[metric] = value
    return metrics


def parse_results(results_dir: Path) -> dict:
    """Parse results from a completed experiment (same as param_search.py)."""
    results = {}
    csv_files = [
        ("comparative_evaluation_edit_samples.csv", "edit"),
        ("comparative_evaluation_projection_samples.csv", "proj"),
        ("comparative_evaluation_test_set.csv", "test"),
        ("comparative_evaluation_edit_discovery_set.csv", "discovery"),
    ]
    for csv_name, prefix in csv_files:
        csv_path = results_dir / csv_name
        metrics = parse_metric_csv(csv_path)
        for metric, value in metrics.items():
            results[f'{prefix}_{metric}'] = value
    return results


def train():
    """Training function called by wandb agent.

    Reads hyperparameters from wandb.config and runs edit+eval pipeline
    via subprocess (same pattern as param_search.py).
    """
    run = wandb.init()
    config = wandb.config

    print(f"\n{'='*70}")
    print(f"W&B SWEEP RUN: {run.name}")
    print(f"{'='*70}")
    print(f"Config: {dict(config)}")

    run_name = build_run_name(config)
    results_dir = Path("results") / run_name

    # Build command (same pattern as param_search.py)
    base_cmd = [
        "uv", "run", "python", str(SRC_DIR / "main.py"),
        "--dataset", config.dataset,
        "--run-name", run_name,
        "--max-samples", "50",
        "--max-edits", str(config.max_edits),
        "--projection-samples", str(config.projection_samples),
        "--nullspace-threshold", str(config.nullspace_threshold),
        "--num-edit-layers", str(config.num_edit_layers),
    ]

    # Add data-path for liver datasets
    if hasattr(config, 'data_path') and config.data_path:
        base_cmd.extend(["--data-path", config.data_path])
    elif config.dataset.startswith('liver'):
        base_cmd.extend(["--data-path", "dataset/"])

    cmd_edit = base_cmd + ["--stage", "edit"]
    cmd_eval = base_cmd + ["--stage", "eval"]

    start_time = time.time()
    edit_time = 0

    try:
        # Run edit stage
        print("\n--- Running EDIT stage ---")
        print(f"Command: {' '.join(cmd_edit)}")
        edit_start = time.time()
        result_edit = subprocess.run(
            cmd_edit,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=3600  # 1 hour timeout
        )
        edit_time = time.time() - edit_start

        if result_edit.returncode != 0:
            print(f"STDOUT: {result_edit.stdout[-2000:]}")
            print(f"STDERR: {result_edit.stderr[-2000:]}")
            raise RuntimeError(f"Edit stage failed (rc={result_edit.returncode})")

        print(f"Edit stage completed in {edit_time:.1f}s")

        # Run eval stage
        print("\n--- Running EVAL stage ---")
        result_eval = subprocess.run(
            cmd_eval,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=1800  # 30 min timeout
        )

        if result_eval.returncode != 0:
            print(f"STDOUT: {result_eval.stdout[-2000:]}")
            print(f"STDERR: {result_eval.stderr[-2000:]}")
            raise RuntimeError(f"Eval stage failed (rc={result_eval.returncode})")

        total_time = time.time() - start_time

        # Parse results (same as param_search.py)
        results = parse_results(results_dir)

        # Log metrics to wandb
        metrics = {
            "edit/fix_rate": results.get("edit_fix_rate", 0),
            "edit/accuracy_delta": results.get("edit_accuracy_delta", 0),
            "test/accuracy_before": results.get("test_accuracy_orig", 0),
            "test/accuracy_after": results.get("test_accuracy_edit", 0),
            "test/accuracy_delta": results.get("test_accuracy_delta", 0),
            "test/stability": results.get("test_stability", 0),
            "test/regression_rate": results.get("test_regression_rate", 0),
            "proj/stability": results.get("proj_stability", 0),
            "proj/regression_rate": results.get("proj_regression_rate", 0),
            "discovery/accuracy_delta": results.get("discovery_accuracy_delta", 0),
            "timing/edit_seconds": edit_time,
            "timing/total_seconds": total_time,
        }
        wandb.log(metrics)

        # Summary for sweep ranking
        wandb.summary["test_accuracy_delta"] = results.get("test_accuracy_delta", 0)
        wandb.summary["test_stability"] = results.get("test_stability", 0)
        wandb.summary["edit_fix_rate"] = results.get("edit_fix_rate", 0)
        wandb.summary["status"] = "success"

        print(f"\n[OK] Run completed in {total_time:.1f}s")
        print(f"  Test accuracy delta: {results.get('test_accuracy_delta', 0)*100:+.2f}%")
        print(f"  Edit fix rate: {results.get('edit_fix_rate', 0)*100:.1f}%")

    except Exception as e:
        wandb.log({"error": str(e)})
        wandb.summary["status"] = "failed"
        print(f"\n[ERROR] Run failed: {e}")
        raise

    finally:
        run.finish()


if __name__ == "__main__":
    train()
