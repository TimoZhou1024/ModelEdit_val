#!/usr/bin/env python
"""
Parameter Search Script for Model Editing (v2)

This script explores different parameter combinations to find optimal editing settings.
It runs both edit and eval stages to collect complete metrics including test set results.

Search Dimensions:
- --datasets: MedMNIST datasets to search
- --projection-samples-range: Number of FT-Train samples for projection matrix
- --nullspace-threshold-range: Eigenvalue threshold for null-space selection
- --num-edit-layers-range: Number of top ASTRA layers to edit
- --fixed-edit-layers: Fixed layer combinations (alternative to ASTRA)
- --max-edits-range: Number of error samples to edit

Multi-GPU Support:
- --parallel N: Run N experiments in parallel (default: 1)
- --parallel 0: Auto-detect GPU count and use all
- --gpu N: Pin all experiments to GPU N
- Backward compatible with CPU-only execution

Output Structure:
    results/{dataset}/proj{N}_edit{M}_{layer_mode}{L}_thresh{T}/
        ├── comparative_evaluation_edit_samples.csv
        ├── comparative_evaluation_projection_samples.csv
        ├── comparative_evaluation_test_set.csv
        └── comparative_evaluation_edit_discovery_set.csv

Usage Examples:
--------------
1. Quick dry run:
   uv run python scripts/param_search.py --dry-run

2. Single dataset, minimal search:
   uv run python scripts/param_search.py --datasets pathmnist \\
       --max-edits-range 30 --num-edit-layers-range 3 --max-experiments 5

3. Full search with multiple parameters:
   uv run python scripts/param_search.py --datasets pathmnist dermamnist \\
       --projection-samples-range 300 500 1000 \\
       --max-edits-range 10 30 50 \\
       --num-edit-layers-range 1 3 5

4. Multi-GPU parallel execution (8x 3090):
   uv run python scripts/param_search.py --datasets pathmnist --parallel 0

5. Single GPU execution:
   uv run python scripts/param_search.py --datasets pathmnist --gpu 0

6. Continue from specific experiment:
   uv run python scripts/param_search.py --continue-from 10
"""

import os
import sys
import subprocess
import argparse
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def detect_gpu_count() -> int:
    """Detect number of available GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter search for optimal model editing configuration (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # === Dataset Selection ===
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["pathmnist"],
        help="Datasets to search (default: pathmnist). "
             "MedMNIST: pathmnist, dermamnist, retinamnist, organamnist, bloodmnist, tissuemnist. "
             "Liver Fibrosis: liver4 (4-class), liver2s (2-class significant), liver2a (2-class any)."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom data path for liver fibrosis datasets (default: dataset/). "
             "Not needed for MedMNIST datasets which use ~/.medmnist/."
    )

    # === Search Space Configuration ===
    parser.add_argument(
        "--projection-samples-range",
        type=int,
        nargs="+",
        default=[500],
        help="Values for --projection-samples to search (default: [500])"
    )

    parser.add_argument(
        "--nullspace-threshold-range",
        type=float,
        nargs="+",
        default=[1e-2],
        help="Values for --nullspace-threshold to search (default: [1e-2])"
    )

    parser.add_argument(
        "--num-edit-layers-range",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Values for --num-edit-layers to search in ASTRA mode (default: [1,2,3,4,5])"
    )

    parser.add_argument(
        "--fixed-edit-layers",
        type=str,
        nargs="+",
        default=None,
        help="Fixed layer combinations to test, e.g. '9,10,11' '10,11' '11'. "
             "These are alternatives to ASTRA-based layer selection."
    )

    parser.add_argument(
        "--max-edits-range",
        type=str,
        nargs="+",
        default=["10", "20", "30", "50"],
        help="Values for --max-edits to search (default: [10,20,30,50], or 'all')"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Fixed value for --max-samples (locator stage, not searched) (default: 50)"
    )

    # === Output Configuration ===
    parser.add_argument(
        "--output-dir",
        type=str,
        default="param_search_results",
        help="Directory to save search summary (default: param_search_results)"
    )

    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable timestamp in output directory (allows overwriting previous results)"
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (used instead of timestamp)"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base directory for experiment results (default: results)"
    )

    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Base directory for experiment logs (default: logs)"
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Base directory for model checkpoints (default: checkpoints)"
    )

    # === Execution Control ===
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )

    parser.add_argument(
        "--continue-from",
        type=int,
        default=0,
        help="Continue from experiment N (skip first N experiments)"
    )

    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run"
    )

    # === GPU/Parallelism Configuration ===
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel experiments. 0 = auto (use all GPUs). Default: 1"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Pin all experiments to specific GPU ID. Default: None (auto)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout per experiment in seconds (default: 7200 = 2 hours)"
    )

    # === Baseline Configuration ===
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Disable running baseline comparisons (baseline1: retrain, baseline2: finetune-errors)"
    )

    parser.add_argument(
        "--baseline-epochs",
        type=int,
        default=10,
        help="Number of training epochs for baselines (default: 10)"
    )

    parser.add_argument(
        "--baseline-lr",
        type=float,
        default=1e-5,
        help="Learning rate for baseline2 finetune-on-errors (default: 1e-5)"
    )

    # === W&B Integration ===
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for all experiments"
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vit-model-editing",
        help="W&B project name (default: vit-model-editing)"
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team or username). Default: None (uses default entity)"
    )

    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags to add to W&B runs (e.g., --wandb-tags param-search pathmnist)"
    )

    return parser.parse_args()


def format_threshold(threshold: float) -> str:
    """Format threshold for directory naming (e.g., 1e-2 -> '1e-2')."""
    if threshold >= 1:
        return str(int(threshold))
    # Format as scientific notation without '+' sign
    exp = int(f"{threshold:.0e}".split("e")[1])
    mantissa = threshold / (10 ** exp)
    if mantissa == 1:
        return f"1e{exp}"
    return f"{mantissa:.0f}e{exp}"


def build_run_name(config: Dict[str, Any]) -> str:
    """Build structured run name from config."""
    dataset = config['dataset']
    proj = config['projection_samples']
    edit = config['max_edits']
    thresh = format_threshold(config['nullspace_threshold'])

    if config['mode'] == 'astra':
        layer_desc = f"astra{config['num_edit_layers']}"
    else:
        layers = config['edit_layers']
        layer_desc = f"fixed{'-'.join(map(str, layers))}"

    return f"{dataset}/proj{proj}_edit{edit}_{layer_desc}_thresh{thresh}"


def build_experiment_configs(args) -> List[Dict[str, Any]]:
    """Build list of experiment configurations to run."""
    configs = []

    def normalize_max_edits(value: str) -> Any:
        if isinstance(value, str) and value.lower() in {"all", "*"}:
            return "all"
        return int(value)

    for dataset in args.datasets:
        for proj_samples in args.projection_samples_range:
            for threshold in args.nullspace_threshold_range:
                for max_edits in args.max_edits_range:
                    max_edits = normalize_max_edits(max_edits)
                    # ASTRA-based experiments
                    astra_layers = [n for n in args.num_edit_layers_range if n > 0]
                    for num_layers in astra_layers:
                        configs.append({
                            'dataset': dataset,
                            'mode': 'astra',
                            'num_edit_layers': num_layers,
                            'edit_layers': None,
                            'projection_samples': proj_samples,
                            'nullspace_threshold': threshold,
                            'max_edits': max_edits,
                            'max_samples': args.max_samples,
                        })

                    # Fixed layer experiments
                    if args.fixed_edit_layers:
                        for layers_str in args.fixed_edit_layers:
                            layers = [int(x) for x in layers_str.split(',')]
                            configs.append({
                                'dataset': dataset,
                                'mode': 'fixed',
                                'num_edit_layers': None,
                                'edit_layers': layers,
                                'projection_samples': proj_samples,
                                'nullspace_threshold': threshold,
                                'max_edits': max_edits,
                                'max_samples': args.max_samples,
                            })

    return configs


def build_commands(config: Dict[str, Any], args, run_name: str) -> List[str]:
    """Build command line for --stage full (includes locate, edit, eval)."""
    cmd = [
        sys.executable,
        str(SRC_DIR / "main.py"),
        "--dataset", config['dataset'],
        "--run-name", run_name,
        "--max-samples", str(config['max_samples']),
        "--max-edits", str(config['max_edits']),
        "--projection-samples", str(config['projection_samples']),
        "--nullspace-threshold", str(config['nullspace_threshold']),
        "--checkpoint-dir", args.checkpoints_dir,
        "--log-dir", args.logs_dir,
        "--results-dir", args.results_dir,
        "--stage", "full",  # Use full pipeline (locate → edit → eval)
    ]

    # Add data-path for liver fibrosis datasets
    if config['dataset'].startswith('liver'):
        data_path = args.data_path if args.data_path else "dataset/"
        cmd.extend(["--data-path", data_path])

    # Layer selection mode
    if config['mode'] == 'astra':
        cmd.extend(["--num-edit-layers", str(config['num_edit_layers'])])
    else:
        cmd.append("--no-astra-layers")
        cmd.extend(["--edit-layers"] + [str(layer) for layer in config['edit_layers']])

    return cmd


def parse_metric_csv(csv_path: Path) -> Dict[str, Any]:
    """Parse a metric CSV file (format: metric,value,notes)."""
    metrics = {}
    if not csv_path.exists():
        return metrics

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get('metric', '')
                value = row.get('value', '')
                try:
                    # Try to parse as float
                    metrics[metric] = float(value)
                except (ValueError, TypeError):
                    metrics[metric] = value
    except Exception as e:
        print(f"Warning: Failed to parse {csv_path}: {e}")

    return metrics


def parse_results(results_dir: Path, log_dir: Path = None, run_name: str = None) -> Dict[str, Any]:
    """Parse results from a completed experiment.

    Args:
        results_dir: Path to the results directory (e.g., results/{run_name}/)
        log_dir: Optional path to logs directory to extract edit layers
        run_name: Optional run name to locate edit_log.csv
    """
    results = {}

    # Define CSV files to parse and their prefixes
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

    # Extract actual edited layers from edit_log.csv if log_dir provided
    if log_dir is not None and run_name is not None:
        edit_layers = extract_edit_layers(log_dir, run_name)
        if edit_layers is not None:
            results['actual_edit_layers'] = edit_layers

    return results


def extract_edit_layers(log_dir: Path, run_name: str) -> Optional[List[int]]:
    """Extract actual edited layers from edit_log.csv.

    Args:
        log_dir: Base logs directory (e.g., "logs")
        run_name: Run name (e.g., "pathmnist/proj500_edit30_astra3_thresh1e-2")

    Returns:
        List of unique layer indices that were edited, or None if not found
    """
    # Try to find edit_log.csv in log_dir/run_name/
    edit_log_path = log_dir / run_name / "edit_log.csv"

    if not edit_log_path.exists():
        # Try alternative location for head editing
        head_edit_log_path = log_dir / run_name / "head_edit_log.csv"
        if head_edit_log_path.exists():
            # Head editing doesn't have layer info in the same way
            return None
        return None

    try:
        with open(edit_log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            layers = set()
            for row in reader:
                if 'layer' in row:
                    try:
                        layers.add(int(row['layer']))
                    except (ValueError, TypeError):
                        pass
            if layers:
                return sorted(list(layers))
    except Exception as e:
        print(f"Warning: Failed to parse edit_log.csv: {e}")

    return None


def get_unique_baseline_keys(configs: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """Extract unique (dataset, max_edits) pairs that need baselines."""
    seen = set()
    keys = []
    for config in configs:
        key = (config['dataset'], config['max_edits'])
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def build_baseline_run_name(dataset: str, max_edits: int, baseline_type: str) -> str:
    """Build run name for a baseline experiment."""
    name = "retrain" if baseline_type == "baseline1" else "finetune_errors"
    return f"{dataset}/baseline_{name}_edit{max_edits}"


def run_baseline_worker(
    exp_idx: int,
    dataset: str,
    max_edits: int,
    baseline_type: str,
    args,
    gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Run a single baseline experiment (baseline1 or baseline2).

    Returns an experiment record with the same structure as alphaedit experiments,
    so baselines can appear as separate rows in the summary CSV.
    """
    run_name = build_baseline_run_name(dataset, max_edits, baseline_type)
    method = "baseline_retrain" if baseline_type == "baseline1" else "baseline_finetune"

    cmd = [
        sys.executable,
        str(SRC_DIR / "main.py"),
        "--stage", baseline_type,
        "--dataset", dataset,
        "--run-name", run_name,
        "--max-edits", str(max_edits),
        "--baseline-epochs", str(args.baseline_epochs),
        "--checkpoint-dir", args.checkpoints_dir,
        "--log-dir", args.logs_dir,
        "--results-dir", args.results_dir,
    ]

    # Add data-path for liver fibrosis datasets
    if dataset.startswith('liver'):
        data_path = args.data_path if args.data_path else "dataset/"
        cmd.extend(["--data-path", data_path])

    if baseline_type == "baseline2":
        cmd.extend(["--baseline-lr", str(args.baseline_lr)])

    # Create config dict similar to alphaedit experiments (with baseline-specific fields)
    config = {
        'dataset': dataset,
        'method': method,
        'mode': 'baseline',
        'num_edit_layers': None,
        'edit_layers': None,
        'projection_samples': None,
        'nullspace_threshold': None,
        'max_edits': max_edits,
        'max_samples': None,
        'baseline_epochs': args.baseline_epochs,
        'baseline_lr': args.baseline_lr if baseline_type == "baseline2" else None,
    }

    record = {
        'exp_idx': exp_idx,
        'run_name': run_name,
        'config': config,
        'command_edit': ' '.join(cmd),
        'command_eval': '',  # Baseline has single command
        'status': 'pending',
        'gpu_id': gpu_id,
        'start_time': None,
        'end_time': None,
        'duration_seconds': None,
        'edit_time_seconds': None,  # For baseline, this equals duration (training time)
        'results': {},
    }

    if args.dry_run:
        print(f"\n[DRY RUN] Baseline {exp_idx}: {run_name}")
        print(f"  GPU: {gpu_id if gpu_id is not None else 'auto'}")
        print(f"  Cmd: {' '.join(cmd)}")
        record['status'] = 'skipped'
        return record

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    bl_label = "RETRAIN" if baseline_type == "baseline1" else "FINETUNE-ERRORS"
    print(f"\n{'='*70}")
    print(f"BASELINE {exp_idx} ({bl_label}): {run_name}")
    print(f"{'='*70}")
    print(f"GPU: {gpu_id if gpu_id is not None else 'auto'}")

    record['start_time'] = datetime.now().isoformat()
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=args.timeout
        )

        if result.returncode != 0:
            print(f"[ERROR] Baseline {baseline_type} failed (rc={result.returncode})")
            print(f"STDERR (last 2000 chars): {result.stderr[-2000:]}")
            record['status'] = 'failed'
            record['error'] = result.stderr[-2000:]
        else:
            record['status'] = 'success'

    except subprocess.TimeoutExpired:
        record['status'] = 'timeout'
        print(f"[ERROR] Baseline timed out ({args.timeout}s)")
    except Exception as e:
        record['status'] = 'error'
        record['error'] = str(e)
        print(f"[ERROR] Exception: {e}")

    record['end_time'] = datetime.now().isoformat()
    record['duration_seconds'] = time.time() - start
    record['edit_time_seconds'] = record['duration_seconds']  # For baseline, training time = edit time

    # Parse results
    if record['status'] == 'success':
        results_path = Path(args.results_dir) / run_name
        if results_path.exists():
            record['results'] = parse_results(results_path)
            print("\nResults Summary:")
            for key in ['edit_fix_rate', 'test_accuracy_delta', 'proj_stability']:
                if key in record['results']:
                    val = record['results'][key]
                    if isinstance(val, float):
                        print(f"  {key}: {val:.4f}")
                    else:
                        print(f"  {key}: {val}")

    # Log to W&B if enabled
    if getattr(args, 'wandb', False) and not args.dry_run:
        _log_experiment_to_wandb(record, config, args)

    return record


def run_experiment_worker(
    exp_idx: int,
    config: Dict[str, Any],
    args,
    gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Worker function to run a single experiment using --stage full."""
    run_name = build_run_name(config)

    # Add method field to config for CSV output
    config['method'] = 'alphaedit'

    # Build single command for full pipeline
    cmd = build_commands(config, args, run_name)

    # Create experiment record
    experiment = {
        'exp_idx': exp_idx,
        'run_name': run_name,
        'config': config,
        'command': ' '.join(cmd),
        'status': 'pending',
        'gpu_id': gpu_id,
        'start_time': None,
        'end_time': None,
        'duration_seconds': None,
        'results': {},
    }

    if args.dry_run:
        print(f"\n[DRY RUN] Experiment {exp_idx}: {run_name}")
        print(f"  GPU: {gpu_id if gpu_id is not None else 'auto'}")
        print(f"  Cmd: {' '.join(cmd)}")
        experiment['status'] = 'skipped'
        return experiment

    # Setup environment for GPU
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_idx}: {run_name}")
    print(f"{'='*70}")
    print(f"GPU: {gpu_id if gpu_id is not None else 'auto'}")
    print(f"Config: {json.dumps(config, indent=2)}")

    experiment['start_time'] = datetime.now().isoformat()
    start = time.time()

    try:
        # Run full pipeline (locate → edit → eval)
        print("\n--- Running FULL pipeline (locate → edit → eval) ---")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=args.timeout
        )

        if result.returncode != 0:
            print(f"[ERROR] Pipeline failed with return code {result.returncode}")
            print(f"STDERR (last 2000 chars): {result.stderr[-2000:]}")
            experiment['status'] = 'failed'
            experiment['error'] = result.stderr[-2000:]
        else:
            experiment['status'] = 'success'
            print("Pipeline completed successfully")

    except subprocess.TimeoutExpired:
        experiment['status'] = 'timeout'
        print(f"[ERROR] Experiment timed out ({args.timeout}s)")
    except Exception as e:
        experiment['status'] = 'error'
        experiment['error'] = str(e)
        print(f"[ERROR] Exception: {e}")

    end = time.time()
    experiment['end_time'] = datetime.now().isoformat()
    experiment['duration_seconds'] = end - start

    # Parse results if successful
    if experiment['status'] == 'success':
        results_path = Path(args.results_dir) / run_name
        if results_path.exists():
            experiment['results'] = parse_results(
                results_path,
                log_dir=Path(args.logs_dir),
                run_name=run_name
            )
            print("\nResults Summary:")
            # Print key metrics
            for key in ['edit_fix_rate', 'test_accuracy_delta', 'proj_stability', 'actual_edit_layers']:
                if key in experiment['results']:
                    val = experiment['results'][key]
                    if isinstance(val, float):
                        print(f"  {key}: {val:.4f}")
                    else:
                        print(f"  {key}: {val}")

    # Log to W&B if enabled
    if getattr(args, 'wandb', False) and not args.dry_run:
        _log_experiment_to_wandb(experiment, config, args)

    return experiment


def _log_experiment_to_wandb(experiment: Dict[str, Any], config: Dict[str, Any], args) -> None:
    """Log a single experiment to Weights & Biases."""
    try:
        import wandb

        # Initialize run for this experiment
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment['run_name'],
            config=config,
            tags=args.wandb_tags + [config['dataset'], config['method']],
            reinit=True,  # Allow multiple runs in same process
        )

        # Log metrics
        results = experiment.get('results', {})
        metrics = {
            "edit/fix_rate": results.get("edit_fix_rate", 0),
            "edit/accuracy_delta": results.get("edit_accuracy_delta", 0),
            "test/accuracy_before": results.get("test_accuracy_orig", 0),
            "test/accuracy_after": results.get("test_accuracy_edit", 0),
            "test/accuracy_delta": results.get("test_accuracy_delta", 0),
            "test/stability": results.get("test_stability", 0),
            "proj/stability": results.get("proj_stability", 0),
            "proj/regression_rate": results.get("proj_regression_rate", 0),
            "timing/edit_seconds": experiment.get("edit_time_seconds", 0),
            "timing/total_seconds": experiment.get("duration_seconds", 0),
            "status": experiment.get("status", "unknown"),
        }
        wandb.log(metrics)

        # Set summary for easy comparison
        wandb.summary["test_accuracy_delta"] = results.get("test_accuracy_delta", 0)
        wandb.summary["status"] = experiment.get("status", "unknown")

        run.finish()

    except ImportError:
        print("[WARN] wandb not installed, skipping logging")
    except Exception as e:
        print(f"[WARN] Failed to log to wandb: {e}")


def save_summary_csv(experiments: List[Dict[str, Any]], output_path: Path):
    """Save experiment results to summary CSV."""
    if not experiments:
        return

    # Collect all possible columns
    config_keys = set()
    result_keys = set()
    for exp in experiments:
        config_keys.update(exp['config'].keys())
        result_keys.update(exp['results'].keys())

    # Build header
    base_columns = ['exp_idx', 'run_name', 'status', 'duration_seconds', 'gpu_id']
    config_columns = sorted(config_keys)
    result_columns = sorted(result_keys)
    all_columns = base_columns + config_columns + result_columns

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()

        for exp in experiments:
            row = {
                'exp_idx': exp['exp_idx'],
                'run_name': exp['run_name'],
                'status': exp['status'],
                'duration_seconds': exp.get('duration_seconds'),
                'gpu_id': exp.get('gpu_id'),
            }
            # Add config values
            for k in config_columns:
                val = exp['config'].get(k)
                if isinstance(val, list):
                    row[k] = ','.join(map(str, val))
                else:
                    row[k] = val
            # Add result values
            for k in result_columns:
                val = exp['results'].get(k)
                # Convert list to comma-separated string (for actual_edit_layers)
                if isinstance(val, list):
                    row[k] = ','.join(map(str, val))
                else:
                    row[k] = val

            writer.writerow(row)

    print(f"\nSummary saved to: {output_path}")


def main():
    args = parse_args()

    # Create output directory with versioning (timestamp by default)
    output_dir = Path(args.output_dir)

    # Apply run name or timestamp suffix (timestamp is default)
    if args.run_name:
        suffix = args.run_name
    elif not args.no_timestamp:
        # Timestamp is now the DEFAULT behavior
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        suffix = None

    if suffix:
        output_dir = output_dir / suffix
        print(f"Run identifier: {suffix}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build experiment configs
    configs = build_experiment_configs(args)
    total_experiments = len(configs)

    # Determine parallelism
    num_gpus = detect_gpu_count()
    if args.gpu is not None:
        # Pin to specific GPU
        parallel = 1
        gpu_ids = [args.gpu]
    elif args.parallel == 0:
        # Auto: use all GPUs
        parallel = max(1, num_gpus)
        gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
    else:
        parallel = args.parallel
        gpu_ids = list(range(min(parallel, num_gpus))) if num_gpus > 0 else [None] * parallel

    print("=" * 70)
    print("PARAMETER SEARCH FOR MODEL EDITING (v2)")
    print("=" * 70)
    print(f"Datasets: {args.datasets}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")
    print(f"Detected GPUs: {num_gpus}")
    print(f"Parallel workers: {parallel}")
    print(f"GPU IDs: {gpu_ids}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No experiments will be executed]")

    # Show search space
    print("\nSearch Space:")
    print(f"  --datasets: {args.datasets}")
    print(f"  --projection-samples-range: {args.projection_samples_range}")
    print(f"  --nullspace-threshold-range: {args.nullspace_threshold_range}")
    print(f"  --num-edit-layers-range: {args.num_edit_layers_range}")
    if args.fixed_edit_layers:
        print(f"  --fixed-edit-layers: {args.fixed_edit_layers}")
    print(f"  --max-edits-range: {args.max_edits_range}")
    print(f"  --max-samples (fixed): {args.max_samples}")
    print(f"  --no-baselines: {args.no_baselines}")
    if not args.no_baselines:
        print(f"  --baseline-epochs: {args.baseline_epochs}")
        print(f"  --baseline-lr: {args.baseline_lr}")

    # Apply filters
    start_idx = args.continue_from
    if args.max_experiments:
        end_idx = min(start_idx + args.max_experiments, total_experiments)
    else:
        end_idx = total_experiments

    configs_to_run = configs[start_idx:end_idx]
    print(f"\nRunning experiments {start_idx} to {end_idx - 1} ({len(configs_to_run)} experiments)")

    # === Phase 1: Run Baselines (deduplicated by dataset + max_edits) ===
    # Baselines are added as separate rows in the experiments list
    experiments = []
    next_exp_idx = 0  # Track experiment index for both baselines and alphaedit

    if not args.no_baselines:
        baseline_keys = get_unique_baseline_keys(configs_to_run)
        total_baseline_runs = len(baseline_keys) * 2  # baseline1 + baseline2 for each key
        print(f"\n{'='*70}")
        print(f"PHASE 1: RUNNING BASELINES ({len(baseline_keys)} unique (dataset, max_edits) pairs, {total_baseline_runs} runs)")
        print(f"{'='*70}")

        for i, (dataset, max_edits) in enumerate(baseline_keys):
            gpu_id = gpu_ids[0] if gpu_ids and gpu_ids[0] is not None else None

            # Run baseline1 (retrain)
            print(f"\n[{i+1}/{len(baseline_keys)}] Baseline for {dataset}, max_edits={max_edits}")
            bl1 = run_baseline_worker(next_exp_idx, dataset, max_edits, "baseline1", args, gpu_id)
            experiments.append(bl1)
            next_exp_idx += 1

            # Run baseline2 (finetune-errors)
            bl2 = run_baseline_worker(next_exp_idx, dataset, max_edits, "baseline2", args, gpu_id)
            experiments.append(bl2)
            next_exp_idx += 1

            # Save intermediate results after each baseline pair
            summary_path = output_dir / f"param_search_summary_{args.datasets[0]}.csv"
            save_summary_csv(experiments, summary_path)

        print(f"\nBaseline phase complete: {len(experiments)} baseline runs")

    # === Phase 2: Run AlphaEdit Experiments ===
    print(f"\n{'='*70}")
    print(f"PHASE 2: RUNNING ALPHAEDIT EXPERIMENTS ({len(configs_to_run)} experiments)")
    print(f"{'='*70}")

    if parallel == 1:
        # Sequential execution
        for i, config in enumerate(configs_to_run):
            exp_idx = next_exp_idx + i
            gpu_id = gpu_ids[0] if gpu_ids and gpu_ids[0] is not None else None
            experiment = run_experiment_worker(exp_idx, config, args, gpu_id)

            experiments.append(experiment)

            # Save intermediate results
            summary_path = output_dir / f"param_search_summary_{args.datasets[0]}.csv"
            save_summary_csv(experiments, summary_path)

            # Save detailed JSON
            json_path = output_dir / f"param_search_details_{args.datasets[0]}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(experiments, f, indent=2)

    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for i, config in enumerate(configs_to_run):
                exp_idx = next_exp_idx + i
                # Round-robin GPU assignment
                gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else None
                future = executor.submit(
                    run_experiment_worker, exp_idx, config, args, gpu_id
                )
                futures[future] = (exp_idx, i)  # Store both exp_idx and config index

            for future in as_completed(futures):
                exp_idx, config_idx = futures[future]
                try:
                    experiment = future.result()

                    experiments.append(experiment)
                except Exception as e:
                    print(f"[ERROR] Experiment {exp_idx} raised exception: {e}")
                    experiments.append({
                        'exp_idx': exp_idx,
                        'status': 'exception',
                        'error': str(e),
                        'config': configs_to_run[config_idx],
                        'results': {}
                    })

                # Save intermediate results
                experiments_sorted = sorted(experiments, key=lambda x: x['exp_idx'])
                summary_path = output_dir / f"param_search_summary_{args.datasets[0]}.csv"
                save_summary_csv(experiments_sorted, summary_path)

                json_path = output_dir / f"param_search_details_{args.datasets[0]}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(experiments_sorted, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("PARAMETER SEARCH COMPLETE")
    print("=" * 70)

    successful = sum(1 for e in experiments if e['status'] == 'success')
    failed = sum(1 for e in experiments if e['status'].startswith('failed'))
    print(f"Successful: {successful}/{len(experiments)}")
    print(f"Failed: {failed}/{len(experiments)}")

    # Find best configuration based on test accuracy improvement
    if successful > 0:
        best_exp = None
        best_metric = float('-inf')

        for exp in experiments:
            if exp['status'] == 'success' and exp['results']:
                # Primary metric: test accuracy improvement
                test_delta = exp['results'].get('test_accuracy_delta', 0)
                if test_delta > best_metric:
                    best_metric = test_delta
                    best_exp = exp

        if best_exp:
            print("\nBest Configuration (by test accuracy delta):")
            print(f"  Experiment: {best_exp['run_name']}")
            print(f"  Config: {json.dumps(best_exp['config'], indent=4)}")
            print(f"  Test Accuracy Delta: {best_metric * 100:+.2f}%")

            # Show other key metrics
            for key in ['proj_stability', 'edit_fix_rate']:
                if key in best_exp['results']:
                    print(f"  {key}: {best_exp['results'][key] * 100:.1f}%")

    print(f"\nResults saved to: {output_dir}")
    print(f"  Summary CSV: {output_dir}/param_search_summary_{args.datasets[0]}.csv")
    print(f"  Details JSON: {output_dir}/param_search_details_{args.datasets[0]}.json")


if __name__ == "__main__":
    main()
