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
        type=int,
        nargs="+",
        default=[10, 20, 30, 50],
        help="Values for --max-edits to search (default: [10,20,30,50])"
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

    for dataset in args.datasets:
        for proj_samples in args.projection_samples_range:
            for threshold in args.nullspace_threshold_range:
                for max_edits in args.max_edits_range:
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


def build_commands(config: Dict[str, Any], args, run_name: str) -> Tuple[List[str], List[str]]:
    """Build command lines for edit and eval stages."""
    base_cmd = [
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
    ]

    # Add data-path for liver fibrosis datasets
    if config['dataset'].startswith('liver'):
        data_path = args.data_path if args.data_path else "dataset/"
        base_cmd.extend(["--data-path", data_path])

    # Edit stage command
    cmd_edit = base_cmd + ["--stage", "edit"]

    if config['mode'] == 'astra':
        cmd_edit.extend(["--num-edit-layers", str(config['num_edit_layers'])])
    else:
        cmd_edit.append("--no-astra-layers")
        cmd_edit.extend(["--edit-layers"] + [str(layer) for layer in config['edit_layers']])

    # Eval stage command
    cmd_eval = base_cmd + ["--stage", "eval"]

    return cmd_edit, cmd_eval


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


def parse_results(results_dir: Path) -> Dict[str, Any]:
    """Parse results from a completed experiment."""
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

    return results


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

    return record


def run_experiment_worker(
    exp_idx: int,
    config: Dict[str, Any],
    args,
    gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Worker function to run a single experiment."""
    run_name = build_run_name(config)

    # Add method field to config for CSV output
    config['method'] = 'alphaedit'

    # Build commands
    cmd_edit, cmd_eval = build_commands(config, args, run_name)

    # Create experiment record
    experiment = {
        'exp_idx': exp_idx,
        'run_name': run_name,
        'config': config,
        'command_edit': ' '.join(cmd_edit),
        'command_eval': ' '.join(cmd_eval),
        'status': 'pending',
        'gpu_id': gpu_id,
        'start_time': None,
        'end_time': None,
        'duration_seconds': None,
        'edit_time_seconds': None,  # Time for edit stage only (excluding eval)
        'results': {},
    }

    if args.dry_run:
        print(f"\n[DRY RUN] Experiment {exp_idx}: {run_name}")
        print(f"  GPU: {gpu_id if gpu_id is not None else 'auto'}")
        print(f"  Edit: {' '.join(cmd_edit)}")
        print(f"  Eval: {' '.join(cmd_eval)}")
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
        # Run edit stage
        print("\n--- Running EDIT stage ---")
        edit_start = time.time()
        result_edit = subprocess.run(
            cmd_edit,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=args.timeout
        )
        edit_end = time.time()
        experiment['edit_time_seconds'] = edit_end - edit_start

        if result_edit.returncode != 0:
            print(f"[ERROR] Edit stage failed with return code {result_edit.returncode}")
            print(f"STDERR (last 2000 chars): {result_edit.stderr[-2000:]}")
            experiment['status'] = 'failed_edit'
            experiment['error'] = result_edit.stderr[-2000:]
        else:
            print(f"Edit stage completed in {experiment['edit_time_seconds']:.1f}s")
            # Run eval stage
            print("\n--- Running EVAL stage ---")
            result_eval = subprocess.run(
                cmd_eval,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                env=env,
                timeout=args.timeout
            )

            if result_eval.returncode != 0:
                print(f"[ERROR] Eval stage failed with return code {result_eval.returncode}")
                print(f"STDERR (last 2000 chars): {result_eval.stderr[-2000:]}")
                experiment['status'] = 'failed_eval'
                experiment['error'] = result_eval.stderr[-2000:]
            else:
                experiment['status'] = 'success'

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
            experiment['results'] = parse_results(results_path)
            print("\nResults Summary:")
            # Print key metrics
            for key in ['edit_fix_rate', 'test_accuracy_delta', 'proj_stability']:
                if key in experiment['results']:
                    val = experiment['results'][key]
                    if isinstance(val, float):
                        print(f"  {key}: {val:.4f}")
                    else:
                        print(f"  {key}: {val}")

    return experiment


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
    base_columns = ['exp_idx', 'run_name', 'status', 'duration_seconds', 'edit_time_seconds', 'gpu_id']
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
                'edit_time_seconds': exp.get('edit_time_seconds'),
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
                row[k] = exp['results'].get(k)

            writer.writerow(row)

    print(f"\nSummary saved to: {output_path}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
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
    print(f"\nSearch Space:")
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
            print(f"\nBest Configuration (by test accuracy delta):")
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
