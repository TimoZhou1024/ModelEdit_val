#!/usr/bin/env python
"""
Parameter Search Script for Model Editing

This script explores different parameter combinations to find optimal editing settings.
It systematically varies:
- --num-edit-layers (ASTRA-based layer selection)
- --edit-layers (fixed layer specification, mutually exclusive with num-edit-layers)
- --max-samples (samples for layer analysis)
- --max-edits (number of edits to apply)

Results are saved to a summary CSV for analysis.

Usage Examples:
--------------
1. Quick test run (dry run to see what would be executed):
   uv run python scripts/param_search.py --dry-run --max-experiments 5

2. Search with ASTRA-based layer selection (default):
   uv run python scripts/param_search.py --dataset pathmnist --max-experiments 20

3. Search with fixed layer combinations:
   uv run python scripts/param_search.py --dataset pathmnist \\
       --fixed-edit-layers "11" "10,11" "9,10,11" "8,9,10,11" \\
       --num-edit-layers-range 0

4. Full search (ASTRA + fixed layers):
   uv run python scripts/param_search.py --dataset pathmnist \\
       --num-edit-layers-range 1 2 3 4 5 \\
       --fixed-edit-layers "11" "10,11" "9,10,11" \\
       --max-samples-range 30 50 100 \\
       --max-edits-range 10 20 30 50

5. Continue from a specific experiment:
   uv run python scripts/param_search.py --continue-from 10

Output Files:
------------
- param_search_results/param_search_summary_{dataset}.csv  - Summary of all experiments
- param_search_results/param_search_details_{dataset}.json - Detailed experiment records
- results/exp_XXX/  - Individual experiment results
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
from itertools import product
from typing import List, Dict, Any, Optional

# Add src to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter search for optimal model editing configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="pathmnist",
        help="MedMNIST dataset to use (default: pathmnist)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="param_search_results",
        help="Directory to save search results (default: param_search_results)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )

    parser.add_argument(
        "--continue-from",
        type=int,
        default=0,
        help="Continue from experiment N (skip first N-1 experiments)"
    )

    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run"
    )

    # Search space configuration
    parser.add_argument(
        "--num-edit-layers-range",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Values for --num-edit-layers to search (ASTRA mode)"
    )

    parser.add_argument(
        "--fixed-edit-layers",
        type=str,
        nargs="+",
        default=None,
        help="Fixed layer combinations to test, e.g. '9,10,11' '10,11' '11'"
    )

    parser.add_argument(
        "--max-samples-range",
        type=int,
        nargs="+",
        default=[30, 50, 100],
        help="Values for --max-samples to search"
    )

    parser.add_argument(
        "--max-edits-range",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50],
        help="Values for --max-edits to search"
    )

    return parser.parse_args()


def build_experiment_configs(args) -> List[Dict[str, Any]]:
    """Build list of experiment configurations to run."""
    configs = []

    # ASTRA-based experiments (using --num-edit-layers)
    # Skip if num_edit_layers_range only contains 0 (used to disable ASTRA mode)
    astra_layers = [n for n in args.num_edit_layers_range if n > 0]
    for num_layers in astra_layers:
        for max_samples in args.max_samples_range:
            for max_edits in args.max_edits_range:
                config = {
                    'mode': 'astra',
                    'num_edit_layers': num_layers,
                    'edit_layers': None,
                    'max_samples': max_samples,
                    'max_edits': max_edits,
                }
                configs.append(config)

    # Fixed layer experiments (using --edit-layers)
    if args.fixed_edit_layers:
        for layers_str in args.fixed_edit_layers:
            layers = [int(x) for x in layers_str.split(',')]
            for max_samples in args.max_samples_range:
                for max_edits in args.max_edits_range:
                    config = {
                        'mode': 'fixed',
                        'num_edit_layers': None,
                        'edit_layers': layers,
                        'max_samples': max_samples,
                        'max_edits': max_edits,
                    }
                    configs.append(config)

    return configs


def build_command(config: Dict[str, Any], args, run_name: str) -> List[str]:
    """Build command line for running main.py with given config."""
    cmd = [
        sys.executable,
        str(SRC_DIR / "main.py"),
        "--stage", "edit",
        "--dataset", args.dataset,
        "--run-name", run_name,
        "--max-samples", str(config['max_samples']),
        "--max-edits", str(config['max_edits']),
    ]

    if config['mode'] == 'astra':
        # Use ASTRA-based layer selection
        cmd.extend(["--num-edit-layers", str(config['num_edit_layers'])])
        # Ensure ASTRA is enabled (default, but be explicit)
        cmd.append("--use-astra-layers")
    else:
        # Use fixed layer specification
        cmd.append("--no-astra-layers")
        cmd.extend(["--edit-layers"] + [str(l) for l in config['edit_layers']])

    return cmd


def parse_results(results_dir: Path) -> Dict[str, Any]:
    """Parse results from a completed experiment."""
    results = {}

    # Parse test set comparative evaluation
    test_csv = results_dir / "comparative_evaluation_test_set.csv"
    if test_csv.exists():
        with open(test_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row['metric']
                value = row['value']
                try:
                    value = float(value)
                except ValueError:
                    pass
                results[f'test_{metric}'] = value

    # Parse edit discovery set comparative evaluation
    discovery_csv = results_dir / "comparative_evaluation_edit_discovery_set.csv"
    if discovery_csv.exists():
        with open(discovery_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row['metric']
                value = row['value']
                try:
                    value = float(value)
                except ValueError:
                    pass
                results[f'discovery_{metric}'] = value

    # Parse projection samples (knowledge preservation)
    proj_csv = results_dir / "comparative_evaluation_projection_samples.csv"
    if proj_csv.exists():
        with open(proj_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row['metric']
                value = row['value']
                try:
                    value = float(value)
                except ValueError:
                    pass
                results[f'proj_{metric}'] = value

    # Parse edit samples
    edit_csv = results_dir / "comparative_evaluation_edit_samples.csv"
    if edit_csv.exists():
        with open(edit_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row['metric']
                value = row['value']
                try:
                    value = float(value)
                except ValueError:
                    pass
                results[f'edit_{metric}'] = value

    return results


def run_experiment(config: Dict[str, Any], args, exp_idx: int) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    run_name = f"exp_{exp_idx:03d}"

    # Build command
    cmd = build_command(config, args, run_name)

    # Create experiment record
    experiment = {
        'exp_idx': exp_idx,
        'run_name': run_name,
        'config': config,
        'command': ' '.join(cmd),
        'status': 'pending',
        'start_time': None,
        'end_time': None,
        'duration_seconds': None,
        'results': {},
    }

    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_idx}: {run_name}")
    print(f"{'='*70}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("[DRY RUN] Skipping execution")
        experiment['status'] = 'skipped'
        return experiment

    # Run experiment
    experiment['start_time'] = datetime.now().isoformat()
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=3600  # 1 hour timeout
        )

        experiment['status'] = 'success' if result.returncode == 0 else 'failed'
        experiment['returncode'] = result.returncode

        if result.returncode != 0:
            print(f"[ERROR] Experiment failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[-2000:]}")  # Last 2000 chars

    except subprocess.TimeoutExpired:
        experiment['status'] = 'timeout'
        print("[ERROR] Experiment timed out (1 hour)")
    except Exception as e:
        experiment['status'] = 'error'
        experiment['error'] = str(e)
        print(f"[ERROR] Exception: {e}")

    end = time.time()
    experiment['end_time'] = datetime.now().isoformat()
    experiment['duration_seconds'] = end - start

    # Parse results if successful
    if experiment['status'] == 'success':
        results_dir = PROJECT_ROOT / "results" / run_name
        if results_dir.exists():
            experiment['results'] = parse_results(results_dir)
            print(f"\nResults Summary:")
            for k, v in experiment['results'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

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
    base_columns = ['exp_idx', 'run_name', 'status', 'duration_seconds']
    config_columns = sorted(config_keys)
    result_columns = sorted(result_keys)
    all_columns = base_columns + config_columns + result_columns

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()

        for exp in experiments:
            row = {
                'exp_idx': exp['exp_idx'],
                'run_name': exp['run_name'],
                'status': exp['status'],
                'duration_seconds': exp.get('duration_seconds'),
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

    print("="*70)
    print("PARAMETER SEARCH FOR MODEL EDITING")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No experiments will be executed]")

    # Show search space
    print(f"\nSearch Space:")
    print(f"  --num-edit-layers: {args.num_edit_layers_range}")
    if args.fixed_edit_layers:
        print(f"  --edit-layers: {args.fixed_edit_layers}")
    print(f"  --max-samples: {args.max_samples_range}")
    print(f"  --max-edits: {args.max_edits_range}")

    # Apply filters
    start_idx = args.continue_from
    if args.max_experiments:
        end_idx = min(start_idx + args.max_experiments, total_experiments)
    else:
        end_idx = total_experiments

    configs_to_run = configs[start_idx:end_idx]
    print(f"\nRunning experiments {start_idx} to {end_idx-1} ({len(configs_to_run)} experiments)")

    # Run experiments
    experiments = []
    for i, config in enumerate(configs_to_run):
        exp_idx = start_idx + i
        experiment = run_experiment(config, args, exp_idx)
        experiments.append(experiment)

        # Save intermediate results
        summary_path = output_dir / f"param_search_summary_{args.dataset}.csv"
        save_summary_csv(experiments, summary_path)

        # Save detailed JSON
        json_path = output_dir / f"param_search_details_{args.dataset}.json"
        with open(json_path, 'w') as f:
            json.dump(experiments, f, indent=2)

    # Final summary
    print("\n" + "="*70)
    print("PARAMETER SEARCH COMPLETE")
    print("="*70)

    successful = sum(1 for e in experiments if e['status'] == 'success')
    failed = sum(1 for e in experiments if e['status'] == 'failed')
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
            print(f"  Test Accuracy Delta: {best_metric*100:+.2f}%")

            # Show other metrics
            if 'proj_stability' in best_exp['results']:
                print(f"  Projection Stability: {best_exp['results']['proj_stability']*100:.1f}%")
            if 'edit_fix_rate' in best_exp['results']:
                print(f"  Edit Fix Rate: {best_exp['results']['edit_fix_rate']*100:.1f}%")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
