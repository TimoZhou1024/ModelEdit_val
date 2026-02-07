#!/usr/bin/env python
"""
Results Collection Script for Model Editing Experiments

This script scans completed experiment directories, extracts core metrics from
the evaluation CSV files, and generates a unified summary table.

Core Metrics Extracted:
-----------------------
Edit Performance:
  - edit_total_wrong: Total error samples used for editing
  - edit_num_fixed: Samples corrected after editing
  - edit_fix_ratio: edit_num_fixed / edit_total_wrong

Test Performance:
  - test_total: Total samples in test set
  - test_acc_before: Accuracy before editing
  - test_acc_after: Accuracy after editing
  - test_acc_delta: Accuracy change (after - before)
  - test_correct_before: Correct predictions before editing
  - test_correct_after: Correct predictions after editing
  - test_errors_before: Errors before editing
  - test_errors_after: Errors after editing
  - test_correct_to_wrong: Regressions (correct -> wrong)
  - test_wrong_to_correct: Fixes (wrong -> correct)

Projection Stability:
  - proj_stability: % of originally correct samples still correct
  - proj_regression_rate: % of originally correct samples broken

Usage Examples:
--------------
1. Basic collection:
   uv run python scripts/collect_results.py

2. Specify results directory:
   uv run python scripts/collect_results.py --results-dir results

3. Filter by dataset:
   uv run python scripts/collect_results.py --datasets pathmnist dermamnist

4. Custom output:
   uv run python scripts/collect_results.py --output my_metrics.csv
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect and aggregate metrics from model editing experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base directory containing experiment results (default: results)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="param_search_results/core_metrics.csv",
        help="Output CSV file path (default: param_search_results/core_metrics.csv)"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific datasets (default: all)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    return parser.parse_args()


def parse_run_name(run_name: str) -> Dict[str, Any]:
    """
    Parse structured run name to extract configuration parameters.

    Expected formats:
        {dataset}/proj{N}_edit{M}_astra{L}_thresh{T}          -> AlphaEdit (astra)
        {dataset}/proj{N}_edit{M}_fixed{L1-L2-L3}_thresh{T}   -> AlphaEdit (fixed)
        {dataset}/baseline_retrain_edit{M}                     -> Baseline1 (retrain)
        {dataset}/baseline_finetune_errors_edit{M}             -> Baseline2 (finetune)

    Returns dict with: dataset, method, mode, projection_samples, max_edits,
                       num_edit_layers or edit_layers, nullspace_threshold
    """
    config = {}

    # Split by /
    parts = run_name.split("/")
    if len(parts) >= 2:
        config['dataset'] = parts[0]
        config_str = parts[1]
    else:
        # Legacy format or flat structure
        config['dataset'] = 'unknown'
        config_str = run_name

    # Check for baseline patterns first
    baseline_retrain = re.search(r'baseline_retrain_edit(\d+)', config_str)
    baseline_finetune = re.search(r'baseline_finetune_errors_edit(\d+)', config_str)

    if baseline_retrain:
        config['method'] = 'baseline_retrain'
        config['mode'] = 'baseline'
        config['max_edits'] = int(baseline_retrain.group(1))
        config['num_edit_layers'] = None
        config['edit_layers'] = None
        config['projection_samples'] = None
        config['nullspace_threshold'] = None
        return config

    if baseline_finetune:
        config['method'] = 'baseline_finetune'
        config['mode'] = 'baseline'
        config['max_edits'] = int(baseline_finetune.group(1))
        config['num_edit_layers'] = None
        config['edit_layers'] = None
        config['projection_samples'] = None
        config['nullspace_threshold'] = None
        return config

    # AlphaEdit experiment
    config['method'] = 'alphaedit'

    # Parse projection samples
    proj_match = re.search(r'proj(\d+)', config_str)
    if proj_match:
        config['projection_samples'] = int(proj_match.group(1))

    # Parse max edits
    edit_match = re.search(r'edit(\d+)', config_str)
    if edit_match:
        config['max_edits'] = int(edit_match.group(1))

    # Parse layer configuration (astra or fixed)
    astra_match = re.search(r'astra(\d+)', config_str)
    fixed_match = re.search(r'fixed([\d-]+)', config_str)

    if astra_match:
        config['mode'] = 'astra'
        config['num_edit_layers'] = int(astra_match.group(1))
        config['edit_layers'] = None
    elif fixed_match:
        config['mode'] = 'fixed'
        layers_str = fixed_match.group(1)
        config['edit_layers'] = [int(x) for x in layers_str.split('-')]
        config['num_edit_layers'] = len(config['edit_layers'])
    else:
        config['mode'] = 'unknown'

    # Parse threshold
    thresh_match = re.search(r'thresh([\d.e+-]+)', config_str)
    if thresh_match:
        config['nullspace_threshold'] = float(thresh_match.group(1))

    return config


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
                    metrics[metric] = float(value)
                except (ValueError, TypeError):
                    metrics[metric] = value
    except Exception as e:
        print(f"Warning: Failed to parse {csv_path}: {e}")

    return metrics


def extract_core_metrics(results_dir: Path, log_dir: Path = None, run_name: str = None) -> Dict[str, Any]:
    """
    Extract core metrics from a single experiment's results directory.

    Args:
        results_dir: Path to the results directory
        log_dir: Optional path to logs directory to extract edit layers
        run_name: Optional run name to locate edit_log.csv

    Returns a dict with standardized metric names.
    """
    metrics = {}

    # === Edit Samples Metrics ===
    edit_csv = results_dir / "comparative_evaluation_edit_samples.csv"
    if edit_csv.exists():
        edit_data = parse_metric_csv(edit_csv)
        metrics['edit_total_wrong'] = edit_data.get('num_total')
        metrics['edit_num_fixed'] = edit_data.get('num_fixed')
        metrics['edit_fix_ratio'] = edit_data.get('fix_rate')
        metrics['edit_num_broken'] = edit_data.get('num_broken')
        metrics['edit_acc_before'] = edit_data.get('accuracy_before')
        metrics['edit_acc_after'] = edit_data.get('accuracy_after')

    # === Projection Samples Metrics (FT-Train stability) ===
    proj_csv = results_dir / "comparative_evaluation_projection_samples.csv"
    if proj_csv.exists():
        proj_data = parse_metric_csv(proj_csv)
        metrics['proj_total'] = proj_data.get('num_total')
        metrics['proj_acc_before'] = proj_data.get('accuracy_before')
        metrics['proj_acc_after'] = proj_data.get('accuracy_after')
        metrics['proj_stability'] = proj_data.get('stability')
        metrics['proj_regression_rate'] = proj_data.get('regression_rate')
        metrics['proj_num_broken'] = proj_data.get('num_broken')
        metrics['proj_num_fixed'] = proj_data.get('num_fixed')

    # === Test Set Metrics ===
    test_csv = results_dir / "comparative_evaluation_test_set.csv"
    if test_csv.exists():
        test_data = parse_metric_csv(test_csv)

        # Map from CSV metric names to our standard names
        # The CSV has: n_total, accuracy_orig, accuracy_edit, accuracy_delta,
        #              n_correct_orig, n_correct_edit, n_error_orig, n_regressed, n_fixed
        metrics['test_total'] = test_data.get('n_total')
        metrics['test_acc_before'] = test_data.get('accuracy_orig')
        metrics['test_acc_after'] = test_data.get('accuracy_edit')
        metrics['test_acc_delta'] = test_data.get('accuracy_delta')
        metrics['test_correct_before'] = test_data.get('n_correct_orig')
        metrics['test_correct_after'] = test_data.get('n_correct_edit')
        metrics['test_errors_before'] = test_data.get('n_error_orig')
        metrics['test_correct_to_wrong'] = test_data.get('n_regressed')
        metrics['test_wrong_to_correct'] = test_data.get('n_fixed')

        # Calculate errors_after if not directly available
        if metrics.get('test_total') is not None and metrics.get('test_correct_after') is not None:
            metrics['test_errors_after'] = metrics['test_total'] - metrics['test_correct_after']

    # === Edit Discovery Set Metrics ===
    discovery_csv = results_dir / "comparative_evaluation_edit_discovery_set.csv"
    if discovery_csv.exists():
        disc_data = parse_metric_csv(discovery_csv)
        metrics['discovery_total'] = disc_data.get('n_total')
        metrics['discovery_acc_before'] = disc_data.get('accuracy_orig')
        metrics['discovery_acc_after'] = disc_data.get('accuracy_edit')
        metrics['discovery_acc_delta'] = disc_data.get('accuracy_delta')

    # === Timing Metrics ===
    timing_csv = results_dir / "timing.csv"
    if timing_csv.exists():
        timing_data = parse_metric_csv(timing_csv)
        metrics['duration_seconds'] = timing_data.get('duration_seconds')
        metrics['edit_seconds'] = timing_data.get('edit_seconds')

    # === Extract actual edited layers from edit_log.csv ===
    if log_dir is not None and run_name is not None:
        edit_layers = extract_edit_layers_from_log(log_dir, run_name)
        if edit_layers is not None:
            metrics['actual_edit_layers'] = edit_layers

    return metrics


def extract_edit_layers_from_log(log_dir: Path, run_name: str) -> Optional[List[int]]:
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


def find_experiment_dirs(results_dir: Path, datasets: Optional[List[str]] = None) -> List[Path]:
    """
    Find all experiment directories containing evaluation results.

    Handles:
    - Structured AlphaEdit: results/{dataset}/proj{N}_edit{M}_astra{L}_thresh{T}/
    - Baseline: results/{dataset}/baseline_retrain_edit{M}/
    - Baseline: results/{dataset}/baseline_finetune_errors_edit{M}/
    - Flat layout: results/{config}/
    """
    experiment_dirs = []

    if not results_dir.exists():
        return experiment_dirs

    # Check for structured layout: results/{dataset}/{config}/
    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Filter by dataset if specified
        if datasets and dataset_dir.name not in datasets:
            continue

        for config_dir in dataset_dir.iterdir():
            if not config_dir.is_dir():
                continue

            # Check if this directory has evaluation CSVs
            # AlphaEdit uses comparative_evaluation_edit_samples.csv
            # Baselines use baseline_*_summary.csv
            has_alphaedit = (config_dir / "comparative_evaluation_edit_samples.csv").exists()
            has_baseline = any(
                f.name.startswith("baseline_") and f.name.endswith("_summary.csv")
                for f in config_dir.iterdir() if f.is_file()
            )

            if has_alphaedit or has_baseline:
                experiment_dirs.append(config_dir)

    # Also check for flat layout: results/{config}/
    for config_dir in results_dir.iterdir():
        if not config_dir.is_dir():
            continue

        # Skip if it's a dataset directory (already processed above)
        if any((config_dir / d).is_dir() for d in config_dir.iterdir() if d.is_dir()):
            continue

        has_alphaedit = (config_dir / "comparative_evaluation_edit_samples.csv").exists()
        has_baseline = any(
            f.name.startswith("baseline_") and f.name.endswith("_summary.csv")
            for f in config_dir.iterdir() if f.is_file()
        )

        if has_alphaedit or has_baseline:
            experiment_dirs.append(config_dir)

    return experiment_dirs


def compute_run_name(results_dir: Path, base_dir: Path) -> str:
    """Compute run name from directory path relative to base."""
    try:
        rel_path = results_dir.relative_to(base_dir)
        return str(rel_path).replace("\\", "/")
    except ValueError:
        return results_dir.name


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    print("=" * 70)
    print("RESULTS COLLECTION FOR MODEL EDITING EXPERIMENTS")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Output file: {output_path}")

    if args.datasets:
        print(f"Filtering datasets: {args.datasets}")

    # Find experiment directories
    experiment_dirs = find_experiment_dirs(results_dir, args.datasets)
    print(f"\nFound {len(experiment_dirs)} experiment directories")

    if not experiment_dirs:
        print("No experiments found. Exiting.")
        return

    # Collect metrics from all experiments
    all_metrics = []

    # Default logs directory
    log_dir = Path("logs")

    for exp_dir in sorted(experiment_dirs):
        run_name = compute_run_name(exp_dir, results_dir)

        if args.verbose:
            print(f"  Processing: {run_name}")

        # Parse configuration from run name
        config = parse_run_name(run_name)

        # Extract metrics (pass log_dir and run_name for actual_edit_layers)
        metrics = extract_core_metrics(exp_dir, log_dir=log_dir, run_name=run_name)

        # Combine config and metrics
        row = {
            'run_name': run_name,
            **config,
            **metrics
        }
        all_metrics.append(row)

    print(f"Collected metrics from {len(all_metrics)} experiments")

    # Define output columns (ordered)
    config_columns = [
        'run_name', 'dataset', 'method', 'mode', 'num_edit_layers', 'edit_layers',
        'actual_edit_layers',  # NEW: Actual layers from edit_log.csv
        'projection_samples', 'nullspace_threshold', 'max_edits'
    ]

    metric_columns = [
        # Edit performance
        'edit_total_wrong', 'edit_num_fixed', 'edit_fix_ratio',
        'edit_num_broken', 'edit_acc_before', 'edit_acc_after',
        # Test set performance
        'test_total', 'test_acc_before', 'test_acc_after', 'test_acc_delta',
        'test_correct_before', 'test_correct_after',
        'test_errors_before', 'test_errors_after',
        'test_correct_to_wrong', 'test_wrong_to_correct',
        # Projection stability
        'proj_total', 'proj_acc_before', 'proj_acc_after',
        'proj_stability', 'proj_regression_rate',
        'proj_num_broken', 'proj_num_fixed',
        # Discovery set
        'discovery_total', 'discovery_acc_before', 'discovery_acc_after', 'discovery_acc_delta',
        # Timing
        'duration_seconds', 'edit_seconds',
    ]

    all_columns = config_columns + metric_columns

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction='ignore')
        writer.writeheader()

        for row in all_metrics:
            # Convert list to string for edit_layers and actual_edit_layers
            if isinstance(row.get('edit_layers'), list):
                row['edit_layers'] = ','.join(map(str, row['edit_layers']))
            if isinstance(row.get('actual_edit_layers'), list):
                row['actual_edit_layers'] = ','.join(map(str, row['actual_edit_layers']))
            writer.writerow(row)

    print(f"\nSummary saved to: {output_path}")

    # Print quick statistics
    print("\n" + "=" * 70)
    print("QUICK STATISTICS")
    print("=" * 70)

    # Group by dataset
    by_dataset = {}
    for row in all_metrics:
        ds = row.get('dataset', 'unknown')
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(row)

    for dataset, rows in sorted(by_dataset.items()):
        print(f"\n{dataset}: {len(rows)} experiments")

        # Find best by test_acc_delta
        valid_rows = [r for r in rows if r.get('test_acc_delta') is not None]
        if valid_rows:
            best = max(valid_rows, key=lambda x: x.get('test_acc_delta', float('-inf')))
            print(f"  Best test_acc_delta: {best.get('test_acc_delta', 0) * 100:+.2f}%")
            print(f"    Config: {best.get('run_name')}")

            # Show edit fix ratio for best
            if best.get('edit_fix_ratio') is not None:
                print(f"    Edit fix ratio: {best.get('edit_fix_ratio') * 100:.1f}%")

        # Find best by edit_fix_ratio
        valid_fix = [r for r in rows if r.get('edit_fix_ratio') is not None]
        if valid_fix:
            best_fix = max(valid_fix, key=lambda x: x.get('edit_fix_ratio', 0))
            if best_fix != best:
                print(f"  Best edit_fix_ratio: {best_fix.get('edit_fix_ratio', 0) * 100:.1f}%")
                print(f"    Config: {best_fix.get('run_name')}")


if __name__ == "__main__":
    main()
