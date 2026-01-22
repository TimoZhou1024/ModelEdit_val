"""
Run All Baselines Script
========================
Runs baseline1 and baseline2 on all MedMNIST datasets in parallel.
Designed for multi-GPU environments (e.g., 8x 3090 GPUs).

Usage:
    python scripts/run_all_baselines.py --num-gpus 8
    python scripts/run_all_baselines.py --num-gpus 4 --datasets pathmnist dermamnist
    python scripts/run_all_baselines.py --dry-run  # Preview commands without running
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# All supported datasets
ALL_DATASETS = [
    "pathmnist",
    "dermamnist",
    "retinamnist",
    "organamnist",
    "bloodmnist",
    "tissuemnist"
]

# Baseline configurations
BASELINES = ["baseline1", "baseline2"]


def run_job(job_config: dict) -> dict:
    """
    Run a single baseline job.

    Args:
        job_config: Dictionary with job configuration

    Returns:
        Dictionary with job results
    """
    dataset = job_config["dataset"]
    baseline = job_config["baseline"]
    gpu_id = job_config["gpu_id"]
    run_name = job_config["run_name"]
    baseline_epochs = job_config["baseline_epochs"]
    baseline_lr = job_config["baseline_lr"]
    max_edits = job_config["max_edits"]

    # Set environment for this GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Build command
    cmd = [
        "uv", "run", "python", "src/main.py",
        "--stage", baseline,
        "--dataset", dataset,
        "--run-name", run_name,
        "--baseline-epochs", str(baseline_epochs),
        "--max-edits", str(max_edits),
    ]

    # Add baseline-lr only for baseline2
    if baseline == "baseline2":
        cmd.extend(["--baseline-lr", str(baseline_lr)])

    start_time = time.time()
    job_name = f"{dataset}_{baseline}"

    print(f"[GPU {gpu_id}] Starting: {job_name}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per job
        )

        elapsed = time.time() - start_time
        success = result.returncode == 0

        if success:
            print(f"[GPU {gpu_id}] Completed: {job_name} ({elapsed:.1f}s)")
        else:
            print(f"[GPU {gpu_id}] FAILED: {job_name} ({elapsed:.1f}s)")
            print(f"  Error: {result.stderr[:500] if result.stderr else 'No error message'}")

        return {
            "job_name": job_name,
            "dataset": dataset,
            "baseline": baseline,
            "gpu_id": gpu_id,
            "success": success,
            "elapsed": elapsed,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_id}] TIMEOUT: {job_name} ({elapsed:.1f}s)")
        return {
            "job_name": job_name,
            "dataset": dataset,
            "baseline": baseline,
            "gpu_id": gpu_id,
            "success": False,
            "elapsed": elapsed,
            "returncode": -1,
            "stdout": "",
            "stderr": "Job timed out after 2 hours"
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_id}] ERROR: {job_name} - {str(e)}")
        return {
            "job_name": job_name,
            "dataset": dataset,
            "baseline": baseline,
            "gpu_id": gpu_id,
            "success": False,
            "elapsed": elapsed,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run all baselines on all datasets in parallel"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to run (default: all)"
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=BASELINES,
        choices=BASELINES,
        help="Baselines to run (default: both)"
    )
    parser.add_argument(
        "--baseline-epochs",
        type=int,
        default=10,
        help="Number of epochs for baseline training (default: 10)"
    )
    parser.add_argument(
        "--baseline-lr",
        type=float,
        default=1e-5,
        help="Learning rate for baseline2 finetuning (default: 1e-5)"
    )
    parser.add_argument(
        "--max-edits",
        type=int,
        default=30,
        help="Maximum number of error samples to use (default: 30)"
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Prefix for run names (default: timestamp)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running"
    )

    args = parser.parse_args()

    # Generate run prefix
    if args.run_prefix is None:
        args.run_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build job list
    jobs = []
    for dataset in args.datasets:
        for baseline in args.baselines:
            run_name = f"{args.run_prefix}_{dataset}_{baseline}"
            jobs.append({
                "dataset": dataset,
                "baseline": baseline,
                "run_name": run_name,
                "baseline_epochs": args.baseline_epochs,
                "baseline_lr": args.baseline_lr,
                "max_edits": args.max_edits,
                "gpu_id": None  # Will be assigned later
            })

    print("=" * 70)
    print("RUN ALL BASELINES")
    print("=" * 70)
    print(f"Datasets: {args.datasets}")
    print(f"Baselines: {args.baselines}")
    print(f"Total jobs: {len(jobs)}")
    print(f"GPUs available: {args.num_gpus}")
    print(f"Run prefix: {args.run_prefix}")
    print(f"Baseline epochs: {args.baseline_epochs}")
    print(f"Max edits: {args.max_edits}")
    print("=" * 70)

    # Print job list
    print("\nJobs to run:")
    for i, job in enumerate(jobs):
        gpu_id = i % args.num_gpus
        job["gpu_id"] = gpu_id
        print(f"  [{i+1:2d}] GPU {gpu_id}: {job['dataset']} - {job['baseline']}")

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for job in jobs:
            cmd = [
                "uv", "run", "python", "src/main.py",
                "--stage", job["baseline"],
                "--dataset", job["dataset"],
                "--run-name", job["run_name"],
                "--baseline-epochs", str(job["baseline_epochs"]),
                "--max-edits", str(job["max_edits"]),
            ]
            if job["baseline"] == "baseline2":
                cmd.extend(["--baseline-lr", str(job["baseline_lr"])])

            print(f"\n  CUDA_VISIBLE_DEVICES={job['gpu_id']} {' '.join(cmd)}")
        return

    # Run jobs in parallel
    print(f"\nStarting {len(jobs)} jobs on {args.num_gpus} GPUs...")
    start_time = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        print("\nSuccessful jobs:")
        for r in successful:
            print(f"  [OK] {r['job_name']} ({r['elapsed']:.1f}s)")

    if failed:
        print("\nFailed jobs:")
        for r in failed:
            print(f"  [FAIL] {r['job_name']} - {r['stderr'][:100] if r['stderr'] else 'Unknown error'}")

    # Save results to log file
    log_dir = Path(__file__).parent.parent / "logs" / args.run_prefix
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "baseline_run_summary.txt"
    with open(log_file, "w") as f:
        f.write(f"Run All Baselines Summary\n")
        f.write(f"========================\n")
        f.write(f"Run prefix: {args.run_prefix}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Successful: {len(successful)}/{len(results)}\n")
        f.write(f"Failed: {len(failed)}/{len(results)}\n\n")

        for r in results:
            status = "OK" if r["success"] else "FAIL"
            f.write(f"[{status}] {r['job_name']} - {r['elapsed']:.1f}s\n")
            if not r["success"] and r["stderr"]:
                f.write(f"  Error: {r['stderr'][:500]}\n")

    print(f"\nLog saved to: {log_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
