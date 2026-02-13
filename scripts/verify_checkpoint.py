#!/usr/bin/env python
"""
Checkpoint Consistency Verification Script
===========================================
Verifies checkpoint integrity across platforms by computing:
1. File hash (SHA256) of the .pt file
2. Weight hash (deterministic hash of all model parameters)
3. Prediction hash (hash of full prediction vector on test set)
4. Accuracy and per-class accuracy on test set

Run on both platforms and compare the output to verify checkpoints are identical.

Usage:
    # Basic verification (finetuned checkpoint)
    uv run python scripts/verify_checkpoint.py --dataset pathmnist --model vit-base

    # Verify edited checkpoint
    uv run python scripts/verify_checkpoint.py --dataset pathmnist --checkpoint-type edited

    # Verify arbitrary checkpoint file
    uv run python scripts/verify_checkpoint.py --dataset pathmnist --checkpoint-path path/to/model.pt

    # Multiple runs with deterministic mode
    uv run python scripts/verify_checkpoint.py --dataset pathmnist --deterministic --num-runs 3

    # Liver dataset
    uv run python scripts/verify_checkpoint.py --dataset liver4 --data-path dataset/

Output:
    JSON file with platform info, hashes, accuracy, and a COMPARE_THESE block
    for easy cross-platform comparison.
"""

import os
import sys
import argparse
import hashlib
import json
import time
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Path setup (follow existing script pattern)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# HuggingFace mirror config (must be set BEFORE importing transformers)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
_MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "pretrained"
_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_MODEL_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(_MODEL_CACHE_DIR)

import torch
import numpy as np

from data_handler import MEDMNIST_INFO, get_data_handler
from trainer import Trainer
from evaluator import Evaluator

# Duplicated from main.py to avoid importing main.py (which has side effects)
MODEL_REGISTRY = {
    "vit-base": "google/vit-base-patch16-224",
    "vit-tiny": "WinKawaks/vit-tiny-patch16-224",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify checkpoint consistency across platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(MEDMNIST_INFO.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit-base",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture (default: vit-base)",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        default="finetuned",
        help="Checkpoint type suffix, e.g. finetuned, edited, best (default: finetuned)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override: explicit path to checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data path (for liver datasets)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of inference runs (default: 1; use >1 to test determinism)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable full deterministic mode (torch.use_deterministic_algorithms)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated in results/)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory for data handler (default: logs)",
    )

    return parser.parse_args()


# ============================================================================
# Hash Functions
# ============================================================================


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of checkpoint file on disk."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_weight_hash(model: torch.nn.Module) -> str:
    """Compute deterministic hash of all model parameter values.

    Processes parameters in sorted key order. Casts to float32 for
    platform-independent byte representation.
    """
    h = hashlib.sha256()
    state_dict = model.state_dict()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].cpu().float()
        h.update(key.encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def compute_prediction_hash(predictions: np.ndarray) -> str:
    """Compute SHA256 hash of integer prediction vector."""
    return hashlib.sha256(predictions.astype(np.int64).tobytes()).hexdigest()


# ============================================================================
# Deterministic Mode
# ============================================================================


def set_deterministic_mode(seed: int = 42):
    """Set all random seeds and enable deterministic algorithms."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"  Warning: torch.use_deterministic_algorithms(True) failed: {e}")
        print(f"  Some operations may be non-deterministic on GPU.")


# ============================================================================
# Inference
# ============================================================================


def run_single_inference(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    desc: str = "Inference",
) -> Dict[str, Any]:
    """Run a single inference pass on the test set and return metrics + hashes."""
    evaluator = Evaluator(
        model=model,
        device=device,
        results_dir="results/_verify_tmp",
        log_dir="logs/_verify_tmp",
    )

    results = evaluator.run_inference(test_loader, desc=desc)
    metrics = evaluator.compute_metrics()

    pred_hash = compute_prediction_hash(results["predictions"])

    # Per-class accuracy (recall per class)
    per_class_acc = {}
    recall_arr = metrics["per_class"]["recall"]
    for cls_id in range(len(recall_arr)):
        per_class_acc[str(cls_id)] = round(float(recall_arr[cls_id]), 6)

    return {
        "accuracy": float(metrics["accuracy"]),
        "auc": float(metrics["auc"]) if metrics.get("auc") is not None else None,
        "macro_f1": float(metrics["macro"]["f1"]),
        "num_errors": int(metrics["num_errors"]),
        "total_samples": int(metrics["total_samples"]),
        "per_class_accuracy": per_class_acc,
        "prediction_hash": pred_hash,
    }


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    print("=" * 70)
    print("CHECKPOINT CONSISTENCY VERIFICATION")
    print("=" * 70)

    model_name = MODEL_REGISTRY[args.model]
    model_short = args.model

    # Deterministic mode
    if args.deterministic:
        print("\n[DETERMINISTIC MODE ENABLED]")
        set_deterministic_mode(args.seed)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ================================================================
    # Step 1: Resolve and validate checkpoint path
    # ================================================================
    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
    else:
        ckpt_path = (
            Path(args.checkpoint_dir)
            / f"{model_short}_{args.dataset}_{args.checkpoint_type}.pt"
        )

    if not ckpt_path.exists():
        print(f"\nERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"  Size: {ckpt_path.stat().st_size:,} bytes")

    # ================================================================
    # Step 2: Compute file hash
    # ================================================================
    print("\nComputing file hash...")
    file_hash = compute_file_hash(ckpt_path)
    print(f"  SHA256 (file): {file_hash}")

    # ================================================================
    # Step 3: Load model and compute weight hash
    # ================================================================
    print(f"\nLoading model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    dataset_info = MEDMNIST_INFO[args.dataset]
    num_classes = dataset_info["n_classes"]
    n_channels = dataset_info["n_channels"]

    trainer = Trainer(
        model_name=model_name,
        model_short=model_short,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_classes=num_classes,
        dataset_name=args.dataset,
        n_channels=n_channels,
    )
    trainer.setup_model()

    # Load checkpoint weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded model_state_dict from checkpoint")

    # Extract checkpoint metadata
    ckpt_meta = {}
    for key in ("config", "best_acc", "epoch", "dataset_name", "hparams"):
        if key in checkpoint:
            val = checkpoint[key]
            # Convert non-serializable types
            if isinstance(val, dict):
                ckpt_meta[key] = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in val.items()}
            else:
                ckpt_meta[key] = val

    # Compute weight hash
    print("\nComputing weight hash...")
    weight_hash = compute_weight_hash(trainer.model)
    print(f"  SHA256 (weights): {weight_hash}")

    # ================================================================
    # Step 4: Load test data
    # ================================================================
    print(f"\nLoading test data for {args.dataset}...")

    data_handler_kwargs = {
        "dataset_name": args.dataset,
        "ft_train_ratio": 0.9,
        "random_seed": args.seed,
        "log_dir": args.log_dir,
    }
    if args.data_path:
        data_handler_kwargs["data_path"] = args.data_path

    data_handler = get_data_handler(**data_handler_kwargs)
    data_handler.load_data()
    data_handler.create_resplit()

    transform = trainer.get_transforms()
    dataloaders = data_handler.get_dataloaders(
        batch_size=args.batch_size,
        transform=transform,
        pin_memory=False,
    )

    test_loader = dataloaders["test"]
    print(f"  Test samples: {len(test_loader.dataset)}")

    # ================================================================
    # Step 5: Run inference N times
    # ================================================================
    print(f"\nRunning inference ({args.num_runs} run(s))...")

    run_results = []
    for run_idx in range(args.num_runs):
        if args.deterministic:
            set_deterministic_mode(args.seed)

        run_start = time.time()
        result = run_single_inference(
            model=trainer.model,
            test_loader=test_loader,
            device=device,
            desc=f"Run {run_idx + 1}/{args.num_runs}",
        )
        run_time = time.time() - run_start
        result["inference_time_seconds"] = round(run_time, 3)

        run_results.append(result)

        print(
            f"  Run {run_idx + 1}: "
            f"accuracy={result['accuracy'] * 100:.4f}%, "
            f"pred_hash={result['prediction_hash'][:16]}..., "
            f"time={run_time:.2f}s"
        )

    # ================================================================
    # Step 6: Check cross-run consistency
    # ================================================================
    all_pred_hashes = [r["prediction_hash"] for r in run_results]
    all_accuracies = [r["accuracy"] for r in run_results]

    pred_consistent = len(set(all_pred_hashes)) == 1
    acc_consistent = len(set(all_accuracies)) == 1

    # ================================================================
    # Step 7: Assemble output summary
    # ================================================================
    summary = {
        "verification_timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "checkpoint": {
            "path": str(ckpt_path.resolve()),
            "file_size_bytes": ckpt_path.stat().st_size,
            "file_hash_sha256": file_hash,
            "weight_hash_sha256": weight_hash,
            "metadata": ckpt_meta,
        },
        "model": {
            "name": model_name,
            "short": model_short,
            "num_classes": num_classes,
        },
        "dataset": {
            "name": args.dataset,
            "test_samples": len(test_loader.dataset),
            "n_channels": n_channels,
        },
        "settings": {
            "deterministic": args.deterministic,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_runs": args.num_runs,
        },
        "runs": run_results,
        "consistency": {
            "predictions_consistent_across_runs": pred_consistent,
            "accuracy_consistent_across_runs": acc_consistent,
            "unique_prediction_hashes": list(set(all_pred_hashes)),
            "unique_accuracy_values": sorted(set(all_accuracies)),
        },
        "COMPARE_THESE": {
            "file_hash": file_hash,
            "weight_hash": weight_hash,
            "prediction_hash": run_results[0]["prediction_hash"],
            "accuracy": run_results[0]["accuracy"],
        },
    }

    # ================================================================
    # Step 8: Output results
    # ================================================================
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            PROJECT_ROOT
            / "results"
            / f"verify_{model_short}_{args.dataset}_{args.checkpoint_type}.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 70}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  File hash (SHA256):   {file_hash}")
    print(f"  Weight hash (SHA256): {weight_hash}")
    print(f"  Prediction hash:      {run_results[0]['prediction_hash']}")
    print(f"  Accuracy:             {run_results[0]['accuracy'] * 100:.4f}%")
    if run_results[0].get("auc") is not None:
        print(f"  AUC:                  {run_results[0]['auc']:.6f}")
    print(f"  Macro-F1:             {run_results[0]['macro_f1']:.6f}")
    print()
    print(f"  Runs: {args.num_runs}")
    print(f"  Predictions consistent: {pred_consistent}")
    print()
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 70}")

    # Cross-platform comparison block
    print(f"\n--- CROSS-PLATFORM COMPARISON BLOCK ---")
    print(f"file_hash:       {file_hash}")
    print(f"weight_hash:     {weight_hash}")
    print(f"prediction_hash: {run_results[0]['prediction_hash']}")
    print(f"accuracy:        {run_results[0]['accuracy'] * 100:.4f}%")
    print(f"--- END ---")


if __name__ == "__main__":
    main()
