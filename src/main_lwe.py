"""
Standalone entry for LWE baseline (FT/HPRD) on selected datasets.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ============================================================
# HuggingFace Mirror Configuration (for users in China)
# Set environment variables BEFORE importing transformers
# ============================================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

_PROJECT_ROOT = Path(__file__).parent.parent
_MODEL_CACHE_DIR = _PROJECT_ROOT / 'models' / 'pretrained'
_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(_MODEL_CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(_MODEL_CACHE_DIR)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_handler import MEDMNIST_INFO, get_data_handler
from trainer import Trainer
from lwe_baseline import run_lwe_baseline


MODEL_REGISTRY = {
    "vit-base": "google/vit-base-patch16-224",
    "vit-tiny": "WinKawaks/vit-tiny-patch16-224",
}

SUPPORTED_DATASETS = [
    "bloodmnist",
    "dermamnist",
    "organamnist",
    "pathmnist",
    "retinamnist",
    "liver2s",
    "liver2a",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone LWE baseline runner (FT/HPRD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_lwe.py --dataset pathmnist --lwe-method ft
  python main_lwe.py --dataset retinamnist --lwe-method hprd --lwe-hprd-epochs 5
""",
    )

    parser.add_argument("--dataset", type=str, required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--model", type=str, default="vit-tiny", choices=list(MODEL_REGISTRY.keys()))

    # Data arguments
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--ft-train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-edits", type=str, default="30", help="Integer or 'all'")

    # LWE arguments
    parser.add_argument("--lwe-method", type=str, choices=["ft", "hprd"], default="ft")
    parser.add_argument("--lwe-layer", type=int, default=None)
    parser.add_argument("--lwe-edit-lr", type=float, default=2e-5)
    parser.add_argument("--lwe-max-steps", type=int, default=100)
    parser.add_argument("--lwe-cloc", type=float, default=1.0)
    parser.add_argument("--lwe-l2-reg", action="store_true")
    parser.add_argument("--lwe-sparsity", type=float, default=0.1)
    parser.add_argument("--lwe-hprd-epochs", type=int, default=3)
    parser.add_argument("--lwe-hprd-lr", type=float, default=1e-4)
    parser.add_argument("--lwe-hprd-max-batches", type=int, default=None)
    parser.add_argument("--lwe-hprd-ckpt", type=str, default=None)

    # Runtime/output arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--timestamp", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--pin-memory", action="store_true", default=None)
    parser.add_argument("--no-pin-memory", action="store_true")

    return parser.parse_args()


def export_timing_metrics(results_dir: str, duration_seconds: float, edit_seconds: float | None) -> None:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    timing_path = results_path / "timing.csv"

    rows = [
        {"metric": "duration_seconds", "value": duration_seconds, "notes": "total stage duration"},
        {"metric": "edit_seconds", "value": edit_seconds, "notes": "edit stage duration"},
    ]

    with open(timing_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_lwe_stage(args):
    print("\n" + "=" * 70)
    print(f"LWE BASELINE ({args.lwe_method.upper()}) - {args.dataset.upper()}")
    print("=" * 70)

    data_handler = get_data_handler(
        dataset_name=args.dataset,
        data_path=args.data_path,
        ft_train_ratio=args.ft_train_ratio,
        random_seed=args.seed,
        log_dir=args.log_dir,
    )
    data_handler.load_data()
    data_handler.create_resplit()

    trainer = Trainer(
        model_name=args.model_name,
        model_short=args.model_short,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_classes=data_handler.n_classes,
        dataset_name=args.dataset,
        n_channels=data_handler.n_channels,
    )
    trainer.setup_model()

    finetuned_path = Path(args.checkpoint_dir) / f"{args.model_short}_{args.dataset}_finetuned.pt"
    if not finetuned_path.exists():
        raise FileNotFoundError(f"Finetuned model not found: {finetuned_path}")

    trainer.load_checkpoint(filepath=finetuned_path, load_optimizer=False)

    transform = trainer.get_transforms()
    dataloaders = data_handler.get_dataloaders(
        batch_size=args.batch_size,
        transform=transform,
        pin_memory=args.pin_memory,
    )

    print("\nFinding misclassified samples from Edit-Discovery set...")
    misclassified = trainer.find_misclassified(
        dataloaders["discovery"],
        max_samples=args.max_edits,
    )

    if len(misclassified["indices"]) == 0:
        print("No misclassified samples found. Skip editing.")
        return None, None

    edited_model, edit_seconds = run_lwe_baseline(
        args=args,
        trainer=trainer,
        data_handler=data_handler,
        dataloaders=dataloaders,
        misclassified=misclassified,
    )
    return edited_model, edit_seconds


def main():
    args = parse_args()

    args.model_name = MODEL_REGISTRY[args.model]
    args.model_short = args.model

    if isinstance(args.max_edits, str) and args.max_edits.lower() in {"all", "*"}:
        args.max_edits = None
    else:
        args.max_edits = int(args.max_edits)

    if args.run_name:
        suffix = args.run_name
    elif args.timestamp:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        suffix = None

    if suffix:
        args.log_dir = f"{args.log_dir}/{suffix}"
        args.results_dir = f"{args.results_dir}/{suffix}"
        print(f"Run identifier: {suffix}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.no_pin_memory:
        args.pin_memory = False
    elif args.pin_memory:
        args.pin_memory = True
    else:
        args.pin_memory = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_info = MEDMNIST_INFO[args.dataset]
    print(f"\n{'=' * 70}")
    print(f"LWE Baseline Pipeline - {args.dataset.upper()}")
    print(f"{'=' * 70}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Classes: {dataset_info['n_classes']}")
    print(f"  Channels: {dataset_info['n_channels']} ({'Grayscale' if dataset_info['n_channels'] == 1 else 'RGB'})")
    print(f"  Description: {dataset_info['description']}")
    print(f"  Model: {args.model_short} ({args.model_name})")

    start = time.time()
    _, edit_seconds = run_lwe_stage(args)
    duration = time.time() - start
    export_timing_metrics(args.results_dir, duration, edit_seconds)


if __name__ == "__main__":
    main()
