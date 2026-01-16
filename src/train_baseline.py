"""
Baseline Training Script for PathMNIST
======================================
Fine-tunes ViT-B/16 on PathMNIST using official MedMNIST splits.

Data Protocol:
- train: Fine-tuning AND finding edit candidates
- val: Model selection (best checkpoint) and locality checks
- test: Final report only

Usage:
    uv run python src/train_baseline.py --epochs 10
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import medmnist
from medmnist import PathMNIST
from transformers import ViTForImageClassification, ViTImageProcessor


# PathMNIST class names (9 tissue types)
CLASS_NAMES = [
    "Adipose",                              # 0
    "Background",                           # 1
    "Debris",                               # 2
    "Lymphocytes",                          # 3
    "Mucus",                                # 4
    "Smooth Muscle",                        # 5
    "Normal Colon Mucosa",                  # 6
    "Cancer-associated Stroma",             # 7
    "Colorectal Adenocarcinoma Epithelium"  # 8
]

SANITY_THRESHOLD = 98.0  # Warn if val/test accuracy exceeds this


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms() -> transforms.Compose:
    """Get image transforms for ViT input."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_datasets(
    data_dir: str = "./data",
    download: bool = True
) -> Dict[str, PathMNIST]:
    """
    Load PathMNIST datasets using official MedMNIST splits.

    Args:
        data_dir: Directory to store/load data
        download: Whether to download if not present

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    print(f"Loading PathMNIST (size=224) from {data_dir}...")
    print(f"MedMNIST version: {medmnist.__version__}")

    transform = get_transforms()

    train_dataset = PathMNIST(
        split='train',
        transform=transform,
        download=download,
        root=data_dir,
        size=224
    )

    val_dataset = PathMNIST(
        split='val',
        transform=transform,
        download=download,
        root=data_dir,
        size=224
    )

    test_dataset = PathMNIST(
        split='test',
        transform=transform,
        download=download,
        root=data_dir,
        size=224
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }


def create_dataloaders(
    datasets: Dict[str, PathMNIST],
    batch_size: int = 32,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """Create DataLoaders for all splits."""
    pin_memory = torch.cuda.is_available()

    return {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }


def setup_model(
    model_name: str = "google/vit-base-patch16-224",
    num_classes: int = 9,
    device: torch.device = None
) -> nn.Module:
    """Initialize ViT model for image classification."""
    print(f"\nLoading model: {model_name}")

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval"
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"[{desc}]")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        outputs = model(images)
        logits = outputs.logits

        loss = criterion(logits, labels)

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss / len(dataloader):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_acc: float,
    history: list,
    filepath: Path
):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
        'training_history': history,
        'config': {
            'model_name': "google/vit-base-patch16-224",
            'num_classes': 9,
            'class_names': CLASS_NAMES
        }
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    filepath: Path,
    device: torch.device
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def plot_training_curve(
    history: list,
    save_path: Path
):
    """Plot and save training curves."""
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_acc = [h['val_acc'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(epochs, train_acc, 'b-', label='Train Acc')
    axes[1].plot(epochs, val_acc, 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def sanity_check(val_acc: float, test_acc: float) -> bool:
    """
    Check if val/test accuracy is suspiciously high.

    Returns:
        True if sanity check passed (no warnings), False otherwise
    """
    passed = True

    if val_acc > SANITY_THRESHOLD:
        print(f"\n{'='*60}")
        print(f"SANITY CHECK WARNING:")
        print(f"  Val accuracy ({val_acc:.2f}%) exceeds {SANITY_THRESHOLD}% threshold.")
        print(f"  This may indicate:")
        print(f"  - Data leakage between splits")
        print(f"  - The task is too easy for the model")
        print(f"  - Potential issues with the dataset")
        print(f"  Please investigate before proceeding with editing experiments.")
        print(f"{'='*60}")
        passed = False

    if test_acc > SANITY_THRESHOLD:
        print(f"\n{'='*60}")
        print(f"SANITY CHECK WARNING:")
        print(f"  Test accuracy ({test_acc:.2f}%) exceeds {SANITY_THRESHOLD}% threshold.")
        print(f"  This may indicate:")
        print(f"  - Data leakage between splits")
        print(f"  - The task is too easy for the model")
        print(f"  - Potential issues with the dataset")
        print(f"  Please investigate before proceeding with editing experiments.")
        print(f"{'='*60}")
        passed = False

    return passed


def train(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    checkpoint_dir: Path = Path("checkpoints"),
    log_dir: Path = Path("logs")
) -> Tuple[nn.Module, list]:
    """
    Full training loop with best checkpoint saving.

    Returns:
        Tuple of (best_model, training_history)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"{'='*60}")

    best_acc = 0.0
    history = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, criterion, device, epoch
        )

        # Validate
        val_metrics = evaluate(
            model, dataloaders['val'], criterion, device, desc="Val"
        )

        # Update scheduler
        scheduler.step()

        # Record metrics
        epoch_time = time.time() - epoch_start
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        }
        history.append(metrics)

        # Print summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

        # Save best checkpoint
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            save_checkpoint(
                model, optimizer, epoch, best_acc, history,
                checkpoint_dir / "vit_baseline_best.pt"
            )
            print(f"  New best model saved! (Val Acc: {best_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT-B/16 on PathMNIST (official splits)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")

    args = parser.parse_args()

    print("=" * 70)
    print("Baseline Training: ViT-B/16 on PathMNIST")
    print("=" * 70)
    print(f"Using official MedMNIST splits (no manual re-splitting)")
    print()

    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)

    # Load data
    datasets = load_datasets(data_dir=args.data_dir, download=True)
    dataloaders = create_dataloaders(datasets, batch_size=args.batch_size)

    # Setup model
    model = setup_model(device=device)

    # Train
    model, history = train(
        model=model,
        dataloaders=dataloaders,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )

    # Load best checkpoint for final evaluation
    print(f"\n{'='*60}")
    print("Loading best checkpoint for final evaluation...")
    best_checkpoint_path = checkpoint_dir / "vit_baseline_best.pt"
    load_checkpoint(model, best_checkpoint_path, device)

    # Final evaluation on all splits
    print(f"\n{'='*60}")
    print("Final Evaluation (Best Checkpoint)")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()

    train_metrics = evaluate(model, dataloaders['train'], criterion, device, desc="Train")
    val_metrics = evaluate(model, dataloaders['val'], criterion, device, desc="Val")
    test_metrics = evaluate(model, dataloaders['test'], criterion, device, desc="Test")

    print(f"\n{'='*60}")
    print("=== Final Results ===")
    print(f"Train Accuracy: {train_metrics['accuracy']:.2f}%")
    print(f"Val Accuracy: {val_metrics['accuracy']:.2f}%")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"{'='*60}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([
        {'split': 'train', 'accuracy': train_metrics['accuracy'], 'loss': train_metrics['loss']},
        {'split': 'val', 'accuracy': val_metrics['accuracy'], 'loss': val_metrics['loss']},
        {'split': 'test', 'accuracy': test_metrics['accuracy'], 'loss': test_metrics['loss']}
    ])

    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "baseline_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = log_dir / "baseline_training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")

    # Plot training curve
    plot_path = log_dir / "training_curve.png"
    plot_training_curve(history, plot_path)
    print(f"Training curve saved to: {plot_path}")

    # Sanity check
    sanity_passed = sanity_check(val_metrics['accuracy'], test_metrics['accuracy'])

    if sanity_passed:
        print(f"\nSanity check passed. Val/Test accuracy is within expected range.")

    print(f"\nBaseline training complete!")


if __name__ == "__main__":
    main()
