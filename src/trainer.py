"""
Trainer Module for ViT Model Editing Pipeline
==============================================
Fine-tunes ViT-B/16 on MedMNIST datasets with:
- Automatic GPU/CPU device selection
- Checkpoint saving and loading
- Training metrics logging
- Multi-dataset support (PathMNIST, DermaMNIST, OrganAMNIST, etc.)
- Grayscale to RGB conversion for ViT compatibility
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import transforms

from data_handler import DataHandler, MedMNISTDataset


class Trainer:
    """
    Fine-tunes ViT model on MedMNIST datasets.
    Handles device selection, checkpointing, and logging automatically.
    Supports multiple datasets with different number of classes.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 9,
        dataset_name: str = "pathmnist",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: str = None,
        n_channels: int = 3
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of output classes (varies by dataset)
            dataset_name: Name of the dataset (for checkpoint naming)
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            n_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.n_channels = n_channels

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Model components (initialized later)
        self.model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.training_history = []

        # Checkpoint naming (includes dataset name)
        self.checkpoint_name = f"vit_{self.dataset_name}_finetuned.pt"
        self.best_checkpoint_name = f"vit_{self.dataset_name}_best.pt"
        
    def setup_model(self) -> nn.Module:
        """Initialize ViT model for image classification."""
        print(f"\nLoading model: {self.model_name}")
        
        # Load pre-trained ViT
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True  # Allow classifier head replacement
        )
        
        # Load image processor for preprocessing
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def get_transforms(self) -> transforms.Compose:
        """
        Get image transforms for ViT input.

        CRITICAL: For grayscale images (n_channels=1), we add Grayscale(3)
        BEFORE ToTensor to convert to 3-channel RGB for ViT compatibility.
        """
        transform_list = [
            transforms.Resize((224, 224)),
        ]

        # Handle grayscale images: convert to 3-channel RGB
        if self.n_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=3))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transforms.Compose(transform_list)
    
    def setup_training(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 2,
        total_epochs: int = 10
    ):
        """Setup optimizer, scheduler, and loss function."""
        if self.model is None:
            self.setup_model()
        
        # AdamW optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Cosine annealing scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_epochs - warmup_epochs,
            T_mult=1
        )
        
        # Cross-entropy loss
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining setup:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Epochs: {total_epochs}")
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            logits = outputs.logits
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_acc': 100. * correct / total
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        desc: str = "Eval"
    ) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc=f"[{desc}]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            logits = outputs.logits
            
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/len(val_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_acc': 100. * correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        save_best: bool = True
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            save_best: Whether to save checkpoint when validation improves
            
        Returns:
            Dictionary with training history and best metrics
        """
        self.setup_training(learning_rate=learning_rate, total_epochs=epochs)
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader, desc="Val")
            
            # Update scheduler
            self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start
            metrics = {
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics,
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': epoch_time
            }
            self.training_history.append(metrics)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.2f}%")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Save best checkpoint
            if save_best and val_metrics['val_acc'] > self.best_acc:
                self.best_acc = val_metrics['val_acc']
                self.save_checkpoint(
                    filepath=self.checkpoint_dir / self.checkpoint_name,
                    is_best=True
                )
                print(f"  ✓ New best model saved! (Acc: {self.best_acc:.2f}%)")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best validation accuracy: {self.best_acc:.2f}%")
        print(f"{'='*60}")
        
        # Export training log
        self.export_training_log()
        
        return {
            'history': self.training_history,
            'best_acc': self.best_acc,
            'total_time': total_time
        }
    
    def save_checkpoint(
        self,
        filepath: Path = None,
        is_best: bool = False
    ):
        """
        Save model checkpoint.

        Checkpoint contains:
        - model_state_dict: Model weights
        - optimizer_state_dict: Optimizer state
        - scheduler_state_dict: Scheduler state
        - epoch: Current epoch
        - best_acc: Best validation accuracy
        - training_history: Full training history
        - config: Model and training configuration
        """
        if filepath is None:
            filepath = self.checkpoint_dir / self.checkpoint_name

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'best_acc': self.best_acc,
            'training_history': self.training_history,
            'config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'dataset_name': self.dataset_name,
                'n_channels': self.n_channels,
                'device': str(self.device)
            }
        }

        torch.save(checkpoint, filepath)

        # Also save a "best" checkpoint
        if is_best:
            best_path = self.checkpoint_dir / self.best_checkpoint_name
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(
        self,
        filepath: Path = None,
        load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to restore optimizer state

        Returns:
            Checkpoint dictionary
        """
        if filepath is None:
            filepath = self.checkpoint_dir / self.checkpoint_name

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        print(f"Loading checkpoint from: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore model
        if self.model is None:
            self.setup_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer if requested
        if load_optimizer and checkpoint.get('optimizer_state_dict'):
            if self.optimizer is None:
                self.setup_training()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"  Restored from epoch {self.current_epoch+1}")
        print(f"  Best accuracy: {self.best_acc:.2f}%")
        
        return checkpoint
    
    def export_training_log(self) -> str:
        """Export training history to CSV."""
        if not self.training_history:
            print("No training history to export.")
            return None

        df = pd.DataFrame(self.training_history)
        csv_path = self.log_dir / f"{self.dataset_name}_training_metrics.csv"
        df.to_csv(csv_path, index=False)

        print(f"Training log exported to: {csv_path}")
        return str(csv_path)
    
    @torch.no_grad()
    def get_predictions(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model predictions for a dataset.
        
        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(self.device)
            
            outputs = self.model(images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return (
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs)
        )
    
    def find_misclassified(
        self,
        dataloader: DataLoader,
        max_samples: int = None
    ) -> Dict[str, Any]:
        """
        Find misclassified samples in the dataset.

        IMPORTANT: This function creates a new DataLoader with shuffle=False
        to ensure that returned indices correspond to dataset's original indices.

        Returns:
            Dictionary with indices, predictions, and true labels of errors
        """
        # Create a new DataLoader with shuffle=False to ensure index consistency
        # This is critical: if the original dataloader has shuffle=True,
        # the iteration order won't match dataset indices
        dataset = dataloader.dataset
        unshuffled_loader = DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,  # Must be False for correct index mapping
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
        )

        preds, labels, probs = self.get_predictions(unshuffled_loader)

        # Find misclassified
        misclassified_mask = preds != labels
        misclassified_indices = np.where(misclassified_mask)[0]

        if max_samples and len(misclassified_indices) > max_samples:
            misclassified_indices = misclassified_indices[:max_samples]

        print(f"\nFound {len(misclassified_indices)} misclassified samples")
        print(f"  Error rate: {100.*len(misclassified_indices)/len(labels):.2f}%")

        # Verify index consistency (debug check)
        if len(misclassified_indices) > 0:
            sample_idx = misclassified_indices[0]
            _, dataset_label = dataset[sample_idx]
            if isinstance(dataset_label, torch.Tensor):
                dataset_label = dataset_label.item()
            if dataset_label != labels[sample_idx]:
                print(f"  WARNING: Index mismatch detected! dataset[{sample_idx}].label={dataset_label}, labels[{sample_idx}]={labels[sample_idx]}")

        return {
            'indices': misclassified_indices,
            'predictions': preds[misclassified_indices],
            'true_labels': labels[misclassified_indices],
            'probabilities': probs[misclassified_indices],
            'total_samples': len(labels),
            'error_count': len(misclassified_indices)
        }

    # ================================================================
    # Baseline Methods
    # ================================================================

    def train_from_scratch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        checkpoint_suffix: str = "retrained"
    ) -> Dict[str, Any]:
        """
        Train a NEW model from scratch (Baseline 1).

        Reinitializes model weights before training.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            checkpoint_suffix: Suffix for checkpoint filename

        Returns:
            Training results dictionary
        """
        # Reinitialize model (fresh weights from pretrained)
        print("\nReinitializing model from pretrained weights...")
        self.model = None
        self.setup_model()

        # Update checkpoint name
        self.checkpoint_name = f"vit_{self.dataset_name}_{checkpoint_suffix}.pt"
        self.best_checkpoint_name = f"vit_{self.dataset_name}_{checkpoint_suffix}_best.pt"

        # Reset training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.training_history = []

        # Train
        return self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate
        )

    def finetune_on_samples(
        self,
        finetune_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 1e-5,
        checkpoint_suffix: str = "finetuned_on_errors"
    ) -> Dict[str, Any]:
        """
        Finetune existing model on specific samples (Baseline 2).

        Loads existing finetuned checkpoint and continues training
        on error samples with lower learning rate.

        Args:
            finetune_loader: DataLoader with samples to finetune on
            val_loader: Validation data loader
            epochs: Number of finetuning epochs
            learning_rate: Learning rate (should be lower than initial training)
            checkpoint_suffix: Suffix for checkpoint filename

        Returns:
            Training results dictionary
        """
        # Load existing finetuned model
        finetuned_path = self.checkpoint_dir / f"vit_{self.dataset_name}_finetuned.pt"
        if not finetuned_path.exists():
            raise FileNotFoundError(
                f"Finetuned model not found: {finetuned_path}. "
                f"Run --stage train first."
            )

        print(f"\nLoading finetuned model from: {finetuned_path}")
        self.load_checkpoint(filepath=finetuned_path, load_optimizer=False)

        # Update checkpoint name for this baseline
        self.checkpoint_name = f"vit_{self.dataset_name}_{checkpoint_suffix}.pt"
        self.best_checkpoint_name = f"vit_{self.dataset_name}_{checkpoint_suffix}_best.pt"

        # Reset training state (but keep model weights)
        self.current_epoch = 0
        self.best_acc = 0.0
        self.training_history = []

        # Train with lower learning rate
        return self.train(
            train_loader=finetune_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate
        )


def main():
    """Test trainer functionality."""
    print("=" * 70)
    print("ViT Model Editing Pipeline - Trainer (Multi-Dataset Support)")
    print("=" * 70)

    # Initialize data handler with default dataset
    dataset_name = "pathmnist"
    data_handler = DataHandler(
        dataset_name=dataset_name,
        ft_train_ratio=0.9,
        random_seed=42
    )
    data_handler.load_data()
    data_handler.create_resplit()

    # Initialize trainer with dynamic num_classes
    trainer = Trainer(
        model_name="google/vit-base-patch16-224",
        num_classes=data_handler.n_classes,
        dataset_name=dataset_name,
        n_channels=data_handler.n_channels
    )

    # Setup model
    trainer.setup_model()

    # Get transforms and dataloaders
    transform = trainer.get_transforms()
    dataloaders = data_handler.get_dataloaders(
        batch_size=32,
        transform=transform
    )

    # Train (short run for testing)
    print("\nStarting training (test run with 2 epochs)...")
    results = trainer.train(
        train_loader=dataloaders['ft_train'],
        val_loader=dataloaders['val'],
        epochs=2,
        learning_rate=1e-4
    )

    print("\n[OK] Trainer test complete!")
    print(f"  Dataset: {dataset_name}")
    print(f"  Classes: {data_handler.n_classes}")
    print(f"  Best accuracy: {results['best_acc']:.2f}%")
    print(f"  Checkpoint: {trainer.checkpoint_dir}/{trainer.checkpoint_name}")


if __name__ == "__main__":
    main()
