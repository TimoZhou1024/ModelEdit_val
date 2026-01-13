"""
Data Handler Module for ViT Model Editing Pipeline
===================================================
Loads PathMNIST dataset and creates strict train/held-out split.
The held-out set is reserved ONLY for final verification.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from PIL import Image


class PathMNISTDataset(Dataset):
    """
    Custom Dataset for PathMNIST loaded from .npz file.
    Images are 224x224x3, labels are integers 0-8 (9 classes).
    """
    
    # Class names for PathMNIST (9 tissue types)
    CLASS_NAMES = [
        "Adipose",           # 0
        "Background",        # 1
        "Debris",            # 2
        "Lymphocytes",       # 3
        "Mucus",             # 4
        "Smooth Muscle",     # 5
        "Normal Colon Mucosa",  # 6
        "Cancer-associated Stroma",  # 7
        "Colorectal Adenocarcinoma Epithelium"  # 8
    ]
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None,
        indices: np.ndarray = None
    ):
        """
        Args:
            images: numpy array of shape (N, 224, 224, 3)
            labels: numpy array of shape (N,) or (N, 1)
            transform: optional transforms to apply
            indices: optional indices to subset the data
        """
        self.images = images
        self.labels = labels.flatten()
        self.transform = transform
        self.indices = indices if indices is not None else np.arange(len(images))
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image = self.images[real_idx]
        label = int(self.labels[real_idx])
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        return image, label


class DataHandler:
    """
    Handles data loading, splitting, and logging for the ViT editing pipeline.
    Ensures strict isolation between training set and held-out validation set.
    """
    
    def __init__(
        self,
        data_path: str = None,
        held_out_ratio: float = 0.2,
        random_seed: int = 42,
        log_dir: str = "logs"
    ):
        """
        Args:
            data_path: Path to pathmnist_224.npz file
            held_out_ratio: Fraction of data to reserve for held-out validation
            random_seed: Random seed for reproducible splitting
            log_dir: Directory to save split information CSV
        """
        if data_path is None:
            data_path = os.path.expanduser("~/.medmnist/pathmnist_224.npz")
        
        self.data_path = Path(data_path)
        self.held_out_ratio = held_out_ratio
        self.random_seed = random_seed
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None
        
        # Split indices
        self.train_indices = None
        self.held_out_indices = None
        
        # Statistics
        self.split_info = {}
        
    def load_data(self) -> Dict[str, np.ndarray]:
        """Load PathMNIST data from .npz file."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"PathMNIST data not found at {self.data_path}. "
                "Please download it first."
            )
        
        print(f"Loading PathMNIST from {self.data_path}...")
        data = np.load(self.data_path)
        
        # Standard MedMNIST split keys
        self.train_images = data['train_images']
        self.train_labels = data['train_labels']
        self.val_images = data['val_images']
        self.val_labels = data['val_labels']
        self.test_images = data['test_images']
        self.test_labels = data['test_labels']
        
        print(f"  Train images: {self.train_images.shape}")
        print(f"  Val images: {self.val_images.shape}")
        print(f"  Test images: {self.test_images.shape}")
        
        return {
            'train_images': self.train_images,
            'train_labels': self.train_labels,
            'val_images': self.val_images,
            'val_labels': self.val_labels,
            'test_images': self.test_images,
            'test_labels': self.test_labels
        }
    
    def create_held_out_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a held-out validation set from the original training data.
        
        CRITICAL: The held-out set must be completely excluded from:
        - Fine-tuning phase
        - Weight editing calculation
        
        Returns:
            Tuple of (train_indices, held_out_indices)
        """
        if self.train_images is None:
            self.load_data()
        
        n_total = len(self.train_images)
        n_held_out = int(n_total * self.held_out_ratio)
        n_train = n_total - n_held_out
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Shuffle indices
        all_indices = np.arange(n_total)
        np.random.shuffle(all_indices)
        
        # Split
        self.train_indices = all_indices[:n_train]
        self.held_out_indices = all_indices[n_train:]
        
        # Verify no overlap
        assert len(set(self.train_indices) & set(self.held_out_indices)) == 0, \
            "CRITICAL: Overlap detected between train and held-out sets!"
        
        print(f"\n=== Data Split (Strict Isolation) ===")
        print(f"  Total samples: {n_total}")
        print(f"  Training set: {n_train} ({100*(1-self.held_out_ratio):.0f}%)")
        print(f"  Held-out set: {n_held_out} ({100*self.held_out_ratio:.0f}%)")
        print(f"  Random seed: {self.random_seed}")
        
        # Record split info
        self.split_info = {
            'total_samples': n_total,
            'train_samples': n_train,
            'held_out_samples': n_held_out,
            'held_out_ratio': self.held_out_ratio,
            'random_seed': self.random_seed
        }
        
        return self.train_indices, self.held_out_indices
    
    def get_class_distribution(self, indices: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
        """Calculate class distribution for given indices."""
        subset_labels = labels.flatten()[indices]
        unique, counts = np.unique(subset_labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def export_split_info(self) -> str:
        """Export split information to CSV file."""
        if self.train_indices is None:
            self.create_held_out_split()
        
        # Calculate class distributions
        train_dist = self.get_class_distribution(self.train_indices, self.train_labels)
        held_out_dist = self.get_class_distribution(self.held_out_indices, self.train_labels)
        
        # Create detailed CSV
        rows = []
        
        # Overall statistics
        rows.append({
            'category': 'overall',
            'metric': 'total_samples',
            'train_set': len(self.train_indices),
            'held_out_set': len(self.held_out_indices),
            'notes': f'seed={self.random_seed}'
        })
        
        # Per-class statistics
        for class_id in range(9):
            class_name = PathMNISTDataset.CLASS_NAMES[class_id]
            rows.append({
                'category': 'class_distribution',
                'metric': f'class_{class_id}_{class_name}',
                'train_set': train_dist.get(class_id, 0),
                'held_out_set': held_out_dist.get(class_id, 0),
                'notes': ''
            })
        
        df = pd.DataFrame(rows)
        csv_path = self.log_dir / 'data_split_info.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\nSplit info exported to: {csv_path}")
        
        # Also print class distribution
        print("\n=== Class Distribution ===")
        print(f"{'Class':<45} {'Train':>8} {'Held-Out':>10}")
        print("-" * 65)
        for class_id in range(9):
            class_name = PathMNISTDataset.CLASS_NAMES[class_id]
            print(f"{class_id}: {class_name:<40} {train_dist.get(class_id, 0):>8} {held_out_dist.get(class_id, 0):>10}")
        
        return str(csv_path)
    
    def get_train_dataset(self, transform=None) -> PathMNISTDataset:
        """Get the training dataset (excludes held-out samples)."""
        if self.train_indices is None:
            self.create_held_out_split()
        
        return PathMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.train_indices
        )
    
    def get_held_out_dataset(self, transform=None) -> PathMNISTDataset:
        """
        Get the held-out validation dataset.
        WARNING: Use ONLY for final verification!
        """
        if self.held_out_indices is None:
            self.create_held_out_split()
        
        return PathMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.held_out_indices
        )
    
    def get_original_val_dataset(self, transform=None) -> PathMNISTDataset:
        """Get the original MedMNIST validation set."""
        return PathMNISTDataset(
            images=self.val_images,
            labels=self.val_labels,
            transform=transform
        )
    
    def get_original_test_dataset(self, transform=None) -> PathMNISTDataset:
        """Get the original MedMNIST test set."""
        return PathMNISTDataset(
            images=self.test_images,
            labels=self.test_labels,
            transform=transform
        )
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        transform=None,
        pin_memory: bool = None
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for all splits.

        Args:
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes for data loading
            transform: Image transforms to apply
            pin_memory: Pin memory for faster CPU-to-GPU transfer.
                        If None, auto-detect based on CUDA availability.

        Returns:
            Dictionary with keys: 'train', 'held_out', 'val', 'test'
        """
        # Auto-detect pin_memory based on CUDA availability
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        train_dataset = self.get_train_dataset(transform)
        held_out_dataset = self.get_held_out_dataset(transform)
        val_dataset = self.get_original_val_dataset(transform)
        test_dataset = self.get_original_test_dataset(transform)

        return {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            ),
            'held_out': DataLoader(
                held_out_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        }


def main():
    """Test data handler functionality."""
    print("=" * 70)
    print("ViT Model Editing Pipeline - Data Handler")
    print("=" * 70)
    
    # Initialize handler
    handler = DataHandler(
        held_out_ratio=0.2,
        random_seed=42,
        log_dir="logs"
    )
    
    # Load and split data
    handler.load_data()
    handler.create_held_out_split()
    
    # Export split info
    handler.export_split_info()
    
    # Test dataset creation
    train_ds = handler.get_train_dataset()
    held_out_ds = handler.get_held_out_dataset()
    
    print(f"\n=== Dataset Verification ===")
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Held-out dataset size: {len(held_out_ds)}")
    
    # Test sample retrieval
    img, label = train_ds[0]
    print(f"Sample image shape: {img.shape}")
    print(f"Sample label: {label} ({PathMNISTDataset.CLASS_NAMES[label]})")
    
    print("\n✓ Data handler initialized successfully!")


if __name__ == "__main__":
    main()
