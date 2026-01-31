"""
Data Handler Module for ViT Model Editing Pipeline
===================================================
Implements the STRICT 4-Set Data Protocol (Re-split Strategy):

| Operational Set   | Source        | Ratio  | Purpose                                    |
|-------------------|---------------|--------|--------------------------------------------|
| 1. FT-Train       | Official Train| 90%    | Fine-tuning + AlphaEdit covariance stats   |
| 2. Edit-Discovery | Official Train| 10%    | Find "Unseen Errors" for editing targets   |
| 3. FT-Val         | Official Val  | 100%   | Early stopping during fine-tuning only     |
| 4. Test Set       | Official Test | 100%   | Final Comparative Evaluation (Pre vs Post) |

CRITICAL: Edit-Discovery samples are NEVER seen during fine-tuning,
simulating realistic generalization errors for model editing experiments.

Supported Datasets:
- pathmnist: Colon Pathology (9 classes, RGB)
- dermamnist: Dermatoscopy (7 classes, RGB) - Imbalanced
- retinamnist: Retinal OCT (5 classes, RGB) - Fine-grained
- organamnist: Abdominal CT (11 classes, Grayscale) - Shape-based
- bloodmnist: Blood Cell Microscopy (8 classes, RGB)
- tissuemnist: Kidney Cortex Microscopy (8 classes, Grayscale)
- liver4: Liver Fibrosis Staging (4 classes, Grayscale) - F0/F1/F2/F3-F4
- liver2s: Liver Fibrosis Binary (2 classes) - Significant fibrosis (F0-F2 vs F3-F4)
- liver2a: Liver Fibrosis Binary (2 classes) - Any fibrosis (F0 vs F1-F4)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================================
# Dataset Registry: Metadata for supported MedMNIST datasets
# ============================================================================
MEDMNIST_INFO = {
    "pathmnist": {
        "n_channels": 3,
        "n_classes": 9,
        "data_file": "pathmnist_224.npz",
        "task": "multi-class",
        "description": "Colon Pathology - 9 tissue types",
        "class_names": [
            "Adipose", "Background", "Debris", "Lymphocytes", "Mucus",
            "Smooth Muscle", "Normal Colon Mucosa", "Cancer-associated Stroma",
            "Colorectal Adenocarcinoma Epithelium"
        ]
    },
    "dermamnist": {
        "n_channels": 3,
        "n_classes": 7,
        "data_file": "dermamnist_224.npz",
        "task": "multi-class",
        "description": "Dermatoscopy - 7 skin lesion types (Imbalanced)",
        "class_names": [
            "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
            "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesions"
        ]
    },
    "retinamnist": {
        "n_channels": 3,
        "n_classes": 5,
        "data_file": "retinamnist_224.npz",
        "task": "ordinal-regression",
        "description": "Retinal Fundus - 5 diabetic retinopathy grades (Fine-grained)",
        "class_names": [
            "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"
        ]
    },
    "organamnist": {
        "n_channels": 1,  # Grayscale!
        "n_classes": 11,
        "data_file": "organamnist_224.npz",
        "task": "multi-class",
        "description": "Abdominal CT - 11 organ types (Grayscale/Shape)",
        "class_names": [
            "Bladder", "Femur-Left", "Femur-Right", "Heart", "Kidney-Left",
            "Kidney-Right", "Liver", "Lung-Left", "Lung-Right", "Spleen", "Pancreas"
        ]
    },
    "bloodmnist": {
        "n_channels": 3,
        "n_classes": 8,
        "data_file": "bloodmnist_224.npz",
        "task": "multi-class",
        "description": "Blood Cell Microscopy - 8 cell types",
        "class_names": [
            "Basophil", "Eosinophil", "Erythroblast", "Immature Granulocytes",
            "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
        ]
    },
    "tissuemnist": {
        "n_channels": 1,  # Grayscale!
        "n_classes": 8,
        "data_file": "tissuemnist_224.npz",
        "task": "multi-class",
        "description": "Kidney Cortex Microscopy - 8 tissue types (Grayscale)",
        "class_names": [
            "Collecting Duct", "Distal Convoluted Tubule", "Glomerular Endothelial",
            "Interstitial", "Leukocytes", "Podocytes", "Proximal Tubule", "Thick Ascending Limb"
        ]
    },
    # ========================================================================
    # Liver Fibrosis Datasets (Custom format - MERIT paper)
    # ========================================================================
    "liver4": {
        "n_channels": 1,  # Grayscale ultrasound
        "n_classes": 4,
        "data_file": "dataset/",  # Directory containing imgs.npy, labs.npy
        "task": "multi-class",
        "description": "Liver Fibrosis Staging - 4 classes (F0, F1, F2, F3-F4)",
        "class_names": ["F0", "F1", "F2", "F3-F4"],
        "label_mapping": None,  # No remapping, use original labels
        "is_liver_dataset": True
    },
    "liver2s": {
        "n_channels": 1,
        "n_classes": 2,
        "data_file": "dataset/",
        "task": "binary",
        "description": "Liver Fibrosis - Significant (F0-F2 vs F3-F4)",
        "class_names": ["No Significant Fibrosis (F0-F2)", "Significant Fibrosis (F3-F4)"],
        "label_mapping": {0: 0, 1: 0, 2: 0, 3: 1},  # F0,F1,F2 -> 0; F3-F4 -> 1
        "is_liver_dataset": True
    },
    "liver2a": {
        "n_channels": 1,
        "n_classes": 2,
        "data_file": "dataset/",
        "task": "binary",
        "description": "Liver Fibrosis - Any (F0 vs F1-F4)",
        "class_names": ["No Fibrosis (F0)", "Any Fibrosis (F1-F4)"],
        "label_mapping": {0: 0, 1: 1, 2: 1, 3: 1},  # F0 -> 0; F1,F2,F3-F4 -> 1
        "is_liver_dataset": True
    }
}


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get metadata for a MedMNIST dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name not in MEDMNIST_INFO:
        available = ", ".join(MEDMNIST_INFO.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return MEDMNIST_INFO[dataset_name]


class MedMNISTDataset(Dataset):
    """
    Generic Dataset for MedMNIST datasets loaded from .npz files.
    Supports both RGB (3-channel) and Grayscale (1-channel) images.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None,
        indices: np.ndarray = None,
        n_channels: int = 3,
        class_names: list = None
    ):
        """
        Args:
            images: numpy array of shape (N, H, W, C) or (N, H, W) for grayscale
            labels: numpy array of shape (N,) or (N, 1)
            transform: optional transforms to apply
            indices: optional indices to subset the data
            n_channels: number of channels in the original images (1 or 3)
            class_names: list of class names for this dataset
        """
        self.images = images
        self.labels = labels.flatten()
        self.transform = transform
        self.indices = indices if indices is not None else np.arange(len(images))
        self.n_channels = n_channels
        self.class_names = class_names or []

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image = self.images[real_idx]
        label = int(self.labels[real_idx])

        # Handle grayscale images: ensure they have a channel dimension
        if self.n_channels == 1 and image.ndim == 2:
            # Add channel dimension: (H, W) -> (H, W, 1)
            image = np.expand_dims(image, axis=-1)

        # Convert to PIL Image for transforms
        if self.n_channels == 1:
            # Grayscale: squeeze to 2D for PIL
            image = Image.fromarray(image.squeeze().astype(np.uint8), mode='L')
        else:
            # RGB
            image = Image.fromarray(image.astype(np.uint8), mode='RGB')

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            image = np.array(image)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)  # Grayscale -> RGB
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


# Legacy alias for backward compatibility
PathMNISTDataset = MedMNISTDataset


class LiverFibrosisDataset(Dataset):
    """
    Dataset for liver fibrosis data stored in special format.
    Handles float32 normalized images and optional label remapping for binary tasks.

    The original images are z-score normalized floats, which are converted to
    uint8 [0, 255] for PIL compatibility and standard transform pipeline.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None,
        indices: np.ndarray = None,
        label_mapping: dict = None,
        class_names: list = None
    ):
        """
        Args:
            images: numpy array of shape (N, H, W, C), float32 z-score normalized
            labels: numpy array of shape (N,) with original labels (0-3)
            transform: optional transforms to apply
            indices: optional indices to subset the data
            label_mapping: optional dict to remap labels (e.g., {0:0, 1:0, 2:0, 3:1} for binary)
            class_names: list of class names for this dataset
        """
        self.images = images
        self.original_labels = labels.flatten()
        self.transform = transform
        self.indices = indices if indices is not None else np.arange(len(images))
        self.label_mapping = label_mapping
        self.class_names = class_names or []
        self.n_channels = 1  # Liver fibrosis images are grayscale

        # Apply label mapping if provided
        if self.label_mapping is not None:
            self.labels = np.array([self.label_mapping[lbl] for lbl in self.original_labels])
        else:
            self.labels = self.original_labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image = self.images[real_idx]  # Shape: (256, 256, 1), float32
        label = int(self.labels[real_idx])

        # Convert z-score normalized float to uint8 [0, 255]
        # Use min-max normalization per image for consistent contrast
        image_2d = image.squeeze()  # (256, 256)
        img_min, img_max = image_2d.min(), image_2d.max()
        if img_max > img_min:
            image_normalized = (image_2d - img_min) / (img_max - img_min)
        else:
            image_normalized = np.zeros_like(image_2d)
        image_uint8 = (image_normalized * 255).astype(np.uint8)

        # Convert to PIL Image (grayscale)
        image_pil = Image.fromarray(image_uint8, mode='L')

        if self.transform:
            image = self.transform(image_pil)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            image = np.array(image_pil)
            image = np.stack([image] * 3, axis=-1)  # Grayscale -> RGB
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


class DataHandler:
    """
    Handles data loading and the STRICT 4-Set Data Protocol for ViT editing.

    Supports multiple MedMNIST datasets:
    - pathmnist: Colon Pathology (9 classes, RGB)
    - dermamnist: Dermatoscopy (7 classes, RGB) - Imbalanced
    - retinamnist: Retinal OCT (5 classes, RGB) - Fine-grained
    - organamnist: Abdominal CT (11 classes, Grayscale)
    - bloodmnist: Blood Cell Microscopy (8 classes, RGB)
    - tissuemnist: Kidney Cortex Microscopy (8 classes, Grayscale)

    Re-split Strategy:
    ------------------
    - Official Train (90%) -> FT-Train: Fine-tuning + AlphaEdit statistics (K^T K)
    - Official Train (10%) -> Edit-Discovery: Find unseen errors for editing
    - Official Val (100%)  -> FT-Val: Early stopping during fine-tuning
    - Official Test (100%) -> Test Set: Final comparative evaluation

    The Edit-Discovery set is NEVER seen during fine-tuning, ensuring that
    errors found there simulate realistic generalization failures.
    """

    # Default split ratio: 90% FT-Train, 10% Edit-Discovery
    DEFAULT_FT_TRAIN_RATIO = 0.9

    def __init__(
        self,
        dataset_name: str = "pathmnist",
        data_path: str = None,
        ft_train_ratio: float = 0.9,
        random_seed: int = 42,
        log_dir: str = "logs"
    ):
        """
        Args:
            dataset_name: Name of MedMNIST dataset (pathmnist, dermamnist, etc.)
            data_path: Path to .npz file (default: ~/.medmnist/{dataset}_224.npz)
            ft_train_ratio: Fraction of official train for FT-Train (default: 0.9)
                            Remaining (1 - ft_train_ratio) goes to Edit-Discovery
            random_seed: Random seed for reproducible splitting
            log_dir: Directory to save split information and indices
        """
        # Get dataset metadata
        self.dataset_name = dataset_name.lower()
        self.dataset_info = get_dataset_info(self.dataset_name)
        self.n_classes = self.dataset_info["n_classes"]
        self.n_channels = self.dataset_info["n_channels"]
        self.class_names = self.dataset_info["class_names"]

        # Resolve data path
        if data_path is None:
            data_file = self.dataset_info["data_file"]
            data_path = os.path.expanduser(f"~/.medmnist/{data_file}")

        self.data_path = Path(data_path)
        self.ft_train_ratio = ft_train_ratio
        self.random_seed = random_seed
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Data containers (from official MedMNIST splits)
        self.train_images = None  # Official train images
        self.train_labels = None  # Official train labels
        self.val_images = None    # Official val images (FT-Val)
        self.val_labels = None    # Official val labels
        self.test_images = None   # Official test images (Test Set)
        self.test_labels = None   # Official test labels

        # Re-split indices (within official train)
        self.ft_train_indices = None      # 90% of official train -> FT-Train
        self.discovery_indices = None     # 10% of official train -> Edit-Discovery

        # Legacy alias for backward compatibility
        self.train_indices = None         # Alias for ft_train_indices
        self.held_out_indices = None      # Alias for discovery_indices

        # Split indices file path (includes dataset name for isolation)
        self.split_indices_path = self.log_dir / f"{self.dataset_name}_split_indices.pt"

        # Statistics
        self.split_info = {}

        print(f"DataHandler initialized for: {self.dataset_name}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Channels: {self.n_channels} ({'Grayscale' if self.n_channels == 1 else 'RGB'})")
        print(f"  Description: {self.dataset_info['description']}")
        
    def load_data(self) -> Dict[str, np.ndarray]:
        """Load MedMNIST data from .npz file."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"{self.dataset_name} data not found at {self.data_path}. "
                f"Please download it first: "
                f"https://medmnist.com/"
            )

        print(f"Loading {self.dataset_name} from {self.data_path}...")
        data = np.load(self.data_path)

        # Standard MedMNIST split keys
        self.train_images = data['train_images']
        self.train_labels = data['train_labels']
        self.val_images = data['val_images']
        self.val_labels = data['val_labels']
        self.test_images = data['test_images']
        self.test_labels = data['test_labels']

        print(f"  Official Train: {self.train_images.shape}")
        print(f"  Official Val (FT-Val): {self.val_images.shape}")
        print(f"  Official Test (Test Set): {self.test_images.shape}")

        return {
            'train_images': self.train_images,
            'train_labels': self.train_labels,
            'val_images': self.val_images,
            'val_labels': self.val_labels,
            'test_images': self.test_images,
            'test_labels': self.test_labels
        }

    def _save_split_indices(self) -> str:
        """Save split indices to file for reproducibility."""
        torch.save({
            'dataset_name': self.dataset_name,
            'ft_train_indices': self.ft_train_indices,
            'discovery_indices': self.discovery_indices,
            'ft_train_ratio': self.ft_train_ratio,
            'random_seed': self.random_seed
        }, self.split_indices_path)
        print(f"  Split indices saved to: {self.split_indices_path}")
        return str(self.split_indices_path)

    def _load_split_indices(self) -> bool:
        """
        Load split indices from file if exists and matches current config.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.split_indices_path.exists():
            return False

        saved = torch.load(self.split_indices_path, weights_only=False)

        # Verify config matches
        if (saved.get('dataset_name') != self.dataset_name or
            saved.get('ft_train_ratio') != self.ft_train_ratio or
            saved.get('random_seed') != self.random_seed):
            print(f"  Warning: Saved split config differs from current config.")
            print(f"    Saved: dataset={saved.get('dataset_name')}, ratio={saved.get('ft_train_ratio')}, seed={saved.get('random_seed')}")
            print(f"    Current: dataset={self.dataset_name}, ratio={self.ft_train_ratio}, seed={self.random_seed}")
            print(f"  Regenerating split...")
            return False

        self.ft_train_indices = saved['ft_train_indices']
        self.discovery_indices = saved['discovery_indices']

        # Legacy aliases
        self.train_indices = self.ft_train_indices
        self.held_out_indices = self.discovery_indices

        print(f"  Loaded existing split from: {self.split_indices_path}")
        return True

    def create_resplit(self, force: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the 90/10 Re-split on Official Training Data.

        This implements the STRICT 4-Set Protocol:
        - 90% -> FT-Train: For fine-tuning AND AlphaEdit statistics (covariance)
        - 10% -> Edit-Discovery: For finding "unseen errors" to edit

        CRITICAL: Edit-Discovery is NEVER seen during fine-tuning!

        Args:
            force: If True, regenerate split even if saved indices exist

        Returns:
            Tuple of (ft_train_indices, discovery_indices)
        """
        if self.train_images is None:
            self.load_data()

        # Try to load existing split (for reproducibility across runs)
        if not force and self._load_split_indices():
            n_ft_train = len(self.ft_train_indices)
            n_discovery = len(self.discovery_indices)
            print(f"\n=== Re-split Protocol (Loaded) ===")
            print(f"  FT-Train: {n_ft_train} samples")
            print(f"  Edit-Discovery: {n_discovery} samples")
            return self.ft_train_indices, self.discovery_indices

        # Create new split
        n_total = len(self.train_images)
        n_ft_train = int(n_total * self.ft_train_ratio)
        n_discovery = n_total - n_ft_train

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Shuffle indices
        all_indices = np.arange(n_total)
        np.random.shuffle(all_indices)

        # Split: first 90% -> FT-Train, last 10% -> Edit-Discovery
        self.ft_train_indices = all_indices[:n_ft_train]
        self.discovery_indices = all_indices[n_ft_train:]

        # Legacy aliases for backward compatibility
        self.train_indices = self.ft_train_indices
        self.held_out_indices = self.discovery_indices

        # Verify no overlap
        assert len(set(self.ft_train_indices) & set(self.discovery_indices)) == 0, \
            "CRITICAL: Overlap detected between FT-Train and Edit-Discovery!"

        print(f"\n=== Re-split Protocol (4-Set Strategy) ===")
        print(f"  Source: Official Training Set ({n_total} samples)")
        print(f"  ├─ FT-Train: {n_ft_train} ({100*self.ft_train_ratio:.0f}%)")
        print(f"  │   Purpose: Fine-tuning + AlphaEdit covariance (K^T K)")
        print(f"  └─ Edit-Discovery: {n_discovery} ({100*(1-self.ft_train_ratio):.0f}%)")
        print(f"      Purpose: Find unseen errors for editing targets")
        print(f"  Random seed: {self.random_seed}")

        # Record split info
        self.split_info = {
            'total_official_train': n_total,
            'ft_train_samples': n_ft_train,
            'discovery_samples': n_discovery,
            'ft_train_ratio': self.ft_train_ratio,
            'random_seed': self.random_seed,
            'val_samples': len(self.val_images),
            'test_samples': len(self.test_images)
        }

        # Save indices for reproducibility
        self._save_split_indices()

        return self.ft_train_indices, self.discovery_indices

    # Legacy alias for backward compatibility
    def create_held_out_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy alias for create_resplit(). Use create_resplit() instead."""
        return self.create_resplit()
    
    def get_class_distribution(self, indices: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
        """Calculate class distribution for given indices."""
        subset_labels = labels.flatten()[indices]
        unique, counts = np.unique(subset_labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def export_split_info(self) -> str:
        """Export comprehensive split information to CSV file."""
        if self.ft_train_indices is None:
            self.create_resplit()

        # Calculate class distributions for all sets
        ft_train_dist = self.get_class_distribution(self.ft_train_indices, self.train_labels)
        discovery_dist = self.get_class_distribution(self.discovery_indices, self.train_labels)
        val_dist = self.get_class_distribution(np.arange(len(self.val_labels)), self.val_labels)
        test_dist = self.get_class_distribution(np.arange(len(self.test_labels)), self.test_labels)

        # Create detailed CSV
        rows = []

        # Overall statistics
        rows.append({
            'category': 'overall',
            'metric': 'total_samples',
            'ft_train': len(self.ft_train_indices),
            'edit_discovery': len(self.discovery_indices),
            'ft_val': len(self.val_labels),
            'test_set': len(self.test_labels),
            'notes': f'dataset={self.dataset_name}, seed={self.random_seed}, ratio={self.ft_train_ratio}'
        })

        # Per-class statistics
        for class_id in range(self.n_classes):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            rows.append({
                'category': 'class_distribution',
                'metric': f'class_{class_id}_{class_name}',
                'ft_train': ft_train_dist.get(class_id, 0),
                'edit_discovery': discovery_dist.get(class_id, 0),
                'ft_val': val_dist.get(class_id, 0),
                'test_set': test_dist.get(class_id, 0),
                'notes': ''
            })

        df = pd.DataFrame(rows)
        csv_path = self.log_dir / f'{self.dataset_name}_data_split_info.csv'
        df.to_csv(csv_path, index=False)

        print(f"\nSplit info exported to: {csv_path}")

        # Print 4-Set Protocol summary
        print(f"\n=== 4-Set Protocol Summary ({self.dataset_name}) ===")
        print(f"{'Set':<20} {'Samples':>10} {'Source':<25} {'Purpose':<35}")
        print("-" * 95)
        print(f"{'FT-Train':<20} {len(self.ft_train_indices):>10} {'Official Train (90%)':<25} {'Fine-tuning + AlphaEdit stats':<35}")
        print(f"{'Edit-Discovery':<20} {len(self.discovery_indices):>10} {'Official Train (10%)':<25} {'Find unseen errors for editing':<35}")
        print(f"{'FT-Val':<20} {len(self.val_labels):>10} {'Official Val (100%)':<25} {'Early stopping only':<35}")
        print(f"{'Test Set':<20} {len(self.test_labels):>10} {'Official Test (100%)':<25} {'Final comparative evaluation':<35}")

        # Print class distribution
        print(f"\n=== Class Distribution ({self.dataset_name}, {self.n_classes} classes) ===")
        print(f"{'Class':<35} {'FT-Train':>10} {'Discovery':>10} {'FT-Val':>10} {'Test':>10}")
        print("-" * 80)
        for class_id in range(self.n_classes):
            class_name = self.class_names[class_id][:30] if class_id < len(self.class_names) else f"Class_{class_id}"
            print(f"{class_id}: {class_name:<32} "
                  f"{ft_train_dist.get(class_id, 0):>10} "
                  f"{discovery_dist.get(class_id, 0):>10} "
                  f"{val_dist.get(class_id, 0):>10} "
                  f"{test_dist.get(class_id, 0):>10}")

        return str(csv_path)
    
    def get_ft_train_dataset(self, transform=None) -> MedMNISTDataset:
        """
        Get FT-Train dataset (90% of official train).

        Purpose:
        - Fine-tuning the model
        - Computing AlphaEdit covariance statistics (K^T K)
        """
        if self.ft_train_indices is None:
            self.create_resplit()

        return MedMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.ft_train_indices,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_discovery_dataset(self, transform=None) -> MedMNISTDataset:
        """
        Get Edit-Discovery dataset (10% of official train).

        Purpose:
        - Finding "unseen errors" for editing targets
        - NEVER seen during fine-tuning (simulates generalization errors)

        WARNING: Do NOT use this for training or statistics computation!
        """
        if self.discovery_indices is None:
            self.create_resplit()

        return MedMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.discovery_indices,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_val_dataset(self, transform=None) -> MedMNISTDataset:
        """
        Get FT-Val dataset (official validation set).

        Purpose:
        - Early stopping during fine-tuning ONLY
        - Discarded after training phase
        """
        return MedMNISTDataset(
            images=self.val_images,
            labels=self.val_labels,
            transform=transform,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_test_dataset(self, transform=None) -> MedMNISTDataset:
        """
        Get Test Set (official test set).

        Purpose:
        - Final Comparative Evaluation (Pre-Edit vs Post-Edit)
        - NEVER used for training, validation, or editing decisions
        """
        return MedMNISTDataset(
            images=self.test_images,
            labels=self.test_labels,
            transform=transform,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    # ================================================================
    # Baseline Methods
    # ================================================================

    def get_combined_ft_train_with_errors(
        self,
        error_indices: np.ndarray,
        transform=None
    ) -> MedMNISTDataset:
        """
        Create combined dataset: FT-Train + Error samples from Edit-Discovery.

        For Baseline 1 (Retrain from Scratch):
        - Combines 90% FT-Train with error samples found in Edit-Discovery
        - Error samples are added to training data

        Args:
            error_indices: Indices of error samples within Edit-Discovery set
            transform: Image transforms

        Returns:
            Combined MedMNISTDataset
        """
        if self.ft_train_indices is None:
            self.create_resplit()

        # Map error indices from discovery set to original train indices
        discovery_error_original_indices = self.discovery_indices[error_indices]

        # Combine FT-Train indices with error sample indices
        combined_indices = np.concatenate([
            self.ft_train_indices,
            discovery_error_original_indices
        ])

        print(f"  Combined dataset: FT-Train ({len(self.ft_train_indices)}) + "
              f"Errors ({len(error_indices)}) = {len(combined_indices)} samples")

        return MedMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=combined_indices,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_error_samples_dataset(
        self,
        error_indices: np.ndarray,
        transform=None
    ) -> MedMNISTDataset:
        """
        Get only the error samples from Edit-Discovery set.

        For Baseline 2 (Finetune on Errors):
        - Returns only the misclassified samples for fine-tuning

        Args:
            error_indices: Indices of error samples within Edit-Discovery set
            transform: Image transforms

        Returns:
            MedMNISTDataset containing only error samples
        """
        if self.discovery_indices is None:
            self.create_resplit()

        # Map to original train indices
        error_original_indices = self.discovery_indices[error_indices]

        print(f"  Error samples dataset: {len(error_original_indices)} samples")

        return MedMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=error_original_indices,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    # Legacy aliases for backward compatibility
    def get_train_dataset(self, transform=None) -> MedMNISTDataset:
        """Legacy alias for get_ft_train_dataset()."""
        return self.get_ft_train_dataset(transform)

    def get_held_out_dataset(self, transform=None) -> MedMNISTDataset:
        """Legacy alias for get_discovery_dataset()."""
        return self.get_discovery_dataset(transform)

    def get_original_val_dataset(self, transform=None) -> MedMNISTDataset:
        """Legacy alias for get_val_dataset()."""
        return self.get_val_dataset(transform)

    def get_original_test_dataset(self, transform=None) -> MedMNISTDataset:
        """Legacy alias for get_test_dataset()."""
        return self.get_test_dataset(transform)
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        transform=None,
        pin_memory: bool = None
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for the 4-Set Protocol.

        Args:
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes for data loading
            transform: Image transforms to apply
            pin_memory: Pin memory for faster CPU-to-GPU transfer.
                        If None, auto-detect based on CUDA availability.

        Returns:
            Dictionary with keys matching the 4-Set Protocol:
            - 'ft_train': FT-Train loader (90% official train) - for fine-tuning + stats
            - 'discovery': Edit-Discovery loader (10% official train) - for finding errors
            - 'val': FT-Val loader (official val) - for early stopping
            - 'test': Test Set loader (official test) - for final evaluation

            Also includes legacy aliases:
            - 'train': alias for 'ft_train'
            - 'held_out': alias for 'discovery'
        """
        # Auto-detect pin_memory based on CUDA availability
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        ft_train_dataset = self.get_ft_train_dataset(transform)
        discovery_dataset = self.get_discovery_dataset(transform)
        val_dataset = self.get_val_dataset(transform)
        test_dataset = self.get_test_dataset(transform)

        ft_train_loader = DataLoader(
            ft_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        discovery_loader = DataLoader(
            discovery_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return {
            # Primary keys (4-Set Protocol)
            'ft_train': ft_train_loader,
            'discovery': discovery_loader,
            'val': val_loader,
            'test': test_loader,
            # Legacy aliases for backward compatibility
            'train': ft_train_loader,
            'held_out': discovery_loader
        }


# ============================================================================
# Liver Fibrosis DataHandler (Custom dataset with no predefined splits)
# ============================================================================

class LiverFibrosisDataHandler(DataHandler):
    """
    DataHandler for liver fibrosis dataset stored in custom .npy format.

    Key differences from standard DataHandler:
    1. Data stored as .npy files (imgs.npy, labs.npy) instead of .npz
    2. Images are float32 z-score normalized (not uint8)
    3. No predefined train/val/test splits - creates stratified splits from scratch
    4. Supports label remapping for binary classification modes

    Split Strategy - Aligned with MERIT paper (reference/MERIT/dataset.py):
    When cross_val=False, MERIT uses fixed 60/20/20 split:
    - 60% -> Train (then 90/10 -> FT-Train/Edit-Discovery)
    - 20% -> Validation (early stopping)
    - 20% -> Test (final evaluation)

    This matches MERIT's non-cross-validation mode exactly.
    """

    # Split ratios aligned with MERIT's non-CV mode (cross_val=False)
    # See reference/MERIT/dataset.py lines 46-65
    TRAIN_RATIO = 0.60  # 60% -> Official Train
    VAL_RATIO = 0.20    # 20% -> FT-Val
    TEST_RATIO = 0.20   # 20% -> Test Set

    def __init__(
        self,
        dataset_name: str = "liver4",
        data_path: str = None,
        ft_train_ratio: float = 0.9,
        random_seed: int = 42,
        log_dir: str = "logs"
    ):
        """
        Args:
            dataset_name: liver4, liver2s, or liver2a
            data_path: Path to directory containing imgs.npy and labs.npy
            ft_train_ratio: Fraction of train set for FT-Train (default: 0.9)
            random_seed: Random seed for reproducible splitting
            log_dir: Directory to save split information
        """
        # Get dataset metadata
        self.dataset_name = dataset_name.lower()
        self.dataset_info = get_dataset_info(self.dataset_name)
        self.n_classes = self.dataset_info["n_classes"]
        self.n_channels = self.dataset_info["n_channels"]
        self.class_names = self.dataset_info["class_names"]
        self.label_mapping = self.dataset_info.get("label_mapping", None)

        # Resolve data path
        if data_path is None:
            data_path = self.dataset_info["data_file"]

        self.data_path = Path(data_path)
        self.ft_train_ratio = ft_train_ratio
        self.random_seed = random_seed
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.all_images = None    # All 703 images
        self.all_labels = None    # All 703 labels (original)
        self.train_images = None  # "Official train" images (70%)
        self.train_labels = None  # "Official train" labels
        self.val_images = None    # FT-Val images (15%)
        self.val_labels = None    # FT-Val labels
        self.test_images = None   # Test Set images (15%)
        self.test_labels = None   # Test Set labels

        # Split indices
        self.train_split_indices = None    # Indices into all_images for train
        self.val_split_indices = None      # Indices into all_images for val
        self.test_split_indices = None     # Indices into all_images for test
        self.ft_train_indices = None       # Indices into train_images for FT-Train
        self.discovery_indices = None      # Indices into train_images for Discovery

        # Legacy aliases
        self.train_indices = None
        self.held_out_indices = None

        # Split indices file path
        self.split_indices_path = self.log_dir / f"{self.dataset_name}_split_indices.pt"

        # Statistics
        self.split_info = {}

        print(f"LiverFibrosisDataHandler initialized for: {self.dataset_name}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Channels: {self.n_channels} (Grayscale)")
        print(f"  Description: {self.dataset_info['description']}")
        if self.label_mapping:
            print(f"  Label mapping: {self.label_mapping}")

    def load_data(self) -> Dict[str, np.ndarray]:
        """Load liver fibrosis data from .npy files and create train/val/test splits."""
        imgs_path = self.data_path / "imgs.npy"
        labs_path = self.data_path / "labs.npy"

        if not imgs_path.exists() or not labs_path.exists():
            raise FileNotFoundError(
                f"Liver fibrosis data not found at {self.data_path}. "
                f"Expected files: imgs.npy, labs.npy"
            )

        print(f"Loading {self.dataset_name} from {self.data_path}...")

        # Load raw data (imgs is array of dicts, labs is array of ints)
        imgs_raw = np.load(imgs_path, allow_pickle=True)
        labs_raw = np.load(labs_path, allow_pickle=True)

        # Extract image arrays from dict format {'4': array}
        images_list = []
        for img_dict in imgs_raw:
            # Each img_dict is like {'4': np.ndarray of shape (256, 256, 1)}
            img_array = list(img_dict.values())[0]
            images_list.append(img_array)

        self.all_images = np.array(images_list, dtype=np.float32)  # (703, 256, 256, 1)
        self.all_labels = labs_raw.astype(np.int64)  # (703,)

        print(f"  Total samples: {len(self.all_images)}")
        print(f"  Image shape: {self.all_images[0].shape}")
        print(f"  Label distribution (original): {dict(zip(*np.unique(self.all_labels, return_counts=True)))}")

        # Create stratified train/val/test split
        self._create_initial_splits()

        return {
            'train_images': self.train_images,
            'train_labels': self.train_labels,
            'val_images': self.val_images,
            'val_labels': self.val_labels,
            'test_images': self.test_images,
            'test_labels': self.test_labels
        }

    def _create_initial_splits(self):
        """Create stratified train/val/test splits from all data."""
        n_total = len(self.all_images)
        all_indices = np.arange(n_total)

        # First split: train vs (val+test)
        train_idx, val_test_idx = train_test_split(
            all_indices,
            test_size=(self.VAL_RATIO + self.TEST_RATIO),
            random_state=self.random_seed,
            stratify=self.all_labels
        )

        # Second split: val vs test
        # Calculate relative test size within val_test portion
        relative_test_size = self.TEST_RATIO / (self.VAL_RATIO + self.TEST_RATIO)
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=relative_test_size,
            random_state=self.random_seed,
            stratify=self.all_labels[val_test_idx]
        )

        # Store split indices
        self.train_split_indices = train_idx
        self.val_split_indices = val_idx
        self.test_split_indices = test_idx

        # Create split datasets
        self.train_images = self.all_images[train_idx]
        self.train_labels = self.all_labels[train_idx]
        self.val_images = self.all_images[val_idx]
        self.val_labels = self.all_labels[val_idx]
        self.test_images = self.all_images[test_idx]
        self.test_labels = self.all_labels[test_idx]

        print(f"  Created stratified splits:")
        print(f"    Train: {len(train_idx)} ({100*len(train_idx)/n_total:.1f}%)")
        print(f"    Val: {len(val_idx)} ({100*len(val_idx)/n_total:.1f}%)")
        print(f"    Test: {len(test_idx)} ({100*len(test_idx)/n_total:.1f}%)")

    def _save_split_indices(self) -> str:
        """Save all split indices to file for reproducibility."""
        torch.save({
            'dataset_name': self.dataset_name,
            'train_split_indices': self.train_split_indices,
            'val_split_indices': self.val_split_indices,
            'test_split_indices': self.test_split_indices,
            'ft_train_indices': self.ft_train_indices,
            'discovery_indices': self.discovery_indices,
            'ft_train_ratio': self.ft_train_ratio,
            'random_seed': self.random_seed
        }, self.split_indices_path)
        print(f"  Split indices saved to: {self.split_indices_path}")
        return str(self.split_indices_path)

    def _load_split_indices(self) -> bool:
        """Load split indices from file if exists and matches config."""
        if not self.split_indices_path.exists():
            return False

        saved = torch.load(self.split_indices_path, weights_only=False)

        # Verify config matches
        if (saved.get('dataset_name') != self.dataset_name or
            saved.get('ft_train_ratio') != self.ft_train_ratio or
            saved.get('random_seed') != self.random_seed):
            print(f"  Warning: Saved split config differs. Regenerating...")
            return False

        # Load all indices
        self.train_split_indices = saved['train_split_indices']
        self.val_split_indices = saved['val_split_indices']
        self.test_split_indices = saved['test_split_indices']
        self.ft_train_indices = saved['ft_train_indices']
        self.discovery_indices = saved['discovery_indices']

        # Legacy aliases
        self.train_indices = self.ft_train_indices
        self.held_out_indices = self.discovery_indices

        print(f"  Loaded existing split from: {self.split_indices_path}")
        return True

    def create_resplit(self, force: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 90/10 FT-Train/Edit-Discovery split on the train portion.

        This is called AFTER load_data() has created train/val/test splits.
        """
        if self.train_images is None:
            self.load_data()

        # Try to load existing split
        if not force and self._load_split_indices():
            # Recreate image splits from loaded indices
            self.train_images = self.all_images[self.train_split_indices]
            self.train_labels = self.all_labels[self.train_split_indices]
            self.val_images = self.all_images[self.val_split_indices]
            self.val_labels = self.all_labels[self.val_split_indices]
            self.test_images = self.all_images[self.test_split_indices]
            self.test_labels = self.all_labels[self.test_split_indices]

            n_ft_train = len(self.ft_train_indices)
            n_discovery = len(self.discovery_indices)
            print(f"\n=== Re-split Protocol (Loaded) ===")
            print(f"  FT-Train: {n_ft_train} samples")
            print(f"  Edit-Discovery: {n_discovery} samples")
            return self.ft_train_indices, self.discovery_indices

        # Create new 90/10 split within train set
        n_train = len(self.train_images)
        n_ft_train = int(n_train * self.ft_train_ratio)

        np.random.seed(self.random_seed)
        all_train_indices = np.arange(n_train)
        np.random.shuffle(all_train_indices)

        self.ft_train_indices = all_train_indices[:n_ft_train]
        self.discovery_indices = all_train_indices[n_ft_train:]

        # Legacy aliases
        self.train_indices = self.ft_train_indices
        self.held_out_indices = self.discovery_indices

        print(f"\n=== Re-split Protocol (4-Set Strategy) ===")
        print(f"  Source: Train Set ({n_train} samples)")
        print(f"  ├─ FT-Train: {n_ft_train} ({100*self.ft_train_ratio:.0f}%)")
        print(f"  └─ Edit-Discovery: {len(self.discovery_indices)} ({100*(1-self.ft_train_ratio):.0f}%)")

        # Record split info
        self.split_info = {
            'total_samples': len(self.all_images),
            'train_samples': n_train,
            'ft_train_samples': n_ft_train,
            'discovery_samples': len(self.discovery_indices),
            'val_samples': len(self.val_images),
            'test_samples': len(self.test_images)
        }

        # Save indices
        self._save_split_indices()

        return self.ft_train_indices, self.discovery_indices

    def get_ft_train_dataset(self, transform=None) -> LiverFibrosisDataset:
        """Get FT-Train dataset using LiverFibrosisDataset class."""
        if self.ft_train_indices is None:
            self.create_resplit()

        return LiverFibrosisDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.ft_train_indices,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_discovery_dataset(self, transform=None) -> LiverFibrosisDataset:
        """Get Edit-Discovery dataset using LiverFibrosisDataset class."""
        if self.discovery_indices is None:
            self.create_resplit()

        return LiverFibrosisDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.discovery_indices,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_val_dataset(self, transform=None) -> LiverFibrosisDataset:
        """Get FT-Val dataset using LiverFibrosisDataset class."""
        return LiverFibrosisDataset(
            images=self.val_images,
            labels=self.val_labels,
            transform=transform,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_test_dataset(self, transform=None) -> LiverFibrosisDataset:
        """Get Test Set using LiverFibrosisDataset class."""
        return LiverFibrosisDataset(
            images=self.test_images,
            labels=self.test_labels,
            transform=transform,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_combined_ft_train_with_errors(
        self,
        error_indices: np.ndarray,
        transform=None
    ) -> LiverFibrosisDataset:
        """Create combined FT-Train + Error samples dataset."""
        if self.ft_train_indices is None:
            self.create_resplit()

        combined_indices = np.concatenate([
            self.ft_train_indices,
            self.discovery_indices[error_indices]
        ])

        print(f"  Combined dataset: FT-Train ({len(self.ft_train_indices)}) + "
              f"Errors ({len(error_indices)}) = {len(combined_indices)} samples")

        return LiverFibrosisDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=combined_indices,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_error_samples_dataset(
        self,
        error_indices: np.ndarray,
        transform=None
    ) -> LiverFibrosisDataset:
        """Get only error samples from Edit-Discovery set."""
        if self.discovery_indices is None:
            self.create_resplit()

        error_original_indices = self.discovery_indices[error_indices]
        print(f"  Error samples dataset: {len(error_original_indices)} samples")

        return LiverFibrosisDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=error_original_indices,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )


# ============================================================================
# Factory Function
# ============================================================================

def get_data_handler(dataset_name: str, **kwargs) -> Union[DataHandler, LiverFibrosisDataHandler]:
    """
    Factory function to get the appropriate DataHandler for a dataset.

    Args:
        dataset_name: Name of the dataset (pathmnist, dermamnist, liver4, etc.)
        **kwargs: Additional arguments passed to DataHandler constructor

    Returns:
        DataHandler or LiverFibrosisDataHandler instance
    """
    dataset_name = dataset_name.lower()
    dataset_info = get_dataset_info(dataset_name)

    if dataset_info.get("is_liver_dataset", False):
        return LiverFibrosisDataHandler(dataset_name=dataset_name, **kwargs)
    else:
        return DataHandler(dataset_name=dataset_name, **kwargs)


def main():
    """Test data handler functionality with 4-Set Protocol."""
    print("=" * 70)
    print("ViT Model Editing Pipeline - Data Handler (Multi-Dataset Support)")
    print("=" * 70)

    # Test with default dataset (pathmnist)
    dataset_name = "pathmnist"
    print(f"\n>>> Testing with {dataset_name} <<<")

    handler = DataHandler(
        dataset_name=dataset_name,
        ft_train_ratio=0.9,
        random_seed=42,
        log_dir="logs"
    )

    # Load and create re-split
    handler.load_data()
    handler.create_resplit()

    # Export split info
    handler.export_split_info()

    # Test dataset creation
    ft_train_ds = handler.get_ft_train_dataset()
    discovery_ds = handler.get_discovery_dataset()
    val_ds = handler.get_val_dataset()
    test_ds = handler.get_test_dataset()

    print(f"\n=== Dataset Verification ===")
    print(f"FT-Train dataset size: {len(ft_train_ds)}")
    print(f"Edit-Discovery dataset size: {len(discovery_ds)}")
    print(f"FT-Val dataset size: {len(val_ds)}")
    print(f"Test Set dataset size: {len(test_ds)}")

    # Test sample retrieval
    img, label = ft_train_ds[0]
    print(f"\nSample image shape: {img.shape}")
    class_name = handler.class_names[label] if label < len(handler.class_names) else f"Class_{label}"
    print(f"Sample label: {label} ({class_name})")

    # Test dataloaders
    loaders = handler.get_dataloaders(batch_size=32)
    print(f"\n=== DataLoader Keys ===")
    for key in loaders.keys():
        print(f"  '{key}': {len(loaders[key])} batches")

    # Print metadata
    print(f"\n=== Dataset Metadata ===")
    print(f"  Dataset: {handler.dataset_name}")
    print(f"  Classes: {handler.n_classes}")
    print(f"  Channels: {handler.n_channels}")

    print("\n[OK] Data handler with multi-dataset support initialized successfully!")


if __name__ == "__main__":
    main()
