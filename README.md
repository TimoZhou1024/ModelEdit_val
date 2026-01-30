# ViT Model Editing Pipeline

Transfer LLM editing techniques (AlphaEdit + ASTRA) to Vision Transformers for correcting misclassified samples on MedMNIST datasets.

## Supported Datasets

| Dataset | Classes | Channels | Task Type | Description |
|---------|---------|----------|-----------|-------------|
| **pathmnist** | 9 | RGB | Multi-class | Colon Pathology - 9 tissue types |
| **dermamnist** | 7 | RGB | Multi-class | Dermatoscopy - 7 skin lesion types (Imbalanced) |
| **retinamnist** | 5 | RGB | Ordinal | Retinal Fundus - 5 diabetic retinopathy grades (Fine-grained) |
| **organamnist** | 11 | Grayscale | Multi-class | Abdominal CT - 11 organ types (Shape-based) |
| **bloodmnist** | 8 | RGB | Multi-class | Blood Cell Microscopy - 8 cell types |
| **tissuemnist** | 8 | Grayscale | Multi-class | Kidney Cortex Microscopy - 8 tissue types |

**Note**: Grayscale datasets are automatically converted to 3-channel RGB for ViT compatibility.

## Project Goal

1. **Data Splitting (4-Set Protocol)**: Implement strict data isolation with FT-Train, Edit-Discovery, FT-Val, and Test Set
2. **Fine-tuning**: Train `vit-base-patch16-224` on FT-Train set
3. **Locate Layers**: Use **ASTRA-style Causal Tracing** on Edit-Discovery set to identify significant layers
4. **Edit Weights**: Apply **AlphaEdit** (MLP layers) or **Head Editing** (classifier only) to correct errors
5. **Baselines**: Compare with traditional approaches (Retrain, Finetune-on-Errors)
6. **Evaluate**: Run **Comparative Evaluation** on Official Test Set (Pre-Edit vs Post-Edit)

## Project Structure

```
E:\ModelEdit_val\
├── src/
│   ├── data_handler.py      # Multi-dataset support + 4-Set Protocol
│   ├── trainer.py           # ViT fine-tuning with dynamic num_classes
│   ├── locator.py           # ASTRA-style causal tracing for layer importance
│   ├── editor.py            # AlphaEdit + HeadEditor (multi-dataset support)
│   ├── evaluator.py         # Comparative evaluation on Official Test Set
│   └── main.py              # CLI entry point (--dataset argument)
├── checkpoints/
│   ├── vit_{dataset}_finetuned.pt     # Fine-tuned model (per dataset)
│   ├── vit_{dataset}_edited.pt        # AlphaEdit edited model
│   ├── vit_{dataset}_head_edited.pt   # Head edited model
│   ├── vit_{dataset}_retrained.pt     # Baseline 1: Retrained model
│   ├── vit_{dataset}_finetuned_on_errors.pt  # Baseline 2: Finetuned on errors
│   └── projection_cache.pt            # Cached null-space projections
├── logs/
│   └── {timestamp}/                 # Timestamped run logs
│       ├── {dataset}_split_indices.pt    # Saved split indices
│       ├── {dataset}_data_split_info.csv # 4-Set Protocol statistics
│       ├── {dataset}_training_metrics.csv # Training loss/accuracy
│       ├── layer_importance.csv     # Per-sample causal tracing scores
│       ├── layer_statistics.csv     # Aggregated layer statistics
│       ├── edit_log.csv             # AlphaEdit records
│       └── head_edit_log.csv        # Head Editing records
├── results/
│   └── {timestamp}/                 # Timestamped run results
│       ├── comparative_evaluation.csv  # Pre vs Post comparison
│       ├── confusion_matrix_orig.csv   # Pre-edit confusion matrix
│       ├── confusion_matrix_edit.csv   # Post-edit confusion matrix
│       ├── confusion_matrix.png        # Visualization
│       ├── baseline_retrain_summary.csv      # Baseline 1 results
│       └── baseline_finetune_errors_summary.csv  # Baseline 2 results
├── reference/                        # Reference implementations
│   ├── AlphaEdit/                   # Null-space projection method
│   └── ASTRA/                       # Activation steering method
├── pyproject.toml                   # uv package configuration
└── README.md
```

## Quick Start

### Prerequisites (using uv package manager)

```bash
# Install uv (if not already installed)
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environment Setup

```bash
cd E:\ModelEdit_val

# Create virtual environment and install dependencies
uv venv
uv sync

# Activate virtual environment (optional, uv run works without activation)
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### Data Location

MedMNIST datasets should be at: `~/.medmnist/{dataset}_224.npz`

Download from: https://medmnist.com/

```bash
# Example paths:
~/.medmnist/pathmnist_224.npz
~/.medmnist/dermamnist_224.npz
~/.medmnist/retinamnist_224.npz
~/.medmnist/organamnist_224.npz
~/.medmnist/bloodmnist_224.npz
~/.medmnist/tissuemnist_224.npz
```

### Running the Pipeline

```bash
# Run complete pipeline with default dataset (pathmnist)
uv run python src/main.py --stage full --timestamp

# Run with a specific dataset
uv run python src/main.py --stage full --dataset dermamnist --timestamp
uv run python src/main.py --stage full --dataset organamnist --timestamp

# Run with custom run name
uv run python src/main.py --stage full --dataset pathmnist --run-name experiment_v1

# Run individual stages
uv run python src/main.py --stage data --dataset dermamnist
uv run python src/main.py --stage train --dataset dermamnist --epochs 10
uv run python src/main.py --stage locate --dataset dermamnist
uv run python src/main.py --stage edit --dataset dermamnist
uv run python src/main.py --stage eval --dataset dermamnist
```

## Pipeline Stages

### Stage 1: Data Preparation (`--stage data`)

Implements the **4-Set Protocol** for rigorous data isolation:

| Operational Set | Source | Ratio | Purpose |
|-----------------|--------|-------|---------|
| **FT-Train** | Official Train | 90% | Fine-tuning + AlphaEdit covariance (K^T K) |
| **Edit-Discovery** | Official Train | 10% | Find "unseen errors" for editing targets |
| **FT-Val** | Official Val | 100% | Early stopping during fine-tuning only |
| **Test Set** | Official Test | 100% | Final Comparative Evaluation |

- **CRITICAL**: Edit-Discovery samples are NEVER seen during fine-tuning
- Split indices saved to `logs/split_indices.pt` for reproducibility
- Exports: `logs/data_split_info.csv`

### Stage 2: Fine-tuning (`--stage train`)

- Uses `google/vit-base-patch16-224` with 9-class head
- Trains on **FT-Train set only** (90% of official train)
- Uses **FT-Val** for early stopping
- **Auto-detects GPU/CPU** for optimal performance
- **Skips training if checkpoint exists** (loads from checkpoint)
- Exports: `checkpoints/vit_pathmnist_finetuned.pt`, `logs/training_metrics.csv`

### Stage 3: Layer Localization (`--stage locate`)

Implements **ASTRA-style Causal Tracing** on **Edit-Discovery set**:

**Core Algorithm:**
1. **Find Errors**: Identify misclassified samples from Edit-Discovery set (unseen during training)
2. **Corrupt Input**: Add Gaussian noise to patch embeddings (positions 1-196)
3. **Run Corrupted Forward**: Observe prediction probability drop
4. **Restore & Measure**: For each layer, restore clean activations and measure probability recovery
5. **Importance Score**: Higher recovery = more important layer for the error

Exports: `logs/layer_importance.csv`, `logs/layer_statistics.csv`

### Stage 4: Weight Editing (`--stage edit`)

Two editing methods available, both using the **4-Set Protocol**:

- **stats_loader**: FT-Train (for covariance/Fisher computation)
- **target_loader**: Edit-Discovery (for finding samples to fix)

#### AlphaEdit (default: `--edit-method alphaedit`)

Implements **AlphaEdit** null-space projection for ViT:

1. **Collect K vectors**: Input activations at target layers
2. **Compute Null-Space Projection**: P = I - K₀(K₀ᵀK₀)⁻¹K₀ᵀ
3. **Optimize Target Z**: Gradient descent to find correct output representation
4. **Apply Update**: ΔW = P × (target adjustment)

**Key Feature**: Uses ASTRA results to determine which layers to edit!

#### Head Editing (`--edit-method head`)

A simpler, faster alternative that modifies only the classification head:

1. **Freeze Backbone**: Keep all transformer layers fixed
2. **Compute Fisher Information**: For EWC regularization
3. **Optimize Classifier**: Gradient descent on `model.classifier` only
4. **EWC Regularization**: Prevents catastrophic forgetting

**Key Features**:
- Only 6,912 parameters modified (768×9 + 9) vs millions in AlphaEdit
- Processes all samples in one batch (faster)
- Optional closed-form solution with `--closed-form`

Exports: `logs/edit_log.csv` or `logs/head_edit_log.csv`, `checkpoints/vit_pathmnist_*.pt`

### Stage 5: Evaluation (`--stage eval`)

Runs **Comparative Evaluation** on **Official Test Set** (4-Set Protocol):

- Compares Pre-Edit vs Post-Edit model performance
- **Metrics**:
  - **Accuracy Delta**: Change in accuracy (positive = improvement)
  - **Stability**: Fraction of correct samples that remained correct
  - **Fix Rate**: Fraction of error samples that became correct
  - **Regression Rate**: Fraction of correct samples that became wrong
- Generates confusion matrices for both models
- Exports: `results/comparative_evaluation.csv`, `results/confusion_matrix_*.csv`

### Baseline Methods

Two baseline methods are provided for comparison with AlphaEdit:

#### Baseline 1: Retrain from Scratch (`--stage baseline1`)

Add error samples to FT-Train dataset, then train a NEW model from scratch.

**Process:**
1. Load finetuned model to identify misclassified samples
2. Combine FT-Train + error samples into new training set
3. Reinitialize model from pretrained weights
4. Train from scratch on combined dataset
5. Run 4-level evaluation comparing original vs retrained model

**Use Case:** Tests whether simply including error samples in training data helps.

#### Baseline 2: Finetune on Errors (`--stage baseline2`)

Load the finetuned model and continue training ONLY on error samples.

**Process:**
1. Load finetuned model checkpoint
2. Create dataset with only error samples
3. Continue finetuning with lower learning rate
4. Run 4-level evaluation comparing original vs finetuned model

**Use Case:** Tests whether targeted finetuning on errors helps without full retraining.

#### 4-Level Evaluation (Same as AlphaEdit)

Both baselines use the same evaluation framework:

| Level | Dataset | Purpose |
|-------|---------|---------|
| 1. Edit Samples | Error samples | Measure fix rate |
| 2. FT-Train Samples | Training set | Knowledge preservation |
| 3. Test Set | Official test | Generalization |
| 4. Edit-Discovery Set | Discovery set | Overall improvement |

## Command Line Arguments

### Stage Selection
| Argument | Description |
|----------|-------------|
| `--stage` | Pipeline stage: `data`, `train`, `locate`, `edit`, `eval`, `full`, `baseline1`, `baseline2` |

### Dataset Selection
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `pathmnist` | MedMNIST dataset: `pathmnist`, `dermamnist`, `retinamnist`, `organamnist`, `bloodmnist`, `tissuemnist` |

### Data Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | `~/.medmnist/{dataset}_224.npz` | MedMNIST data file |
| `--ft-train-ratio` | 0.9 | Fraction of official train for FT-Train (remaining goes to Edit-Discovery) |
| `--seed` | 42 | Random seed for reproducibility |

### Training Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--resume` | False | Resume training from checkpoint |
| `--pin-memory` | auto | Enable pin_memory for faster CPU-to-GPU transfer |
| `--no-pin-memory` | False | Disable pin_memory (use when CUDA unavailable) |

### Layer Localization Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--num-ablations` | 32 | Number of corrupted samples for averaging |
| `--max-samples` | 50 | Max misclassified samples to analyze |

### Editing Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--edit-method` | `alphaedit` | Editing method: `alphaedit` (MLP layers) or `head` (classifier only) |
| `--edit-layers` | None | Manually specify layers to edit (e.g., `9 10 11`) |
| `--num-edit-layers` | 3 | Number of top ASTRA layers to edit |
| `--no-astra-layers` | False | Disable ASTRA layer selection, use default [9,10,11] |
| `--max-edits` | 30 | Maximum samples to edit |
| `--projection-samples` | 500 | Number of FT-Train samples for projection matrix construction |
| `--nullspace-threshold` | 1e-2 | Threshold for null-space eigenvalue selection |

### Head Editing Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--head-lr` | 0.01 | Learning rate for head editing |
| `--head-steps` | 50 | Number of optimization steps |
| `--ewc-lambda` | 1000.0 | EWC regularization strength |
| `--closed-form` | False | Use closed-form solution (faster, less precise) |

### Baseline Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--baseline-epochs` | 10 | Number of epochs for baseline training |
| `--baseline-lr` | 1e-5 | Learning rate for baseline finetuning (Baseline 2) |

### Output Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint-dir` | `checkpoints` | Directory for model checkpoints |
| `--log-dir` | `logs` | Directory for log files |
| `--results-dir` | `results` | Directory for evaluation results |
| `--timestamp` | False | Add timestamp to logs/results directories |
| `--run-name` | None | Custom name for this run |

## Layer Selection Priority

The pipeline determines which layers to edit using this priority:

```
1. --edit-layers [user specified]     <- Highest priority
2. ASTRA results (default enabled)    <- Auto-selected from causal tracing
3. Default [9, 10, 11]                <- Fallback
```

### Examples

```bash
# === Default Dataset (PathMNIST) ===
# AlphaEdit with ASTRA results, edit top 3 layers (default)
uv run python src/main.py --stage full

# === Multi-Dataset Examples ===
# DermaMNIST (Imbalanced skin lesion classification)
uv run python src/main.py --stage full --dataset dermamnist --timestamp

# OrganAMNIST (Grayscale CT organ classification)
uv run python src/main.py --stage full --dataset organamnist --timestamp

# RetinaMNIST (Fine-grained diabetic retinopathy grading)
uv run python src/main.py --stage full --dataset retinamnist --timestamp

# === Editing Method Options ===
# AlphaEdit with ASTRA results, edit top 5 layers
uv run python src/main.py --stage full --dataset dermamnist --num-edit-layers 5

# AlphaEdit with manually specified layers
uv run python src/main.py --stage full --dataset dermamnist --edit-layers 4 5 6

# AlphaEdit with custom projection matrix settings
uv run python src/main.py --stage edit --dataset pathmnist --projection-samples 1000 --nullspace-threshold 1e-3

# Head Editing (simpler, faster)
uv run python src/main.py --stage full --dataset organamnist --edit-method head

# Head Editing with custom parameters
uv run python src/main.py --stage full --dataset organamnist --edit-method head --head-lr 0.005

# Head Editing with closed-form solution (fastest)
uv run python src/main.py --stage full --dataset organamnist --edit-method head --closed-form

# === Baseline Comparisons ===
# Baseline 1: Retrain from scratch with error samples added to training
uv run python src/main.py --stage baseline1 --dataset pathmnist --baseline-epochs 10 --max-edits 30

# Baseline 2: Finetune on error samples only
uv run python src/main.py --stage baseline2 --dataset pathmnist --baseline-epochs 5 --baseline-lr 1e-5 --max-edits 30

# Compare all methods on same dataset
uv run python src/main.py --stage full --dataset pathmnist --run-name alphaedit_exp
uv run python src/main.py --stage baseline1 --dataset pathmnist --run-name baseline1_exp
uv run python src/main.py --stage baseline2 --dataset pathmnist --run-name baseline2_exp

# === Compare Methods Across Datasets ===
uv run python src/main.py --stage full --dataset dermamnist --edit-method alphaedit --run-name derma_alpha
uv run python src/main.py --stage full --dataset dermamnist --edit-method head --run-name derma_head
```

## Technical Details

### Data Isolation Rules (4-Set Protocol)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Official MedMNIST Data                              │
├────────────────────┬───────────────┬─────────────────┬──────────────────────┤
│  FT-Train (90%)    │ Edit-Discovery│  FT-Val         │  Test Set            │
│  ~81,000 samples   │ (10%) ~9,000  │  (Official Val) │  (Official Test)     │
├────────────────────┼───────────────┼─────────────────┼──────────────────────┤
│ ✓ Fine-tuning      │ ✗ NEVER train │ ✓ Early stop    │ ✗ NEVER train        │
│ ✓ AlphaEdit stats  │ ✓ Find errors │ ✗ NEVER edit    │ ✗ NEVER edit         │
│ ✓ Fisher info      │ ✓ Edit targets│ (discard after) │ ✓ FINAL evaluation   │
└────────────────────┴───────────────┴─────────────────┴──────────────────────┘
```

### Complete Pipeline Flow (4-Set Protocol)

```
Official MedMNIST Data
         │
         ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Data Split (4-Set Protocol)       │
│  FT-Train: 90% of Official Train            │
│  Edit-Discovery: 10% of Official Train      │
│  FT-Val: Official Val (early stopping)      │
│  Test Set: Official Test (final eval)       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Fine-tuning (on FT-Train)         │
│  ViT-B/16 → ~99% accuracy                   │
│  Early stopping with FT-Val                 │
│  [Skipped if checkpoint exists]             │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Stage 3: ASTRA Layer Localization          │
│  Find errors from Edit-Discovery set        │
│  (unseen during training!)                  │
│  Causal trace → identify important layers   │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Stage 4: Weight Editing                    │
│  stats_loader: FT-Train (covariance/Fisher) │
│  target_loader: Edit-Discovery (errors)     │
│                                             │
│  AlphaEdit: Null-space on MLP layers        │
│  Head Edit: Classifier head + EWC           │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Stage 5: Comparative Evaluation            │
│  Test on Official Test Set                  │
│  Metrics: Accuracy Delta, Stability,        │
│           Fix Rate, Regression Rate         │
└─────────────────────────────────────────────┘
```

### AlphaEdit vs Head Editing Comparison

| Aspect | AlphaEdit | Head Editing |
|--------|-----------|--------------|
| **Target** | MLP output layers | Classifier head only |
| **Parameters** | ~2.36M per layer | 6,912 (768×9 + 9) |
| **Precomputation** | Null-space projection | Fisher information |
| **Regularization** | Null-space constraint | EWC |
| **Processing** | Per-sample | Batch (all samples) |
| **Speed** | Slower | Faster |
| **ASTRA Required** | Yes (for layer selection) | No |
| **Use Case** | Deep representation changes | Direct classification fixes |

### AlphaEdit Formula

$$\Delta W = R K^T P (K K^T P + \lambda I)^{-1}$$

Where:
- $R = Z_{target} - W K$ (residual to correct)
- $P = I - K_0(K_0^T K_0)^{-1} K_0^T$ (null-space projection)
- $K$ = input activations at CLS token position

The null-space projection P ensures edits don't affect predictions for correctly classified samples.

## Changelog

### v1.6.0 (2026-01-21)
- **Baseline Methods**: Added two baseline methods for comparison with AlphaEdit
  - **Baseline 1 (Retrain)**: Add error samples to FT-Train, train from scratch
  - **Baseline 2 (Finetune-Errors)**: Finetune existing model on error samples only
- **4-Level Evaluation**: Both baselines use same evaluation framework as AlphaEdit
  - Edit Samples, FT-Train Samples, Test Set, Edit-Discovery Set
- **New CLI stages**: `--stage baseline1`, `--stage baseline2`
- **New arguments**: `--baseline-epochs`, `--baseline-lr`
- **New functions**:
  - `data_handler.py`: `get_combined_ft_train_with_errors()`, `get_error_samples_dataset()`
  - `trainer.py`: `train_from_scratch()`, `finetune_on_samples()`
  - `evaluator.py`: `evaluate_baseline_4level()`, `export_baseline_summary()`

### v1.5.0 (2026-01-16)
- **Multi-Dataset Support**: Pipeline now supports multiple MedMNIST datasets
  - PathMNIST (9 classes, RGB) - Colon Pathology
  - DermaMNIST (7 classes, RGB) - Dermatoscopy (Imbalanced)
  - RetinaMNIST (5 classes, RGB) - Retinal OCT (Fine-grained)
  - OrganAMNIST (11 classes, Grayscale) - Abdominal CT
  - BloodMNIST (8 classes, RGB) - Blood Cell Microscopy
  - TissueMNIST (8 classes, Grayscale) - Kidney Cortex Microscopy
- **Grayscale Handling**: Automatic conversion to 3-channel RGB for ViT compatibility
- **Dynamic num_classes**: Model head size adapts to dataset
- **Dataset-specific checkpoints**: `vit_{dataset}_finetuned.pt`, `vit_{dataset}_edited.pt`
- **New argument**: `--dataset` to select MedMNIST dataset

### v1.4.0 (2026-01-16)
- **4-Set Protocol**: Implemented strict data isolation strategy
  - FT-Train (90% official train): Fine-tuning + AlphaEdit covariance statistics
  - Edit-Discovery (10% official train): Find unseen errors for editing targets
  - FT-Val (official val): Early stopping during fine-tuning only
  - Test Set (official test): Final comparative evaluation
- **Dual Dataloader Support**: Editor classes now accept `stats_loader` and `target_loader`
- **Comparative Evaluation**: New `evaluate_comparative()` function with metrics:
  - Accuracy Delta, Stability, Fix Rate, Regression Rate
- **Split Reproducibility**: Split indices saved to `logs/split_indices.pt`
- **New argument**: `--ft-train-ratio` (replaces `--held-out-ratio`)

### v1.3.0 (2026-01-14)
- **Head Editing**: New alternative editing method that modifies only the classifier head
  - Faster than AlphaEdit (batch processing, fewer parameters)
  - EWC regularization to prevent catastrophic forgetting
  - Optional closed-form solution with `--closed-form`
- **New arguments**: `--edit-method`, `--head-lr`, `--head-steps`, `--ewc-lambda`, `--closed-form`
- **HeadEditor class**: New class in editor.py with gradient-based and closed-form methods

### v1.2.0 (2026-01-14)
- **ASTRA-guided layer selection**: Edit stage now uses causal tracing results
- **New arguments**: `--num-edit-layers`, `--no-astra-layers` for flexible layer control
- **Timestamp support**: `--timestamp` and `--run-name` for preserving multiple runs
- **Skip training**: Automatically loads checkpoint if exists
- **Bug fixes**: Fixed device mismatch in cache_KKT, parameter name mismatches

### v1.1.0 (2026-01-14)
- **Refactored Locator**: Changed from ASTRA ablation to AlphaEdit-style causal tracing
- Added `trace_with_patch()`, `trace_important_states()` functions
- Added `CausalTracer` class for encapsulated analysis
- Aligned with AlphaEdit methodology

### v1.0.1 (2026-01-13)
- Migrated to **uv** package manager
- Added `pyproject.toml` configuration
- Updated commands to use `uv run`

### v1.0.0 (2026-01-13)
- Initial implementation
- Modular pipeline: DataHandler, Trainer, Locator, Editor, Evaluator
- AlphaEdit null-space projection for ViT
- ASTRA patch-level ablation for layer importance
- Auto GPU/CPU detection
- CLI interface with `--stage` argument

## References

- **AlphaEdit**: Null-space projection for knowledge editing without catastrophic forgetting
- **ASTRA**: Activation steering for targeted representation adjustment
- **ViT**: "An Image is Worth 16x16 Words" (Dosovitskiy et al.)
- **MedMNIST**: Standardized medical image classification benchmark (PathMNIST)

## Important Notes

1. **GPU Support**: Automatically uses CUDA if available (configured via pyproject.toml)
2. **Held-Out Isolation**: The held-out set is strictly isolated - never used during training or editing
3. **Checkpoint Caching**: Training and projection matrices are cached for faster reruns
4. **Reproducibility**: Use `--seed` argument for reproducible results
5. **Timestamps**: Use `--timestamp` to preserve results from each run
