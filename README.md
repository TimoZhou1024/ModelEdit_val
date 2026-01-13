# ViT Model Editing Pipeline

Transfer LLM editing techniques (AlphaEdit + ASTRA) to Vision Transformers for correcting misclassified samples on MedMNIST (PathMNIST).

## 🎯 Project Goal

1. **Data Splitting**: Rigorously isolate a "Held-Out Validation Set" before any training
2. **Fine-tuning**: Train `vit-base-patch16-224` on PathMNIST
3. **Locate Layers**: Adapt ASTRA to identify significant layers for error samples
4. **Edit Weights**: Adapt AlphaEdit to correct these errors
5. **Evaluate**: Generate Confusion Matrix and Accuracy using the Held-Out set

## 📁 Project Structure

```
d:\ModelEdit\
├── src/
│   ├── data_handler.py      # PathMNIST loading & strict train/held-out split
│   ├── trainer.py           # ViT fine-tuning with auto GPU/CPU + checkpointing
│   ├── locator.py           # ASTRA-based layer importance scoring
│   ├── editor.py            # AlphaEdit null-space projection editing
│   ├── evaluator.py         # Evaluation with confusion matrix & reports
│   └── main.py              # CLI entry point
├── checkpoints/
│   ├── vit_pathmnist_finetuned.pt   # Fine-tuned model
│   └── vit_pathmnist_edited.pt      # Edited model
├── logs/
│   ├── data_split_info.csv          # Dataset split statistics
│   ├── training_metrics.csv         # Training loss/accuracy per epoch
│   ├── layer_importance.csv         # Per-sample layer importance scores
│   ├── layer_statistics.csv         # Aggregated layer statistics
│   └── edit_log.csv                 # Weight edit records
├── results/
│   ├── confusion_matrix.csv         # Confusion matrix
│   ├── evaluation_report.csv        # Detailed metrics
│   ├── predictions.csv              # All predictions with probabilities
│   └── confusion_matrix.png         # Visualization
├── reference/                        # Reference implementations
│   ├── AlphaEdit/                   # Null-space projection method
│   └── ASTRA/                       # Activation steering method
├── pyproject.toml                   # uv package configuration
└── README.md
```

## 🚀 Quick Start

### Prerequisites (使用 uv 包管理器)

```bash
# 安装 uv (如果尚未安装)
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 环境设置

```bash
cd d:\ModelEdit

# 创建虚拟环境并安装依赖 (uv 会自动读取 pyproject.toml)
uv venv
uv sync

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### Data Location

PathMNIST should be at: `~/.medmnist/pathmnist_224.npz`

### Running the Pipeline

```bash
cd d:\ModelEdit

# 使用 uv run 直接运行 (无需手动激活环境)
# Stage 1: Prepare data with strict isolation
uv run python src/main.py --stage data

# Stage 2: Fine-tune ViT (auto GPU/CPU detection)
uv run python src/main.py --stage train --epochs 10 --batch-size 32

# Stage 3: Locate important layers (ASTRA)
uv run python src/main.py --stage locate --num-ablations 32

# Stage 4: Apply weight edits (AlphaEdit)
uv run python src/main.py --stage edit --edit-layers 9 10 11 --max-edits 30

# Stage 5: Evaluate on held-out set
uv run python src/main.py --stage eval

# Or run complete pipeline
uv run python src/main.py --stage full --epochs 10
```

### 手动激活环境后运行

```bash
# 激活环境后，可以直接使用 python
.venv\Scripts\activate  # Windows
python src/main.py --stage full --epochs 10
```

## 📊 Pipeline Stages

### Stage 1: Data Preparation (`--stage data`)

- Loads PathMNIST (224×224, 9 classes) from local `.npz` file
- Creates **strict train/held-out split** (default 80:20)
- **CRITICAL**: Held-out set is isolated from ALL training and editing
- Exports: `logs/data_split_info.csv`

### Stage 2: Fine-tuning (`--stage train`)

- Uses `google/vit-base-patch16-224` with 9-class head
- **Auto-detects GPU/CPU** for optimal performance
- Saves checkpoints with model weights, optimizer state, and training history
- Exports: `checkpoints/vit_pathmnist_finetuned.pt`, `logs/training_metrics.csv`

### Stage 3: Layer Localization (`--stage locate`)

Adapts **ASTRA methodology** for ViT:
- Patch-level ablation study (zero out 14×14 patches)
- Lasso regression to estimate layer importance
- Analyzes CLS token activations across all 12 encoder layers
- Exports: `logs/layer_importance.csv`, `logs/layer_statistics.csv`

### Stage 4: Weight Editing (`--stage edit`)

Adapts **AlphaEdit** for ViT:
- Collects K vectors (input to `output.dense` layer)
- Computes null-space projection matrix P via SVD
- Optimizes target Z vectors through gradient descent
- Applies update: $\Delta = R K^T P (KK^T P + \lambda I)^{-1}$
- Exports: `logs/edit_log.csv`, `checkpoints/vit_pathmnist_edited.pt`

### Stage 5: Evaluation (`--stage eval`)

- Runs inference on **HELD-OUT set only** (ensures valid evaluation)
- Computes per-class precision, recall, F1
- Generates confusion matrix and visualization
- Exports: `results/confusion_matrix.csv`, `results/evaluation_report.csv`

## 🔧 Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage` | required | Pipeline stage: data, train, locate, edit, eval, full |
| `--data-path` | `~/.medmnist/pathmnist_224.npz` | PathMNIST data file |
| `--held-out-ratio` | 0.2 | Fraction for held-out validation |
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--edit-layers` | 9 10 11 | Layers to edit |
| `--max-edits` | 30 | Maximum samples to edit |
| `--num-ablations` | 32 | Ablations for layer analysis |
| `--seed` | 42 | Random seed |

## 📐 Technical Details

### Data Isolation Rules

```
┌─────────────────────────────────────────────────────────┐
│                    Original Data                        │
├───────────────────────────┬─────────────────────────────┤
│     Training Set (80%)    │   Held-Out Set (20%)        │
├───────────────────────────┼─────────────────────────────┤
│ ✓ Fine-tuning            │ ✗ NEVER used for training   │
│ ✓ Misclassified analysis │ ✗ NEVER used for editing    │
│ ✓ Layer localization     │ ✓ ONLY for final evaluation │
│ ✓ Weight editing         │                             │
└───────────────────────────┴─────────────────────────────┘
```

### AlphaEdit Formula (Adapted for ViT)

$$\Delta_{\text{AlphaEdit}} = R K^T P (K K^T P + K_p K_p^T P + \lambda I)^{-1}$$

Where:
- $R = V_{\text{target}} - W K$ (residual)
- $P = \hat{U} \hat{U}^T$ (null-space projection)
- $K$ = input activations at CLS token position

### ASTRA Importance Scoring

$$\text{Importance}_l = \sum_{i} |w_i^l|$$

Where $w^l$ are Lasso regression coefficients predicting output probability from layer $l$ activations under random patch ablations.

## 📝 Changelog

### v1.0.1 (2026-01-13)
- 迁移到 **uv** 包管理器
- 添加 `pyproject.toml` 配置文件
- 更新运行命令使用 `uv run`
- 删除 `requirements.txt`，统一使用 `pyproject.toml`

### v1.0.0 (2026-01-13)
- Initial implementation
- Created modular pipeline: DataHandler, Trainer, Locator, Editor, Evaluator
- Adapted AlphaEdit null-space projection for ViT MLP layers
- Adapted ASTRA patch-level ablation for layer importance
- Auto GPU/CPU detection for training
- Checkpoint saving with full state (model, optimizer, scheduler, history)
- CLI interface with `--stage` argument
- CSV logging for all stages
- Confusion matrix visualization

## 🔗 References

- **AlphaEdit**: Null-space projection for knowledge editing without catastrophic forgetting
- **ASTRA**: Adaptive activation steering for VLM safety
- **ViT**: "An Image is Worth 16x16 Words" (Dosovitskiy et al.)
- **MedMNIST**: Standardized medical image classification benchmark

## ⚠️ Important Notes

1. **Device Selection**: The pipeline automatically detects and uses GPU if available
2. **Held-Out Isolation**: The held-out set is strictly isolated - never used during training or editing
3. **Checkpoint Format**: Checkpoints include model weights, optimizer state, scheduler state, and full training history
4. **Reproducibility**: Use `--seed` argument for reproducible results
