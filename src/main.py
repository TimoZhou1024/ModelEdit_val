"""
Main Entry Point for ViT Model Editing Pipeline
================================================
CLI interface for running all pipeline stages:
- data: Load and split MedMNIST dataset
- train: Fine-tune ViT on training set
- locate: Find important layers for error samples (ASTRA)
- edit: Apply weight edits (AlphaEdit)
- eval: Evaluate on held-out set
- full: Run complete pipeline

Supported Datasets:
- pathmnist: Colon Pathology (9 classes, RGB)
- dermamnist: Dermatoscopy (7 classes, RGB) - Imbalanced
- retinamnist: Retinal OCT (5 classes, RGB) - Fine-grained
- organamnist: Abdominal CT (11 classes, Grayscale)
- bloodmnist: Blood Cell Microscopy (8 classes, RGB)
- tissuemnist: Kidney Cortex Microscopy (8 classes, Grayscale)
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from tqdm import tqdm

from data_handler import DataHandler, MedMNISTDataset, MEDMNIST_INFO
from trainer import Trainer
from locator import Locator
from editor import Editor, AlphaEditHyperParams, HeadEditor, HeadEditHyperParams
from evaluator import Evaluator, evaluate_before_after, evaluate_comparative


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ViT Model Editing Pipeline (AlphaEdit + ASTRA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage data              # Prepare dataset
  python main.py --stage train --epochs 10 # Fine-tune model
  python main.py --stage locate            # Find important layers
  python main.py --stage edit              # Apply weight edits
  python main.py --stage eval              # Evaluate on held-out set
  python main.py --stage full              # Run complete pipeline
        """
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["data", "train", "locate", "edit", "eval", "full"],
        help="Pipeline stage to run"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="pathmnist",
        choices=list(MEDMNIST_INFO.keys()),
        help="MedMNIST dataset to use (default: pathmnist)"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to pathmnist_224.npz (default: ~/.medmnist/pathmnist_224.npz)"
    )
    parser.add_argument(
        "--ft-train-ratio",
        type=float,
        default=0.9,
        help="Fraction of official train for FT-Train (default: 0.9, remaining goes to Edit-Discovery)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    
    # Locator arguments
    parser.add_argument(
        "--num-ablations",
        type=int,
        default=32,
        help="Number of ablations for layer importance (default: 32)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max samples for layer analysis (default: 50)"
    )
    
    # Editor arguments
    parser.add_argument(
        "--edit-method",
        type=str,
        choices=["alphaedit", "head"],
        default="alphaedit",
        help="Editing method: 'alphaedit' (MLP layers) or 'head' (classifier only)"
    )
    parser.add_argument(
        "--edit-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to edit (default: use ASTRA results or [9,10,11])"
    )
    parser.add_argument(
        "--num-edit-layers",
        type=int,
        default=3,
        help="Number of layers to edit when using ASTRA results (default: 3)"
    )
    parser.add_argument(
        "--use-astra-layers",
        action="store_true",
        default=True,
        help="Use ASTRA causal tracing results to select edit layers (default: True)"
    )
    parser.add_argument(
        "--no-astra-layers",
        action="store_true",
        help="Disable ASTRA layer selection, use --edit-layers or default [9,10,11]"
    )
    parser.add_argument(
        "--max-edits",
        type=int,
        default=30,
        help="Max number of samples to edit (default: 30)"
    )

    # Head editing specific arguments
    parser.add_argument(
        "--head-lr",
        type=float,
        default=0.01,
        help="Learning rate for head editing (default: 0.01)"
    )
    parser.add_argument(
        "--head-steps",
        type=int,
        default=50,
        help="Number of optimization steps for head editing (default: 50)"
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=1000.0,
        help="EWC regularization strength for head editing (default: 1000.0)"
    )
    parser.add_argument(
        "--closed-form",
        action="store_true",
        help="Use closed-form solution for head editing (faster, less precise)"
    )

    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=None,
        help="Enable pin_memory for DataLoaders (faster GPU transfer)"
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pin_memory for DataLoaders"
    )
    
    # Output arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory (default: logs)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory (default: results)"
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Add timestamp to log and results directories (preserves each run)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (used instead of timestamp)"
    )

    return parser.parse_args()


def run_data_stage(args):
    """Stage 1: Data preparation with 4-Set Protocol."""
    print("\n" + "=" * 70)
    print(f"STAGE 1: DATA PREPARATION (4-Set Protocol) - {args.dataset.upper()}")
    print("=" * 70)

    handler = DataHandler(
        dataset_name=args.dataset,
        data_path=args.data_path,
        ft_train_ratio=args.ft_train_ratio,
        random_seed=args.seed,
        log_dir=args.log_dir
    )

    # Load and create re-split
    handler.load_data()
    handler.create_resplit()

    # Export info
    handler.export_split_info()

    print("\n[OK] Data preparation complete!")
    print(f"  Dataset: {args.dataset}")
    print(f"  Classes: {handler.n_classes}")
    print(f"  FT-Train samples: {len(handler.ft_train_indices)}")
    print(f"  Edit-Discovery samples: {len(handler.discovery_indices)}")

    return handler


def run_train_stage(args, data_handler=None):
    """Stage 2: Fine-tune ViT on FT-Train set."""
    print("\n" + "=" * 70)
    print(f"STAGE 2: FINE-TUNING (on FT-Train) - {args.dataset.upper()}")
    print("=" * 70)

    # Get data handler
    if data_handler is None:
        data_handler = DataHandler(
            dataset_name=args.dataset,
            data_path=args.data_path,
            ft_train_ratio=args.ft_train_ratio,
            random_seed=args.seed,
            log_dir=args.log_dir
        )
        data_handler.load_data()
        data_handler.create_resplit()

    # Initialize trainer with dynamic num_classes
    trainer = Trainer(
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_classes=data_handler.n_classes,
        dataset_name=args.dataset,
        n_channels=data_handler.n_channels
    )

    # Setup model
    trainer.setup_model()

    # Resume if requested
    if args.resume:
        try:
            trainer.load_checkpoint()
            print("Resumed from checkpoint")
        except FileNotFoundError:
            print("No checkpoint found, starting fresh")

    # Get transforms and dataloaders
    transform = trainer.get_transforms()
    dataloaders = data_handler.get_dataloaders(
        batch_size=args.batch_size,
        transform=transform,
        pin_memory=args.pin_memory
    )

    # Train (on FT-Train set only!)
    results = trainer.train(
        train_loader=dataloaders['ft_train'],
        val_loader=dataloaders['val'],
        epochs=args.epochs,
        learning_rate=args.lr
    )

    print("\n[OK] Fine-tuning complete!")
    print(f"  Best accuracy: {results['best_acc']:.2f}%")
    print(f"  Checkpoint: {args.checkpoint_dir}/{trainer.checkpoint_name}")

    return trainer


def run_locate_stage(args, trainer=None, data_handler=None):
    """Stage 3: Find important layers for error samples (ASTRA)."""
    print("\n" + "=" * 70)
    print(f"STAGE 3: LAYER LOCALIZATION (ASTRA) - {args.dataset.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data handler
    if data_handler is None:
        data_handler = DataHandler(
            dataset_name=args.dataset,
            data_path=args.data_path,
            ft_train_ratio=args.ft_train_ratio,
            random_seed=args.seed,
            log_dir=args.log_dir
        )
        data_handler.load_data()
        data_handler.create_resplit()

    # Get trainer/model
    if trainer is None:
        trainer = Trainer(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            num_classes=data_handler.n_classes,
            dataset_name=args.dataset,
            n_channels=data_handler.n_channels
        )
        trainer.setup_model()
        trainer.load_checkpoint()
    
    transform = trainer.get_transforms()
    dataloaders = data_handler.get_dataloaders(
        batch_size=args.batch_size,
        transform=transform,
        pin_memory=args.pin_memory
    )
    
    # Find misclassified samples from Edit-Discovery set (unseen errors)
    print("\nFinding misclassified samples from Edit-Discovery set...")
    misclassified = trainer.find_misclassified(
        dataloaders['discovery'],
        max_samples=args.max_samples
    )
    
    if len(misclassified['indices']) == 0:
        print("No misclassified samples found!")
        return None
    
    # Initialize locator
    locator = Locator(
        model=trainer.model,
        device=device,
        log_dir=args.log_dir,
        num_layers=12  # Analyze all layers
    )
    
    # Analyze samples from Edit-Discovery set
    discovery_dataset = data_handler.get_discovery_dataset(transform)

    for i, idx in enumerate(tqdm(misclassified['indices'][:args.max_samples], desc="Analyzing")):
        image, label = discovery_dataset[idx]
        pred = misclassified['predictions'][i]
        
        locator.analyze_sample(
            image=image,
            true_label=label,
            predicted_label=pred,
            sample_idx=idx,
            samples=args.num_ablations
        )
    
    # Export and visualize
    locator.export_results()
    locator.visualize_importance()
    
    top_layers = locator.get_top_layers(n=3)
    print(f"\n✓ Layer localization complete!")
    print(f"  Top layers for editing: {top_layers}")
    
    return locator, misclassified


def run_edit_stage(args, trainer=None, data_handler=None, misclassified=None, astra_layers=None):
    """Stage 4: Apply weight edits (AlphaEdit or Head Editing)."""
    method_name = "Head Editing" if args.edit_method == "head" else "AlphaEdit"
    print("\n" + "=" * 70)
    print(f"STAGE 4: WEIGHT EDITING ({method_name}) - {args.dataset.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data handler
    if data_handler is None:
        data_handler = DataHandler(
            dataset_name=args.dataset,
            data_path=args.data_path,
            ft_train_ratio=args.ft_train_ratio,
            random_seed=args.seed,
            log_dir=args.log_dir
        )
        data_handler.load_data()
        data_handler.create_resplit()

    # Get trainer/model
    if trainer is None:
        trainer = Trainer(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            num_classes=data_handler.n_classes,
            dataset_name=args.dataset,
            n_channels=data_handler.n_channels
        )
        trainer.setup_model()
        trainer.load_checkpoint()

    transform = trainer.get_transforms()
    dataloaders = data_handler.get_dataloaders(
        batch_size=args.batch_size,
        transform=transform,
        pin_memory=args.pin_memory
    )

    # Find misclassified from Edit-Discovery set if not provided
    if misclassified is None:
        print("\nFinding misclassified samples from Edit-Discovery set...")
        misclassified = trainer.find_misclassified(
            dataloaders['discovery'],
            max_samples=args.max_edits
        )

    if len(misclassified['indices']) == 0:
        print("No misclassified samples to edit!")
        return trainer

    # Select editing method
    if args.edit_method == "head":
        # Head Editing: modify only classifier
        print(f"\nUsing Head Editing method")
        print(f"  Learning rate: {args.head_lr}")
        print(f"  Optimization steps: {args.head_steps}")
        print(f"  EWC lambda: {args.ewc_lambda}")
        print(f"  Closed-form: {args.closed_form}")

        hparams = HeadEditHyperParams(
            num_steps=args.head_steps,
            lr=args.head_lr,
            ewc_lambda=args.ewc_lambda,
            closed_form=args.closed_form
        )

        editor = HeadEditor(
            model=trainer.model,
            device=device,
            hparams=hparams,
            log_dir=args.log_dir,
            dataset_name=args.dataset
        )

        # Compute Fisher information for EWC (skip if using closed-form)
        # IMPORTANT: Use FT-Train (stats_loader) for Fisher computation!
        if not args.closed_form:
            print("\nComputing Fisher information for EWC (using FT-Train)...")
            editor.compute_fisher_information(
                stats_loader=dataloaders['ft_train'],
                num_samples=500
            )

        # Apply batch edit (head editing processes all samples together)
        # IMPORTANT: Use Edit-Discovery set for finding targets!
        discovery_dataset = data_handler.get_discovery_dataset(transform)
        edit_indices = misclassified['indices'][:args.max_edits]

        images_list = []
        labels_list = []
        for idx in edit_indices:
            image, label = discovery_dataset[idx]
            images_list.append(image)
            labels_list.append(label)

        images = torch.stack(images_list)
        labels = torch.tensor(labels_list)

        print(f"\nApplying head edit to {len(edit_indices)} samples...")
        editor.apply_edit(
            images=images,
            true_labels=labels,
            sample_indices=[int(i) for i in edit_indices]
        )

        # Save
        editor.export_edit_log()
        editor.save_edited_model()

        print(f"\n✓ Head editing complete!")
        print(f"  Edited samples: {len(edit_indices)}")

    else:
        # AlphaEdit: modify MLP layers
        # Determine which layers to edit
        if args.edit_layers is not None:
            edit_layers = args.edit_layers
            print(f"Using user-specified layers: {edit_layers}")
        elif not args.no_astra_layers and astra_layers is not None:
            edit_layers = astra_layers[:args.num_edit_layers]
            print(f"Using ASTRA-selected layers (top {args.num_edit_layers}): {edit_layers}")
        else:
            edit_layers = [9, 10, 11]
            print(f"Using default layers: {edit_layers}")

        hparams = AlphaEditHyperParams(
            layers=edit_layers,
            v_num_grad_steps=25,
            v_lr=0.1,
            L2=1e-4
        )

        editor = Editor(
            model=trainer.model,
            device=device,
            hparams=hparams,
            log_dir=args.log_dir,
            dataset_name=args.dataset
        )

        # Precompute projection matrices
        # IMPORTANT: Use FT-Train (stats_loader) for covariance computation!
        print("\nPrecomputing null-space projections (using FT-Train for stats)...")
        editor.precompute_projection(
            stats_loader=dataloaders['ft_train'],
            num_samples=500
        )

        # Apply edits one by one
        # IMPORTANT: Use Edit-Discovery set for finding targets!
        discovery_dataset = data_handler.get_discovery_dataset(transform)
        edit_indices = misclassified['indices'][:args.max_edits]

        for idx in tqdm(edit_indices, desc="Applying edits"):
            image, label = discovery_dataset[idx]
            image = image.unsqueeze(0)
            label_tensor = torch.tensor([label])

            editor.apply_edit(
                images=image,
                true_labels=label_tensor,
                sample_indices=[int(idx)]
            )

        # Save edited model
        editor.export_edit_log()
        editor.save_edited_model()

        print(f"\n✓ AlphaEdit complete!")
        print(f"  Edited samples: {len(edit_indices)}")
        print(f"  Edited layers: {edit_layers}")

    return editor


def run_eval_stage(args, trainer=None, data_handler=None, edited_model=None):
    """Stage 5: Evaluate on Official Test Set (Comparative Evaluation)."""
    print("\n" + "=" * 70)
    print(f"STAGE 5: EVALUATION (Official Test Set) - {args.dataset.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data handler
    if data_handler is None:
        data_handler = DataHandler(
            dataset_name=args.dataset,
            data_path=args.data_path,
            ft_train_ratio=args.ft_train_ratio,
            random_seed=args.seed,
            log_dir=args.log_dir
        )
        data_handler.load_data()
        data_handler.create_resplit()

    # Get transforms
    if trainer is None:
        trainer = Trainer(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            num_classes=data_handler.n_classes,
            dataset_name=args.dataset,
            n_channels=data_handler.n_channels
        )
    
    if trainer.model is None:
        trainer.setup_model()
        trainer.load_checkpoint()
    
    transform = trainer.get_transforms()
    dataloaders = data_handler.get_dataloaders(
        batch_size=args.batch_size,
        transform=transform,
        pin_memory=args.pin_memory
    )
    
    # Evaluate on OFFICIAL TEST SET (4-Set Protocol)
    print("\n>>> EVALUATING ON OFFICIAL TEST SET <<<")
    print(">>> This set was NEVER used for training, validation, or editing <<<\n")

    edited_path = Path(args.checkpoint_dir) / f"vit_{args.dataset}_edited.pt"
    finetuned_path = Path(args.checkpoint_dir) / f"vit_{args.dataset}_finetuned.pt"

    # Case A: have both edited and finetuned checkpoints -> comparative evaluation
    if edited_path.exists() and finetuned_path.exists():
        print("Running COMPARATIVE EVALUATION on Official Test Set...")

        # Load baseline (before edit)
        baseline_trainer = Trainer(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            num_classes=data_handler.n_classes,
            dataset_name=args.dataset,
            n_channels=data_handler.n_channels
        )
        baseline_trainer.setup_model()
        baseline_trainer.load_checkpoint(filepath=finetuned_path, load_optimizer=False)
        model_before = baseline_trainer.model

        # Load edited model
        edited_trainer = Trainer(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            num_classes=data_handler.n_classes,
            dataset_name=args.dataset,
            n_channels=data_handler.n_channels
        )
        edited_trainer.setup_model()
        checkpoint = torch.load(edited_path, map_location=device)
        edited_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        model_after = edited_trainer.model

        # Run comparative evaluation on Official Test Set
        evaluate_comparative(
            model_orig=model_before,
            model_edit=model_after,
            test_loader=dataloaders['test'],
            device=device,
            results_dir=args.results_dir
        )

        return None

    # Case B: only edited model provided
    if edited_model is not None:
        model = edited_model
        print("Evaluating EDITED model (in-memory)")
    elif edited_path.exists():
        print(f"Loading edited model from {edited_path}")
        checkpoint = torch.load(edited_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        model = trainer.model
    else:
        model = trainer.model
        print("Evaluating FINE-TUNED model (no edits found)")

    evaluator = Evaluator(
        model=model,
        device=device,
        results_dir=args.results_dir,
        log_dir=args.log_dir
    )

    evaluator.run_inference(dataloaders['test'], desc="Test Set Eval")

    # Generate reports
    evaluator.print_summary()
    evaluator.export_confusion_matrix()
    evaluator.export_evaluation_report()
    evaluator.export_predictions()

    try:
        evaluator.plot_confusion_matrix()
    except Exception as e:
        print(f"Warning: Could not plot confusion matrix: {e}")

    print(f"\n[OK] Evaluation complete!")
    print(f"  Results saved to: {args.results_dir}/")

    return evaluator


def run_full_pipeline(args):
    """Run complete pipeline from start to finish (4-Set Protocol)."""
    print("\n" + "=" * 70)
    print(f"RUNNING COMPLETE PIPELINE (4-Set Protocol) - {args.dataset.upper()}")
    print("=" * 70)

    # Stage 1: Data
    data_handler = run_data_stage(args)

    # Stage 2: Training (skip if checkpoint exists)
    checkpoint_path = Path(args.checkpoint_dir) / f"vit_{args.dataset}_finetuned.pt"
    if checkpoint_path.exists():
        print("\n" + "=" * 70)
        print("STAGE 2: FINE-TUNING (SKIPPED - checkpoint found)")
        print("=" * 70)
        print(f"Loading existing checkpoint: {checkpoint_path}")

        trainer = Trainer(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            num_classes=data_handler.n_classes,
            dataset_name=args.dataset,
            n_channels=data_handler.n_channels
        )
        trainer.setup_model()
        trainer.load_checkpoint(filepath=checkpoint_path, load_optimizer=False)
        print(f"✓ Loaded model with accuracy: {trainer.best_acc:.2f}%")
    else:
        trainer = run_train_stage(args, data_handler)
    
    # Stage 3: Localization
    locator, misclassified = run_locate_stage(args, trainer, data_handler)

    # Get ASTRA-determined layers for editing
    astra_layers = None
    if locator is not None:
        astra_layers = locator.get_top_layers(n=args.num_edit_layers)
        print(f"\nASTRA top {args.num_edit_layers} layers: {astra_layers}")

    # Stage 4: Editing (pass ASTRA layers)
    editor = run_edit_stage(args, trainer, data_handler, misclassified, astra_layers=astra_layers)
    
    # Stage 5: Evaluation
    evaluator = run_eval_stage(args, trainer, data_handler, editor.model if editor else None)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Checkpoints: {args.checkpoint_dir}/")
    print(f"  Logs: {args.log_dir}/")
    print(f"  Results: {args.results_dir}/")


def main():
    """Main entry point."""
    args = parse_args()

    # Apply timestamp or run name to output directories
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

    # Print dataset info
    print(f"\n{'=' * 70}")
    print(f"ViT Model Editing Pipeline - {args.dataset.upper()}")
    print(f"{'=' * 70}")
    dataset_info = MEDMNIST_INFO[args.dataset]
    print(f"  Dataset: {args.dataset}")
    print(f"  Classes: {dataset_info['n_classes']}")
    print(f"  Channels: {dataset_info['n_channels']} ({'Grayscale' if dataset_info['n_channels'] == 1 else 'RGB'})")
    print(f"  Description: {dataset_info['description']}")

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve pin_memory (None = auto-detect in data_handler)
    if args.no_pin_memory:
        args.pin_memory = False
    elif args.pin_memory:
        args.pin_memory = True
    else:
        args.pin_memory = None  # Auto-detect based on CUDA availability
    
    # Run selected stage
    if args.stage == "data":
        run_data_stage(args)
    elif args.stage == "train":
        run_train_stage(args)
    elif args.stage == "locate":
        run_locate_stage(args)
    elif args.stage == "edit":
        run_edit_stage(args)
    elif args.stage == "eval":
        run_eval_stage(args)
    elif args.stage == "full":
        run_full_pipeline(args)
    else:
        print(f"Unknown stage: {args.stage}")
        sys.exit(1)


if __name__ == "__main__":
    main()
