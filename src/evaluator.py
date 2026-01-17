"""
Evaluator Module for ViT Model Editing Pipeline
================================================
Evaluates model performance on the held-out validation set.
Generates:
- Confusion matrix
- Per-class accuracy
- Edit success rate
- Detailed evaluation reports
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import seaborn as sns
import matplotlib.pyplot as plt


class Evaluator:
    """
    Comprehensive evaluator for the ViT model editing pipeline.
    """
    
    # PathMNIST class names
    CLASS_NAMES = [
        "Adipose",
        "Background", 
        "Debris",
        "Lymphocytes",
        "Mucus",
        "Smooth Muscle",
        "Normal Colon Mucosa",
        "Cancer-associated Stroma",
        "Colorectal Adenocarcinoma"
    ]
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        results_dir: str = "results",
        log_dir: str = "logs"
    ):
        """
        Args:
            model: Model to evaluate
            device: Computation device
            results_dir: Directory for saving evaluation results
            log_dir: Directory for saving logs
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(device)
        self.device = device
        self.results_dir = Path(results_dir)
        self.log_dir = Path(log_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation results storage
        self.predictions = None
        self.true_labels = None
        self.probabilities = None
        self.sample_indices = None
        
    @torch.no_grad()
    def run_inference(
        self,
        dataloader: torch.utils.data.DataLoader,
        desc: str = "Evaluating"
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a dataset.
        
        Returns:
            Dictionary with predictions, true_labels, probabilities
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(self.device)
            
            outputs = self.model(images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        self.predictions = np.array(all_preds)
        self.true_labels = np.array(all_labels)
        self.probabilities = np.array(all_probs)
        self.sample_indices = np.arange(len(self.predictions))
        
        return {
            'predictions': self.predictions,
            'true_labels': self.true_labels,
            'probabilities': self.probabilities
        }
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        # Overall accuracy
        accuracy = accuracy_score(self.true_labels, self.predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            average=None,
            zero_division=0
        )
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            average='macro',
            zero_division=0
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            average='weighted',
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        # Error analysis
        errors = self.predictions != self.true_labels
        error_rate = errors.mean()
        error_indices = np.where(errors)[0]
        
        return {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'num_errors': len(error_indices),
            'total_samples': len(self.predictions),
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            },
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            },
            'weighted': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            },
            'confusion_matrix': cm,
            'error_indices': error_indices
        }
    
    def compute_edit_success_rate(
        self,
        edit_indices: List[int],
        original_predictions: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Compute success rate of model edits.
        
        Args:
            edit_indices: Indices of samples that were edited
            original_predictions: Predictions before editing (optional)
            
        Returns:
            Dictionary with edit success metrics
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        edit_indices = np.array(edit_indices)
        
        # Check which edits were successful (now correctly classified)
        edit_preds = self.predictions[edit_indices]
        edit_labels = self.true_labels[edit_indices]
        
        successful = edit_preds == edit_labels
        success_rate = successful.mean()
        
        result = {
            'total_edits': len(edit_indices),
            'successful_edits': successful.sum(),
            'failed_edits': (~successful).sum(),
            'success_rate': success_rate,
            'successful_indices': edit_indices[successful].tolist(),
            'failed_indices': edit_indices[~successful].tolist()
        }
        
        # If we have original predictions, compute flip analysis
        if original_predictions is not None:
            original_preds = original_predictions[edit_indices]
            
            # Samples that were wrong and are now correct
            fixed = (original_preds != edit_labels) & (edit_preds == edit_labels)
            
            # Samples that were correct and are now wrong (regression)
            broken = (original_preds == edit_labels) & (edit_preds != edit_labels)
            
            # Samples that changed but still wrong
            changed_still_wrong = (original_preds != edit_preds) & (edit_preds != edit_labels)
            
            result['flip_analysis'] = {
                'fixed': fixed.sum(),
                'broken': broken.sum(),
                'changed_still_wrong': changed_still_wrong.sum(),
                'unchanged': (original_preds == edit_preds).sum()
            }
        
        return result
    
    def export_confusion_matrix(
        self,
        filename: str = "confusion_matrix.csv",
        normalize: bool = False
    ) -> str:
        """
        Export confusion matrix to CSV.
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
        
        # Create DataFrame with class names
        df = pd.DataFrame(
            cm,
            index=[f"True_{i}_{name}" for i, name in enumerate(self.CLASS_NAMES)],
            columns=[f"Pred_{i}_{name}" for i, name in enumerate(self.CLASS_NAMES)]
        )
        
        csv_path = self.results_dir / filename
        df.to_csv(csv_path)
        
        print(f"Confusion matrix exported to: {csv_path}")
        return str(csv_path)
    
    def export_evaluation_report(
        self,
        filename: str = "evaluation_report.csv",
        additional_info: Dict[str, Any] = None
    ) -> str:
        """
        Export comprehensive evaluation report to CSV.
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        metrics = self.compute_metrics()
        
        rows = []
        
        # Overall metrics
        rows.append({
            'category': 'overall',
            'metric': 'accuracy',
            'value': metrics['accuracy'],
            'notes': f"{metrics['accuracy']*100:.2f}%"
        })
        rows.append({
            'category': 'overall',
            'metric': 'error_rate',
            'value': metrics['error_rate'],
            'notes': f"{metrics['num_errors']}/{metrics['total_samples']}"
        })
        
        # Macro averages
        for metric_name in ['precision', 'recall', 'f1']:
            rows.append({
                'category': 'macro_avg',
                'metric': metric_name,
                'value': metrics['macro'][metric_name],
                'notes': ''
            })
        
        # Weighted averages
        for metric_name in ['precision', 'recall', 'f1']:
            rows.append({
                'category': 'weighted_avg',
                'metric': metric_name,
                'value': metrics['weighted'][metric_name],
                'notes': ''
            })
        
        # Per-class metrics
        for class_id in range(len(self.CLASS_NAMES)):
            class_name = self.CLASS_NAMES[class_id]
            for metric_name in ['precision', 'recall', 'f1', 'support']:
                rows.append({
                    'category': f'class_{class_id}',
                    'metric': metric_name,
                    'value': metrics['per_class'][metric_name][class_id],
                    'notes': class_name
                })
        
        # Additional info
        if additional_info:
            for key, value in additional_info.items():
                rows.append({
                    'category': 'additional',
                    'metric': key,
                    'value': value if isinstance(value, (int, float)) else str(value),
                    'notes': ''
                })
        
        df = pd.DataFrame(rows)
        csv_path = self.results_dir / filename
        df.to_csv(csv_path, index=False)
        
        print(f"Evaluation report exported to: {csv_path}")
        return str(csv_path)
    
    def export_predictions(
        self,
        filename: str = "predictions.csv"
    ) -> str:
        """
        Export all predictions with probabilities.
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        rows = []
        for i in range(len(self.predictions)):
            row = {
                'sample_idx': i,
                'true_label': self.true_labels[i],
                'true_class': self.CLASS_NAMES[self.true_labels[i]],
                'predicted_label': self.predictions[i],
                'predicted_class': self.CLASS_NAMES[self.predictions[i]],
                'correct': self.predictions[i] == self.true_labels[i],
                'confidence': self.probabilities[i].max()
            }
            
            # Add per-class probabilities
            for j in range(len(self.CLASS_NAMES)):
                row[f'prob_class_{j}'] = self.probabilities[i, j]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.results_dir / filename
        df.to_csv(csv_path, index=False)
        
        print(f"Predictions exported to: {csv_path}")
        return str(csv_path)
    
    def print_summary(self):
        """
        Print evaluation summary to console.
        """
        if self.predictions is None:
            print("No predictions available. Run inference first.")
            return
        
        metrics = self.compute_metrics()
        
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Error Rate: {metrics['error_rate']*100:.2f}%")
        print(f"  Total Samples: {metrics['total_samples']}")
        print(f"  Errors: {metrics['num_errors']}")
        
        print(f"\nMacro Averages:")
        print(f"  Precision: {metrics['macro']['precision']:.4f}")
        print(f"  Recall: {metrics['macro']['recall']:.4f}")
        print(f"  F1-Score: {metrics['macro']['f1']:.4f}")
        
        print(f"\nPer-Class Performance:")
        print(f"{'Class':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 75)
        
        for i, class_name in enumerate(self.CLASS_NAMES):
            print(f"{i}: {class_name:<32} "
                  f"{metrics['per_class']['precision'][i]:>10.4f} "
                  f"{metrics['per_class']['recall'][i]:>10.4f} "
                  f"{metrics['per_class']['f1'][i]:>10.4f} "
                  f"{metrics['per_class']['support'][i]:>10.0f}")
        
        print("\nConfusion Matrix (top-5 confusions):")
        cm = metrics['confusion_matrix']
        
        # Find top confusions (off-diagonal)
        confusions = []
        for i in range(len(self.CLASS_NAMES)):
            for j in range(len(self.CLASS_NAMES)):
                if i != j and cm[i, j] > 0:
                    confusions.append((i, j, cm[i, j]))
        
        confusions.sort(key=lambda x: x[2], reverse=True)
        
        for true_idx, pred_idx, count in confusions[:5]:
            print(f"  {self.CLASS_NAMES[true_idx]} → {self.CLASS_NAMES[pred_idx]}: {count}")
        
        print("=" * 70)
    
    def plot_confusion_matrix(
        self,
        filename: str = "confusion_matrix.png",
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ) -> str:
        """
        Plot and save confusion matrix visualization.
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        
        # Shorter names for display
        short_names = [name[:15] + "..." if len(name) > 15 else name 
                       for name in self.CLASS_NAMES]
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=short_names,
            yticklabels=short_names
        )
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        png_path = self.results_dir / filename
        plt.savefig(png_path, dpi=150)
        plt.close()
        
        print(f"Confusion matrix plot saved to: {png_path}")
        return str(png_path)


def evaluate_before_after(
    model_before: nn.Module,
    model_after: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = None,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Compare model performance before and after editing.
    
    Args:
        model_before: Model before editing
        model_after: Model after editing
        dataloader: Held-out validation dataloader
        device: Computation device
        results_dir: Directory for results
        
    Returns:
        Comparison metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate BEFORE editing
    print("\n=== Evaluating BEFORE editing ===")
    eval_before = Evaluator(model_before, device, results_dir / "before")
    eval_before.run_inference(dataloader, desc="Before")
    metrics_before = eval_before.compute_metrics()
    eval_before.export_confusion_matrix("confusion_matrix_before.csv")
    eval_before.export_evaluation_report("evaluation_before.csv")
    eval_before.print_summary()

    # Evaluate AFTER editing
    print("\n=== Evaluating AFTER editing ===")
    eval_after = Evaluator(model_after, device, results_dir / "after")
    eval_after.run_inference(dataloader, desc="After")
    metrics_after = eval_after.compute_metrics()
    eval_after.export_confusion_matrix("confusion_matrix_after.csv")
    eval_after.export_evaluation_report("evaluation_after.csv")
    eval_after.print_summary()

    # Comparison with multiple success criteria
    comparison = {
        'accuracy_before': metrics_before['accuracy'],
        'accuracy_after': metrics_after['accuracy'],
        'accuracy_change': metrics_after['accuracy'] - metrics_before['accuracy'],
        'macro_f1_before': metrics_before['macro']['f1'],
        'macro_f1_after': metrics_after['macro']['f1'],
        'macro_f1_change': metrics_after['macro']['f1'] - metrics_before['macro']['f1'],
        'errors_before': metrics_before['num_errors'],
        'errors_after': metrics_after['num_errors'],
        'errors_fixed': metrics_before['num_errors'] - metrics_after['num_errors']
    }

    # Export comparison
    df = pd.DataFrame([comparison])
    df.to_csv(results_dir / "comparison.csv", index=False)

    # Terminal summary for quick judgment of edit success
    print("\n" + "=" * 70)
    print("EDIT SUCCESS CHECKLIST")
    print("=" * 70)
    print(f"Accuracy: {comparison['accuracy_before']*100:.2f}% → {comparison['accuracy_after']*100:.2f}% "
          f"({comparison['accuracy_change']*100:+.2f}%)")
    print(f"Macro-F1: {comparison['macro_f1_before']:.4f} → {comparison['macro_f1_after']:.4f} "
          f"({comparison['macro_f1_change']:+.4f})")
    print(f"Errors: {comparison['errors_before']} → {comparison['errors_after']} "
          f"({comparison['errors_fixed']:+d} fixed)")

    # Highlight pass/fail style indicators
    print("\nJudgment Indicators:")
    print("  • Accuracy improvement    : " + ("PASS" if comparison['accuracy_change'] > 0 else "CHECK"))
    print("  • Macro-F1 improvement    : " + ("PASS" if comparison['macro_f1_change'] > 0 else "CHECK"))
    print("  • Errors reduced          : " + ("PASS" if comparison['errors_fixed'] > 0 else "CHECK"))
    print("  • No regression (errors)  : " + ("PASS" if comparison['errors_after'] <= comparison['errors_before'] else "CHECK"))
    print("=" * 70)

    return comparison


def evaluate_comparative(
    model_orig: nn.Module,
    model_edit: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = None,
    results_dir: str = "results",
    set_name: str = "Test Set"
) -> Dict[str, Any]:
    """
    Comparative Evaluation on a given dataset (4-Set Protocol).

    This compares Pre-Edit vs Post-Edit models on the specified dataset.

    Args:
        model_orig: Original model (before editing)
        model_edit: Edited model (after editing)
        test_loader: DataLoader for the evaluation set
        device: Computation device
        results_dir: Directory for saving results
        set_name: Name of the evaluation set (e.g., "Test Set", "Edit-Discovery Set")

    Returns:
        Dictionary with comparative metrics:
        - accuracy_delta: Change in accuracy (positive = improvement)
        - stability: Fraction of correct samples that remained correct
        - fix_rate: Fraction of error samples that became correct
        - regression_rate: Fraction of correct samples that became wrong
        - confusion_matrices: Before and after confusion matrices
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create file-safe name for the set
    set_name_safe = set_name.lower().replace(" ", "_").replace("-", "_")

    print("\n" + "=" * 70)
    print(f"COMPARATIVE EVALUATION ON {set_name.upper()}")
    print("(4-Set Protocol: Pre-Edit vs Post-Edit Comparison)")
    print("=" * 70)

    # Collect predictions from both models
    model_orig.eval()
    model_edit.eval()

    all_labels = []
    preds_orig = []
    preds_edit = []
    probs_orig = []
    probs_edit = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating on {set_name}"):
            images = images.to(device)

            # Original model predictions
            out_orig = model_orig(images)
            logits_orig = out_orig.logits
            prob_orig = torch.softmax(logits_orig, dim=-1)

            # Edited model predictions
            out_edit = model_edit(images)
            logits_edit = out_edit.logits
            prob_edit = torch.softmax(logits_edit, dim=-1)

            all_labels.extend(labels.numpy())
            preds_orig.extend(logits_orig.argmax(dim=1).cpu().numpy())
            preds_edit.extend(logits_edit.argmax(dim=1).cpu().numpy())
            probs_orig.extend(prob_orig.cpu().numpy())
            probs_edit.extend(prob_edit.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    preds_orig = np.array(preds_orig)
    preds_edit = np.array(preds_edit)

    # Compute basic accuracy metrics
    acc_orig = accuracy_score(all_labels, preds_orig)
    acc_edit = accuracy_score(all_labels, preds_edit)
    accuracy_delta = acc_edit - acc_orig

    # Compute stability and fix rate
    correct_orig = (preds_orig == all_labels)
    correct_edit = (preds_edit == all_labels)

    # Stability: samples that were correct and remained correct
    stable_mask = correct_orig & correct_edit
    stability = stable_mask.sum() / correct_orig.sum() if correct_orig.sum() > 0 else 0.0

    # Fix Rate: samples that were wrong and became correct
    error_orig = ~correct_orig
    fixed_mask = error_orig & correct_edit
    fix_rate = fixed_mask.sum() / error_orig.sum() if error_orig.sum() > 0 else 0.0

    # Regression Rate: samples that were correct and became wrong
    regression_mask = correct_orig & ~correct_edit
    regression_rate = regression_mask.sum() / correct_orig.sum() if correct_orig.sum() > 0 else 0.0

    # Confusion matrices
    cm_orig = confusion_matrix(all_labels, preds_orig)
    cm_edit = confusion_matrix(all_labels, preds_edit)

    # Detailed transition analysis
    n_total = len(all_labels)
    n_correct_orig = correct_orig.sum()
    n_error_orig = error_orig.sum()
    n_correct_edit = correct_edit.sum()
    n_fixed = fixed_mask.sum()
    n_regressed = regression_mask.sum()
    n_stable = stable_mask.sum()

    # Build result dictionary
    result = {
        'accuracy_orig': acc_orig,
        'accuracy_edit': acc_edit,
        'accuracy_delta': accuracy_delta,
        'stability': stability,
        'fix_rate': fix_rate,
        'regression_rate': regression_rate,
        'n_total': n_total,
        'n_correct_orig': int(n_correct_orig),
        'n_error_orig': int(n_error_orig),
        'n_correct_edit': int(n_correct_edit),
        'n_fixed': int(n_fixed),
        'n_regressed': int(n_regressed),
        'n_stable': int(n_stable),
        'confusion_matrix_orig': cm_orig,
        'confusion_matrix_edit': cm_edit
    }

    # Print summary
    print(f"\n=== Comparative Evaluation Results ({set_name}) ===")
    print(f"{set_name} Size: {n_total} samples")
    print(f"\nAccuracy:")
    print(f"  Pre-Edit:  {acc_orig*100:.2f}% ({n_correct_orig}/{n_total})")
    print(f"  Post-Edit: {acc_edit*100:.2f}% ({n_correct_edit}/{n_total})")
    print(f"  Delta:     {accuracy_delta*100:+.2f}%")

    print(f"\nTransition Analysis:")
    print(f"  Stability (correct->correct): {stability*100:.1f}% ({n_stable}/{n_correct_orig})")
    print(f"  Fix Rate (error->correct):    {fix_rate*100:.1f}% ({n_fixed}/{n_error_orig})")
    print(f"  Regression (correct->error):  {regression_rate*100:.1f}% ({n_regressed}/{n_correct_orig})")

    print(f"\nJudgment Indicators:")
    print(f"  [{'PASS' if accuracy_delta > 0 else 'FAIL'}] Accuracy improved")
    print(f"  [{'PASS' if stability > 0.95 else 'WARN' if stability > 0.90 else 'FAIL'}] Stability > 95%")
    print(f"  [{'PASS' if fix_rate > 0 else 'INFO'}] Some errors fixed")
    print(f"  [{'PASS' if regression_rate < 0.05 else 'WARN' if regression_rate < 0.10 else 'FAIL'}] Regression < 5%")

    # Export results to CSV
    summary_rows = [
        {'metric': 'accuracy_orig', 'value': acc_orig, 'notes': f'{acc_orig*100:.2f}%'},
        {'metric': 'accuracy_edit', 'value': acc_edit, 'notes': f'{acc_edit*100:.2f}%'},
        {'metric': 'accuracy_delta', 'value': accuracy_delta, 'notes': f'{accuracy_delta*100:+.2f}%'},
        {'metric': 'stability', 'value': stability, 'notes': f'{n_stable}/{n_correct_orig}'},
        {'metric': 'fix_rate', 'value': fix_rate, 'notes': f'{n_fixed}/{n_error_orig}'},
        {'metric': 'regression_rate', 'value': regression_rate, 'notes': f'{n_regressed}/{n_correct_orig}'},
        {'metric': 'n_total', 'value': n_total, 'notes': f'{set_name} size'},
        {'metric': 'n_fixed', 'value': n_fixed, 'notes': 'errors corrected'},
        {'metric': 'n_regressed', 'value': n_regressed, 'notes': 'new errors introduced'},
    ]

    df_summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / f'comparative_evaluation_{set_name_safe}.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"\nResults exported to: {summary_path}")

    # Export confusion matrices
    # Use confusion matrix shape to determine number of classes dynamically
    n_classes = cm_orig.shape[0]
    df_cm_orig = pd.DataFrame(
        cm_orig,
        index=[f"True_{i}" for i in range(n_classes)],
        columns=[f"Pred_{i}" for i in range(n_classes)]
    )
    df_cm_edit = pd.DataFrame(
        cm_edit,
        index=[f"True_{i}" for i in range(n_classes)],
        columns=[f"Pred_{i}" for i in range(n_classes)]
    )

    df_cm_orig.to_csv(results_dir / f'confusion_matrix_orig_{set_name_safe}.csv')
    df_cm_edit.to_csv(results_dir / f'confusion_matrix_edit_{set_name_safe}.csv')

    print("=" * 70)

    return result


def evaluate_edit_samples(
    model: nn.Module,
    images: torch.Tensor,
    true_labels: torch.Tensor,
    sample_indices: List[int],
    device: torch.device = None,
    desc: str = "Edit Samples"
) -> Dict[str, Any]:
    """
    Evaluate model performance on specific edit samples.

    Args:
        model: Model to evaluate
        images: Edit sample images (B, C, H, W)
        true_labels: True labels for edit samples (B,)
        sample_indices: Original dataset indices of these samples
        device: Computation device
        desc: Description for logging

    Returns:
        Dictionary with:
        - predictions: Model predictions for each sample
        - probabilities: Prediction probabilities
        - correct: Boolean array of correct predictions
        - accuracy: Overall accuracy on edit samples
        - per_sample_info: Detailed info for each sample
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    images = images.to(device)
    true_labels = true_labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=1)

    predictions = predictions.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()
    probs_np = probs.cpu().numpy()

    correct = predictions == true_labels_np
    accuracy = correct.mean()

    # Per-sample detailed info
    per_sample_info = []
    for i in range(len(predictions)):
        per_sample_info.append({
            'dataset_idx': sample_indices[i],
            'true_label': int(true_labels_np[i]),
            'predicted_label': int(predictions[i]),
            'correct': bool(correct[i]),
            'confidence': float(probs_np[i].max()),
            'true_class_prob': float(probs_np[i, true_labels_np[i]])
        })

    return {
        'predictions': predictions,
        'true_labels': true_labels_np,
        'probabilities': probs_np,
        'correct': correct,
        'accuracy': accuracy,
        'num_correct': int(correct.sum()),
        'num_total': len(predictions),
        'per_sample_info': per_sample_info
    }


def compare_edit_samples_before_after(
    results_before: Dict[str, Any],
    results_after: Dict[str, Any],
    sample_indices: List[int]
) -> Dict[str, Any]:
    """
    Compare model performance on edit samples before and after editing.

    Args:
        results_before: Results from evaluate_edit_samples before editing
        results_after: Results from evaluate_edit_samples after editing
        sample_indices: Original dataset indices

    Returns:
        Dictionary with comparison metrics and per-sample changes
    """
    preds_before = results_before['predictions']
    preds_after = results_after['predictions']
    true_labels = results_before['true_labels']

    correct_before = results_before['correct']
    correct_after = results_after['correct']

    # Transition analysis
    fixed = (~correct_before) & correct_after  # Was wrong, now correct
    broken = correct_before & (~correct_after)  # Was correct, now wrong
    stayed_correct = correct_before & correct_after
    stayed_wrong = (~correct_before) & (~correct_after)

    # Per-sample transition info
    per_sample_transitions = []
    for i in range(len(preds_before)):
        if fixed[i]:
            status = "FIXED"
        elif broken[i]:
            status = "BROKEN"
        elif stayed_correct[i]:
            status = "STAYED_CORRECT"
        else:
            status = "STAYED_WRONG"

        per_sample_transitions.append({
            'dataset_idx': sample_indices[i],
            'true_label': int(true_labels[i]),
            'pred_before': int(preds_before[i]),
            'pred_after': int(preds_after[i]),
            'correct_before': bool(correct_before[i]),
            'correct_after': bool(correct_after[i]),
            'status': status
        })

    comparison = {
        'accuracy_before': results_before['accuracy'],
        'accuracy_after': results_after['accuracy'],
        'accuracy_delta': results_after['accuracy'] - results_before['accuracy'],
        'num_fixed': int(fixed.sum()),
        'num_broken': int(broken.sum()),
        'num_stayed_correct': int(stayed_correct.sum()),
        'num_stayed_wrong': int(stayed_wrong.sum()),
        'num_total': len(preds_before),
        'fix_rate': float(fixed.sum()) / max(1, (~correct_before).sum()),
        'break_rate': float(broken.sum()) / max(1, correct_before.sum()),
        'per_sample_transitions': per_sample_transitions
    }

    return comparison


def print_edit_samples_comparison(
    comparison: Dict[str, Any],
    results_before: Dict[str, Any],
    results_after: Dict[str, Any]
):
    """
    Print a formatted comparison of edit samples before and after editing.
    """
    print("\n" + "=" * 70)
    print("EDIT SAMPLES PERFORMANCE COMPARISON")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Total Edit Samples: {comparison['num_total']}")
    print(f"  Accuracy Before: {comparison['accuracy_before']*100:.1f}% "
          f"({results_before['num_correct']}/{results_before['num_total']})")
    print(f"  Accuracy After:  {comparison['accuracy_after']*100:.1f}% "
          f"({results_after['num_correct']}/{results_after['num_total']})")
    print(f"  Accuracy Change: {comparison['accuracy_delta']*100:+.1f}%")

    print(f"\nTransition Analysis:")
    print(f"  FIXED (wrong->correct):    {comparison['num_fixed']} samples")
    print(f"  BROKEN (correct->wrong):   {comparison['num_broken']} samples")
    print(f"  STAYED_CORRECT:            {comparison['num_stayed_correct']} samples")
    print(f"  STAYED_WRONG:              {comparison['num_stayed_wrong']} samples")

    # Show per-sample details (first 20)
    print(f"\nPer-Sample Details (showing up to 20):")
    print(f"{'Idx':>6} {'True':>6} {'Before':>8} {'After':>8} {'Status':>15}")
    print("-" * 50)

    for info in comparison['per_sample_transitions'][:20]:
        status_color = {
            'FIXED': '✓',
            'BROKEN': '✗',
            'STAYED_CORRECT': '=',
            'STAYED_WRONG': '-'
        }.get(info['status'], '?')

        print(f"{info['dataset_idx']:>6} {info['true_label']:>6} "
              f"{info['pred_before']:>8} {info['pred_after']:>8} "
              f"{status_color} {info['status']:>13}")

    if len(comparison['per_sample_transitions']) > 20:
        print(f"  ... and {len(comparison['per_sample_transitions']) - 20} more samples")

    # Summary judgment
    print(f"\nEdit Success Judgment:")
    if comparison['num_fixed'] > 0 and comparison['num_broken'] == 0:
        print(f"  [EXCELLENT] Fixed {comparison['num_fixed']} errors with no regressions!")
    elif comparison['num_fixed'] > comparison['num_broken']:
        print(f"  [GOOD] Net improvement: +{comparison['num_fixed'] - comparison['num_broken']} correct predictions")
    elif comparison['num_fixed'] == comparison['num_broken']:
        print(f"  [NEUTRAL] Equal fixes and regressions")
    else:
        print(f"  [POOR] More regressions than fixes: {comparison['num_broken']} broken vs {comparison['num_fixed']} fixed")

    print("=" * 70)


def evaluate_projection_samples(
    model: nn.Module,
    projection_samples: Dict[str, Any],
    device: torch.device = None,
    desc: str = "Projection Samples"
) -> Dict[str, Any]:
    """
    Evaluate model performance on the FT-Train samples used for projection matrix.

    Args:
        model: Model to evaluate
        projection_samples: Dictionary from Editor.get_projection_samples()
                           containing 'images', 'labels', 'num_samples'
        device: Computation device
        desc: Description for logging

    Returns:
        Dictionary with:
        - predictions: Model predictions for each sample
        - probabilities: Prediction probabilities
        - correct: Boolean array of correct predictions
        - accuracy: Overall accuracy on projection samples
        - per_sample_info: Detailed info for each sample
        - class_distribution: Count of samples per class
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if projection_samples is None:
        print(f"Warning: No projection samples available")
        return None

    model.eval()
    model.to(device)

    images = projection_samples['images'].to(device)
    labels = projection_samples['labels']
    if isinstance(labels, torch.Tensor):
        labels = labels.to(device)
    else:
        labels = torch.tensor(labels).to(device)

    num_samples = projection_samples['num_samples']

    # Process in batches to avoid memory issues
    batch_size = 32
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i+batch_size]
            outputs = model(batch_images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=1)

            all_predictions.append(predictions.cpu())
            all_probs.append(probs.cpu())

    predictions = torch.cat(all_predictions, dim=0).numpy()
    probs_np = torch.cat(all_probs, dim=0).numpy()
    true_labels_np = labels.cpu().numpy()

    correct = predictions == true_labels_np
    accuracy = correct.mean()

    # Class distribution
    unique_classes, class_counts = np.unique(true_labels_np, return_counts=True)
    class_distribution = dict(zip(unique_classes.tolist(), class_counts.tolist()))

    # Per-class accuracy
    per_class_accuracy = {}
    for cls in unique_classes:
        mask = true_labels_np == cls
        if mask.sum() > 0:
            per_class_accuracy[int(cls)] = float(correct[mask].mean())

    # Per-sample detailed info (for first 100 samples to save memory)
    per_sample_info = []
    for i in range(min(100, num_samples)):
        per_sample_info.append({
            'sample_idx': i,
            'true_label': int(true_labels_np[i]),
            'predicted_label': int(predictions[i]),
            'correct': bool(correct[i]),
            'confidence': float(probs_np[i].max()),
            'true_class_prob': float(probs_np[i, true_labels_np[i]])
        })

    return {
        'predictions': predictions,
        'true_labels': true_labels_np,
        'probabilities': probs_np,
        'correct': correct,
        'accuracy': accuracy,
        'num_correct': int(correct.sum()),
        'num_total': num_samples,
        'class_distribution': class_distribution,
        'per_class_accuracy': per_class_accuracy,
        'per_sample_info': per_sample_info
    }


def compare_projection_samples_before_after(
    results_before: Dict[str, Any],
    results_after: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare model performance on projection samples before and after editing.

    These are the FT-Train samples used to construct the AlphaEdit projection matrix P.
    Changes in performance here indicate how well the edit preserves knowledge.

    Args:
        results_before: Results from evaluate_projection_samples before editing
        results_after: Results from evaluate_projection_samples after editing

    Returns:
        Dictionary with comparison metrics
    """
    if results_before is None or results_after is None:
        return None

    preds_before = results_before['predictions']
    preds_after = results_after['predictions']
    true_labels = results_before['true_labels']

    correct_before = results_before['correct']
    correct_after = results_after['correct']

    # Transition analysis
    fixed = (~correct_before) & correct_after  # Was wrong, now correct
    broken = correct_before & (~correct_after)  # Was correct, now wrong (REGRESSION!)
    stayed_correct = correct_before & correct_after
    stayed_wrong = (~correct_before) & (~correct_after)

    # Per-class stability analysis
    per_class_stability = {}
    for cls in np.unique(true_labels):
        mask = true_labels == cls
        cls_correct_before = correct_before[mask]
        cls_correct_after = correct_after[mask]
        cls_stayed_correct = (cls_correct_before & cls_correct_after).sum()
        cls_total_correct_before = cls_correct_before.sum()

        per_class_stability[int(cls)] = {
            'stability': float(cls_stayed_correct / max(1, cls_total_correct_before)),
            'correct_before': int(cls_total_correct_before),
            'correct_after': int(cls_correct_after.sum()),
            'broken': int((cls_correct_before & ~cls_correct_after).sum())
        }

    comparison = {
        'accuracy_before': results_before['accuracy'],
        'accuracy_after': results_after['accuracy'],
        'accuracy_delta': results_after['accuracy'] - results_before['accuracy'],
        'num_fixed': int(fixed.sum()),
        'num_broken': int(broken.sum()),  # This is REGRESSION - should be minimal!
        'num_stayed_correct': int(stayed_correct.sum()),
        'num_stayed_wrong': int(stayed_wrong.sum()),
        'num_total': len(preds_before),
        'stability': float(stayed_correct.sum()) / max(1, correct_before.sum()),
        'regression_rate': float(broken.sum()) / max(1, correct_before.sum()),
        'per_class_stability': per_class_stability
    }

    return comparison


def print_projection_samples_comparison(
    comparison: Dict[str, Any],
    results_before: Dict[str, Any],
    results_after: Dict[str, Any]
):
    """
    Print a formatted comparison of projection samples (FT-Train) before and after editing.

    This is critical for evaluating knowledge preservation - these samples were used
    to construct the projection matrix P, so changes here indicate whether the
    null-space projection is working correctly.
    """
    if comparison is None:
        print("\n[WARNING] No projection samples comparison available")
        return

    print("\n" + "=" * 70)
    print("PROJECTION SAMPLES (FT-Train) PERFORMANCE COMPARISON")
    print("These samples were used to construct AlphaEdit's projection matrix P")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Total Projection Samples: {comparison['num_total']}")
    print(f"  Accuracy Before: {comparison['accuracy_before']*100:.2f}% "
          f"({results_before['num_correct']}/{results_before['num_total']})")
    print(f"  Accuracy After:  {comparison['accuracy_after']*100:.2f}% "
          f"({results_after['num_correct']}/{results_after['num_total']})")
    print(f"  Accuracy Change: {comparison['accuracy_delta']*100:+.2f}%")

    print(f"\nKnowledge Preservation Analysis:")
    print(f"  Stability (correct->correct): {comparison['stability']*100:.1f}% "
          f"({comparison['num_stayed_correct']}/{results_before['num_correct']})")
    print(f"  Regression (correct->wrong):  {comparison['regression_rate']*100:.1f}% "
          f"({comparison['num_broken']}/{results_before['num_correct']})")
    print(f"  Fixed (wrong->correct):       {comparison['num_fixed']} samples")
    print(f"  Stayed Wrong:                 {comparison['num_stayed_wrong']} samples")

    # Per-class stability
    print(f"\nPer-Class Stability (sorted by regression count):")
    print(f"{'Class':>6} {'Before':>8} {'After':>8} {'Broken':>8} {'Stability':>10}")
    print("-" * 45)

    sorted_classes = sorted(
        comparison['per_class_stability'].items(),
        key=lambda x: x[1]['broken'],
        reverse=True
    )

    for cls, stats in sorted_classes[:10]:  # Show top 10
        print(f"{cls:>6} {stats['correct_before']:>8} {stats['correct_after']:>8} "
              f"{stats['broken']:>8} {stats['stability']*100:>9.1f}%")

    # Judgment
    print(f"\nKnowledge Preservation Judgment:")
    if comparison['stability'] >= 0.99 and comparison['num_broken'] == 0:
        print(f"  [EXCELLENT] Perfect preservation - no regressions on projection samples!")
    elif comparison['stability'] >= 0.95:
        print(f"  [GOOD] High stability (>95%) - null-space projection working well")
    elif comparison['stability'] >= 0.90:
        print(f"  [WARNING] Moderate stability (90-95%) - some knowledge loss")
    else:
        print(f"  [POOR] Low stability (<90%) - significant knowledge loss, check projection matrix")

    if comparison['num_broken'] > 0:
        print(f"  [INFO] {comparison['num_broken']} previously correct samples now wrong (regression)")

    if comparison['accuracy_delta'] < -0.01:
        print(f"  [WARNING] Accuracy dropped by {abs(comparison['accuracy_delta'])*100:.2f}%")
    elif comparison['accuracy_delta'] > 0.01:
        print(f"  [INFO] Accuracy improved by {comparison['accuracy_delta']*100:.2f}% (unexpected but good)")

    print("=" * 70)

def main():
    """Test evaluator functionality."""
    print("=" * 70)
    print("ViT Model Editing Pipeline - Evaluator")
    print("=" * 70)
    
    from transformers import ViTForImageClassification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=9,
        ignore_mismatched_sizes=True
    )
    
    # Initialize evaluator
    evaluator = Evaluator(model, device)
    
    # Demo with random data
    from torch.utils.data import DataLoader, TensorDataset
    
    dummy_images = torch.randn(100, 3, 224, 224)
    dummy_labels = torch.randint(0, 9, (100,))
    
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=16)
    
    # Run evaluation
    print("\nRunning inference...")
    evaluator.run_inference(dummy_loader)
    
    # Print summary
    evaluator.print_summary()
    
    # Export results
    evaluator.export_confusion_matrix()
    evaluator.export_evaluation_report()
    evaluator.export_predictions()
    
    print("\n✓ Evaluator test complete!")


if __name__ == "__main__":
    main()
