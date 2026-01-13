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
