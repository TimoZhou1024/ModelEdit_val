"""
Locator Module for ViT Model Editing Pipeline (ASTRA Adaptation)
================================================================
Identifies important layers for specific error samples using:
- Patch-level ablation study
- Lasso regression for importance scoring
- Layer-wise activation analysis

Based on ASTRA methodology adapted for Vision Transformers.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from tqdm import tqdm
from sklearn.linear_model import Lasso
from PIL import Image
import contextlib
from collections import OrderedDict

from transformers import ViTForImageClassification


class Trace(contextlib.AbstractContextManager):
    """
    Single-layer activation tracer.
    Adapted from nethook.py for ViT models.
    """
    
    def __init__(
        self,
        module: nn.Module,
        layer: str = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = True,
        edit_output: Callable = None,
        stop: bool = False
    ):
        self.module = module
        self.layer = layer
        self.retain_output = retain_output
        self.retain_input = retain_input
        self.clone = clone
        self.detach = detach
        self.edit_output = edit_output
        self.stop = stop
        
        self.output = None
        self.input = None
        self._hook = None
        
    def __enter__(self):
        if self.layer is not None:
            target = self._get_module(self.module, self.layer)
        else:
            target = self.module
            
        self._hook = target.register_forward_hook(self._hook_fn)
        return self
    
    def __exit__(self, *args):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        
    def _hook_fn(self, module, input, output):
        if self.retain_input:
            self.input = self._process(input[0] if isinstance(input, tuple) else input)
        if self.retain_output:
            out = output[0] if isinstance(output, tuple) else output
            if self.edit_output is not None:
                out = self.edit_output(out)
            self.output = self._process(out)
        if self.stop:
            raise StopIteration()
    
    def _process(self, x):
        if x is None:
            return None
        if self.clone:
            x = x.clone()
        if self.detach:
            x = x.detach()
        return x
    
    @staticmethod
    def _get_module(model, name):
        """Get a submodule by dot-separated name."""
        for part in name.split('.'):
            model = getattr(model, part)
        return model


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    Multi-layer activation tracer.
    Traces activations from multiple layers simultaneously.
    """
    
    def __init__(
        self,
        module: nn.Module,
        layers: List[str],
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = True,
        edit_output: Callable = None
    ):
        super().__init__()
        self.module = module
        self.layers = layers
        self.retain_output = retain_output
        self.retain_input = retain_input
        self.clone = clone
        self.detach = detach
        self.edit_output = edit_output
        
        self._hooks = []
        
        for layer in layers:
            self[layer] = {'output': None, 'input': None}
    
    def __enter__(self):
        for layer in self.layers:
            target = self._get_module(self.module, layer)
            hook = target.register_forward_hook(
                self._make_hook_fn(layer)
            )
            self._hooks.append(hook)
        return self
    
    def __exit__(self, *args):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def _make_hook_fn(self, layer_name):
        def hook_fn(module, input, output):
            if self.retain_input:
                inp = input[0] if isinstance(input, tuple) else input
                self[layer_name]['input'] = self._process(inp)
            if self.retain_output:
                out = output[0] if isinstance(output, tuple) else output
                if self.edit_output is not None:
                    out = self.edit_output(out, layer_name)
                self[layer_name]['output'] = self._process(out)
        return hook_fn
    
    def _process(self, x):
        if x is None:
            return None
        if self.clone:
            x = x.clone()
        if self.detach:
            x = x.detach()
        return x
    
    @staticmethod
    def _get_module(model, name):
        for part in name.split('.'):
            model = getattr(model, part)
        return model


class PatchAblator:
    """
    Performs patch-level ablation on ViT inputs.
    Creates masked versions of input images by zeroing out patches.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        ablation_value: float = 0.0
    ):
        """
        Args:
            image_size: Input image size (224 for ViT-B/16)
            patch_size: ViT patch size (16 for ViT-B/16)
            ablation_value: Value to replace ablated patches (0.0 = black)
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.ablation_value = ablation_value
        self.num_patches = (image_size // patch_size) ** 2  # 196 for ViT-B/16
        self.patches_per_side = image_size // patch_size  # 14
        
    def ablate_patches(
        self,
        image: torch.Tensor,
        mask: np.ndarray
    ) -> torch.Tensor:
        """
        Ablate patches according to binary mask.
        
        Args:
            image: Input tensor of shape (C, H, W) or (B, C, H, W)
            mask: Binary mask of shape (num_patches,), 1=keep, 0=ablate
            
        Returns:
            Ablated image tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, C, H, W = image.shape
        ablated = image.clone()
        
        for patch_idx in range(self.num_patches):
            if mask[patch_idx] == 0:  # Ablate this patch
                row = patch_idx // self.patches_per_side
                col = patch_idx % self.patches_per_side
                
                y_start = row * self.patch_size
                y_end = y_start + self.patch_size
                x_start = col * self.patch_size
                x_end = x_start + self.patch_size
                
                ablated[:, :, y_start:y_end, x_start:x_end] = self.ablation_value
        
        if squeeze:
            ablated = ablated.squeeze(0)
        
        return ablated
    
    def generate_random_masks(
        self,
        num_ablations: int,
        keep_prob: float = 0.5
    ) -> np.ndarray:
        """
        Generate random ablation masks.
        
        Args:
            num_ablations: Number of masks to generate
            keep_prob: Probability of keeping each patch
            
        Returns:
            Binary masks of shape (num_ablations, num_patches)
        """
        return (np.random.rand(num_ablations, self.num_patches) < keep_prob).astype(np.float32)


class LayerImportanceScorer:
    """
    Computes importance scores for each layer using Lasso regression.
    Based on ASTRA's image attribution methodology.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        layers: List[str] = None
    ):
        """
        Args:
            model: ViT model
            device: Computation device
            layers: List of layer names to analyze. If None, uses all encoder layers.
        """
        self.model = model
        self.device = device
        
        # Default: analyze all encoder layers
        if layers is None:
            self.layers = [f"vit.encoder.layer.{i}" for i in range(12)]
        else:
            self.layers = layers
        
        self.ablator = PatchAblator()
        self.lasso = Lasso(alpha=0.01, max_iter=1000)
        
    def compute_layer_importance(
        self,
        image: torch.Tensor,
        true_label: int,
        num_ablations: int = 64,
        keep_prob: float = 0.5,
        token_position: str = "cls"
    ) -> Dict[str, float]:
        """
        Compute importance score for each layer for a given sample.
        
        The importance score measures how much each layer's activation
        contributes to the model's prediction for the true class.
        
        Args:
            image: Input image tensor (C, H, W)
            true_label: Ground truth class label
            num_ablations: Number of random ablations
            keep_prob: Probability of keeping each patch
            token_position: Which token to analyze ('cls' or 'mean')
            
        Returns:
            Dictionary mapping layer names to importance scores
        """
        self.model.eval()
        
        # Generate random masks
        masks = self.ablator.generate_random_masks(num_ablations, keep_prob)
        
        # Storage for layer activations
        layer_activations = {layer: [] for layer in self.layers}
        output_probs = []
        
        # Run ablations
        for mask in tqdm(masks, desc="Ablating", leave=False):
            ablated_image = self.ablator.ablate_patches(image, mask)
            ablated_image = ablated_image.unsqueeze(0).to(self.device)
            
            # Forward pass with activation tracing
            with TraceDict(self.model, self.layers, retain_output=True) as traces:
                with torch.no_grad():
                    outputs = self.model(ablated_image)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    output_probs.append(probs[0, true_label].cpu().item())
                
                # Collect activations
                for layer in self.layers:
                    act = traces[layer]['output']  # Shape: (1, 197, 768)
                    if token_position == "cls":
                        # Use CLS token (position 0)
                        act = act[0, 0, :]  # (768,)
                    else:
                        # Use mean of all tokens
                        act = act[0].mean(dim=0)  # (768,)
                    layer_activations[layer].append(act.cpu().numpy())
        
        # Convert to numpy arrays
        output_probs = np.array(output_probs)  # (num_ablations,)
        
        # Compute importance using Lasso regression for each layer
        importance_scores = {}
        
        for layer in self.layers:
            activations = np.stack(layer_activations[layer])  # (num_ablations, 768)
            
            # Fit Lasso: predict output probability from activations
            try:
                self.lasso.fit(activations, output_probs)
                # Importance = sum of absolute coefficients
                importance = np.sum(np.abs(self.lasso.coef_))
                importance_scores[layer] = float(importance)
            except Exception as e:
                print(f"Warning: Lasso fit failed for {layer}: {e}")
                importance_scores[layer] = 0.0
        
        return importance_scores
    
    def compute_activation_difference(
        self,
        original_image: torch.Tensor,
        ablated_image: torch.Tensor,
        token_position: str = "cls"
    ) -> Dict[str, np.ndarray]:
        """
        Compute activation difference between original and ablated images.
        This is the steering vector in ASTRA terminology.
        
        Args:
            original_image: Original image tensor (C, H, W)
            ablated_image: Ablated image tensor (C, H, W)
            token_position: Which token to analyze
            
        Returns:
            Dictionary mapping layer names to activation differences
        """
        self.model.eval()
        
        original = original_image.unsqueeze(0).to(self.device)
        ablated = ablated_image.unsqueeze(0).to(self.device)
        
        diff_vectors = {}
        
        with TraceDict(self.model, self.layers, retain_output=True) as traces:
            with torch.no_grad():
                # Forward pass for original
                _ = self.model(original)
                original_acts = {
                    layer: traces[layer]['output'].clone()
                    for layer in self.layers
                }
                
        with TraceDict(self.model, self.layers, retain_output=True) as traces:
            with torch.no_grad():
                # Forward pass for ablated
                _ = self.model(ablated)
                ablated_acts = {
                    layer: traces[layer]['output'].clone()
                    for layer in self.layers
                }
        
        # Compute differences
        for layer in self.layers:
            orig = original_acts[layer]
            abl = ablated_acts[layer]
            
            if token_position == "cls":
                diff = (orig[0, 0, :] - abl[0, 0, :]).cpu().numpy()
            else:
                diff = (orig[0].mean(dim=0) - abl[0].mean(dim=0)).cpu().numpy()
            
            diff_vectors[layer] = diff
        
        return diff_vectors


class Locator:
    """
    Main class for layer localization in ViT models.
    Identifies which layers are most important for specific error samples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        log_dir: str = "logs",
        target_layers: List[int] = None
    ):
        """
        Args:
            model: Fine-tuned ViT model
            device: Computation device (auto-detect if None)
            log_dir: Directory for saving results
            target_layers: Which layer indices to analyze (default: all 12)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(device)
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Default: analyze all encoder layers
        if target_layers is None:
            target_layers = list(range(12))
        
        self.target_layers = target_layers
        self.layer_names = [f"vit.encoder.layer.{i}" for i in target_layers]
        
        self.scorer = LayerImportanceScorer(
            model=self.model,
            device=self.device,
            layers=self.layer_names
        )
        
        # Results storage
        self.importance_results = []
        
    def analyze_sample(
        self,
        image: torch.Tensor,
        true_label: int,
        predicted_label: int,
        sample_idx: int,
        num_ablations: int = 64
    ) -> Dict[str, Any]:
        """
        Analyze a single misclassified sample.
        
        Args:
            image: Input image tensor
            true_label: Ground truth label
            predicted_label: Model's prediction
            sample_idx: Index of the sample in the dataset
            num_ablations: Number of ablation trials
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\nAnalyzing sample {sample_idx}: true={true_label}, pred={predicted_label}")
        
        # Compute importance scores
        importance_scores = self.scorer.compute_layer_importance(
            image=image,
            true_label=true_label,
            num_ablations=num_ablations
        )
        
        # Create result record
        result = {
            'sample_idx': sample_idx,
            'true_label': true_label,
            'predicted_label': predicted_label,
            **{f'layer_{i}_importance': importance_scores.get(f'vit.encoder.layer.{i}', 0.0)
               for i in self.target_layers}
        }
        
        # Find most important layers
        sorted_layers = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        result['top_layer'] = sorted_layers[0][0] if sorted_layers else None
        result['top_importance'] = sorted_layers[0][1] if sorted_layers else 0.0
        
        self.importance_results.append(result)
        
        return result
    
    def analyze_batch(
        self,
        images: List[torch.Tensor],
        true_labels: List[int],
        predicted_labels: List[int],
        sample_indices: List[int],
        num_ablations: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple misclassified samples.
        """
        results = []
        
        for img, true_label, pred_label, idx in tqdm(
            zip(images, true_labels, predicted_labels, sample_indices),
            total=len(images),
            desc="Analyzing samples"
        ):
            result = self.analyze_sample(
                image=img,
                true_label=true_label,
                predicted_label=pred_label,
                sample_idx=idx,
                num_ablations=num_ablations
            )
            results.append(result)
        
        return results
    
    def get_layer_statistics(self) -> pd.DataFrame:
        """
        Compute aggregate statistics across all analyzed samples.
        """
        if not self.importance_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.importance_results)
        
        # Compute mean importance per layer
        layer_cols = [c for c in df.columns if c.startswith('layer_') and c.endswith('_importance')]
        
        stats = []
        for col in layer_cols:
            layer_idx = int(col.split('_')[1])
            stats.append({
                'layer_index': layer_idx,
                'layer_name': f'vit.encoder.layer.{layer_idx}',
                'mean_importance': df[col].mean(),
                'std_importance': df[col].std(),
                'max_importance': df[col].max(),
                'min_importance': df[col].min()
            })
        
        return pd.DataFrame(stats).sort_values('mean_importance', ascending=False)
    
    def get_top_layers(self, n: int = 3) -> List[int]:
        """
        Get the top N most important layer indices.
        """
        stats = self.get_layer_statistics()
        if stats.empty:
            return []
        
        return stats.head(n)['layer_index'].tolist()
    
    def export_results(self, filename: str = "layer_importance.csv") -> str:
        """
        Export all analysis results to CSV.
        """
        if not self.importance_results:
            print("No results to export.")
            return None
        
        df = pd.DataFrame(self.importance_results)
        csv_path = self.log_dir / filename
        df.to_csv(csv_path, index=False)
        
        print(f"\nResults exported to: {csv_path}")
        
        # Also export aggregate statistics
        stats = self.get_layer_statistics()
        if not stats.empty:
            stats_path = self.log_dir / "layer_statistics.csv"
            stats.to_csv(stats_path, index=False)
            print(f"Statistics exported to: {stats_path}")
        
        return str(csv_path)
    
    def visualize_importance(self) -> None:
        """
        Print a text visualization of layer importance.
        """
        stats = self.get_layer_statistics()
        if stats.empty:
            print("No data to visualize.")
            return
        
        print("\n" + "=" * 60)
        print("Layer Importance Summary")
        print("=" * 60)
        
        max_imp = stats['mean_importance'].max()
        
        for _, row in stats.iterrows():
            layer_idx = int(row['layer_index'])
            importance = row['mean_importance']
            bar_len = int(40 * importance / max_imp) if max_imp > 0 else 0
            bar = "█" * bar_len
            print(f"Layer {layer_idx:2d}: {bar:<40} {importance:.4f}")
        
        print("=" * 60)
        top_layers = self.get_top_layers(3)
        print(f"Top 3 layers for editing: {top_layers}")


def main():
    """Test locator functionality."""
    print("=" * 70)
    print("ViT Model Editing Pipeline - Layer Locator (ASTRA)")
    print("=" * 70)
    
    # This is a demonstration - requires trained model
    from transformers import ViTForImageClassification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=9,
        ignore_mismatched_sizes=True
    )
    
    # Initialize locator
    locator = Locator(
        model=model,
        device=device,
        target_layers=[4, 5, 6, 7, 8, 9, 10, 11]  # Focus on later layers
    )
    
    # Demo with random image
    dummy_image = torch.randn(3, 224, 224)
    result = locator.analyze_sample(
        image=dummy_image,
        true_label=0,
        predicted_label=1,
        sample_idx=0,
        num_ablations=16  # Reduced for demo
    )
    
    print(f"\nSample analysis result: {result}")
    
    locator.visualize_importance()
    locator.export_results()
    
    print("\n✓ Locator test complete!")


if __name__ == "__main__":
    main()
