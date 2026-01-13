"""
Editor Module for ViT Model Editing Pipeline (AlphaEdit Adaptation)
===================================================================
Applies targeted weight edits to ViT models using null-space projection.

Key AlphaEdit formulas adapted for ViT:
- Projection matrix: P = Û @ Û^T (from SVD of preserved knowledge covariance)
- Update: Δ = R @ K^T @ P @ (K @ K^T @ P + λI)^{-1}
- Where R = V_target - W @ K (residual to desired output)

Target modules in ViT-B/16:
- vit.encoder.layer.{i}.intermediate.dense (similar to MLP up_proj)
- vit.encoder.layer.{i}.output.dense (similar to MLP down_proj)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from copy import deepcopy
import contextlib
from collections import OrderedDict


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """Multi-layer activation tracer for collecting K vectors."""
    
    def __init__(
        self,
        module: nn.Module,
        layers: List[str],
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = True,
        detach: bool = True
    ):
        super().__init__()
        self.module = module
        self.layers = layers
        self.retain_output = retain_output
        self.retain_input = retain_input
        self.clone = clone
        self.detach = detach
        self._hooks = []
        
        for layer in layers:
            self[layer] = {'output': None, 'input': None}
    
    def __enter__(self):
        for layer in self.layers:
            target = self._get_module(self.module, layer)
            hook = target.register_forward_hook(self._make_hook_fn(layer))
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


class AlphaEditHyperParams:
    """Hyperparameters for AlphaEdit weight editing."""
    
    def __init__(
        self,
        # Target layers
        layers: List[int] = None,
        # Module templates for ViT
        rewrite_module_tmp: str = "vit.encoder.layer.{}.output.dense",
        layer_module_tmp: str = "vit.encoder.layer.{}",
        # Optimization
        v_num_grad_steps: int = 25,
        v_lr: float = 0.1,
        v_weight_decay: float = 0.01,
        # Regularization
        nullspace_threshold: float = 1e-2,
        L2: float = 1e-4,
        clamp_norm_factor: float = 0.75,
        # Token position
        fact_token: str = "cls",  # 'cls' for CLS token, 'mean' for all patches
    ):
        self.layers = layers if layers is not None else [8, 9, 10, 11]
        self.rewrite_module_tmp = rewrite_module_tmp
        self.layer_module_tmp = layer_module_tmp
        self.v_num_grad_steps = v_num_grad_steps
        self.v_lr = v_lr
        self.v_weight_decay = v_weight_decay
        self.nullspace_threshold = nullspace_threshold
        self.L2 = L2
        self.clamp_norm_factor = clamp_norm_factor
        self.fact_token = fact_token


class KCollector:
    """
    Collects K vectors (inputs to target module) for AlphaEdit.
    
    For ViT: K is the input to the MLP output projection layer,
    extracted at the CLS token position.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        hparams: AlphaEditHyperParams
    ):
        self.model = model
        self.device = device
        self.hparams = hparams
    
    def compute_ks(
        self,
        images: torch.Tensor,
        layer: int
    ) -> torch.Tensor:
        """
        Compute K vectors for given images at specified layer.
        
        Args:
            images: Input images of shape (B, C, H, W)
            layer: Target layer index
            
        Returns:
            K vectors of shape (hidden_size, B) - transposed for matrix ops
        """
        self.model.eval()
        
        # Get the module name
        module_name = self.hparams.rewrite_module_tmp.format(layer)
        
        # Trace the input to target module
        with TraceDict(self.model, [module_name], retain_input=True) as traces:
            with torch.no_grad():
                images = images.to(self.device)
                _ = self.model(images)
                
                # Get input activations: shape (B, 197, 768)
                k_input = traces[module_name]['input']
        
        # Extract based on token position
        if self.hparams.fact_token == "cls":
            # Use CLS token (position 0)
            k = k_input[:, 0, :]  # (B, 768)
        else:
            # Use mean of all tokens
            k = k_input.mean(dim=1)  # (B, 768)
        
        # Transpose for matrix operations: (768, B)
        return k.T
    
    def compute_current_output(
        self,
        images: torch.Tensor,
        layer: int
    ) -> torch.Tensor:
        """
        Compute current output of target module.
        
        Returns:
            Current output of shape (B, hidden_size)
        """
        self.model.eval()
        
        module_name = self.hparams.rewrite_module_tmp.format(layer)
        
        with TraceDict(self.model, [module_name], retain_output=True) as traces:
            with torch.no_grad():
                images = images.to(self.device)
                _ = self.model(images)
                
                output = traces[module_name]['output']
        
        if self.hparams.fact_token == "cls":
            return output[:, 0, :]  # (B, 768)
        else:
            return output.mean(dim=1)  # (B, 768)


class ZComputer:
    """
    Computes target Z vectors (desired outputs) for AlphaEdit.
    
    For classification correction, we want the model to output
    representations that lead to the correct class prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        hparams: AlphaEditHyperParams
    ):
        self.model = model
        self.device = device
        self.hparams = hparams
    
    def compute_target_z(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        layer: int
    ) -> torch.Tensor:
        """
        Compute target Z vectors through gradient optimization.
        
        Optimizes delta such that adding it to current activations
        causes the model to predict the correct class.
        
        Args:
            images: Input images (B, C, H, W)
            true_labels: Target class labels (B,)
            layer: Target layer
            
        Returns:
            Target Z vectors (B, hidden_size)
        """
        self.model.eval()
        images = images.to(self.device)
        true_labels = true_labels.to(self.device)
        
        module_name = self.hparams.rewrite_module_tmp.format(layer)
        
        # Get current output
        with TraceDict(self.model, [module_name], retain_output=True) as traces:
            with torch.no_grad():
                _ = self.model(images)
                current_z = traces[module_name]['output'].clone()
        
        # Extract token position
        if self.hparams.fact_token == "cls":
            current_z_token = current_z[:, 0, :].clone()  # (B, 768)
        else:
            current_z_token = current_z.mean(dim=1).clone()  # (B, 768)
        
        # Initialize delta
        delta = torch.zeros_like(current_z_token, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.hparams.v_lr)
        
        # Optimization loop
        for step in range(self.hparams.v_num_grad_steps):
            optimizer.zero_grad()
            
            # Define edit function
            def edit_output(output, layer_name):
                if layer_name == module_name:
                    new_output = output.clone()
                    if self.hparams.fact_token == "cls":
                        new_output[:, 0, :] = current_z_token + delta
                    else:
                        # Broadcast delta to all tokens
                        new_output = new_output + delta.unsqueeze(1)
                    return new_output
                return output
            
            # Forward with edited activations
            with TraceDict(
                self.model,
                [module_name],
                retain_output=True
            ) as traces:
                # Register edit hook
                target_module = TraceDict._get_module(self.model, module_name)
                
                def hook_fn(module, input, output):
                    return edit_output(output, module_name)
                
                hook = target_module.register_forward_hook(hook_fn)
                
                try:
                    outputs = self.model(images)
                    logits = outputs.logits
                finally:
                    hook.remove()
            
            # Classification loss
            loss = nn.CrossEntropyLoss()(logits, true_labels)
            
            # L2 regularization on delta
            loss = loss + self.hparams.v_weight_decay * torch.norm(delta) ** 2
            
            loss.backward()
            optimizer.step()
            
            # Clamp norm
            with torch.no_grad():
                max_norm = self.hparams.clamp_norm_factor * current_z_token.norm(dim=-1, keepdim=True)
                delta_norm = delta.norm(dim=-1, keepdim=True)
                scale = torch.clamp(max_norm / (delta_norm + 1e-8), max=1.0)
                delta.data = delta.data * scale
        
        # Final target Z
        target_z = current_z_token + delta.detach()
        
        return target_z


class NullSpaceProjector:
    """
    Constructs null-space projection matrix P for AlphaEdit.
    
    P projects updates into the null space of preserved knowledge,
    ensuring edits don't affect unrelated inputs.
    """
    
    def __init__(
        self,
        threshold: float = 1e-2
    ):
        self.threshold = threshold
    
    def compute_projection_matrix(
        self,
        K0: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute null-space projection matrix.
        
        P = Û @ Û^T where Û contains eigenvectors corresponding
        to near-zero eigenvalues of K0 @ K0^T.
        
        Args:
            K0: Preserved knowledge matrix of shape (hidden_size, num_samples)
            
        Returns:
            Projection matrix P of shape (hidden_size, hidden_size)
        """
        # Compute covariance: K0 @ K0^T (hidden_size, hidden_size)
        cov = K0 @ K0.T
        
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(cov)
        
        # Find eigenvalues below threshold
        null_mask = S < self.threshold
        
        if not null_mask.any():
            # No null space found, use small regularization
            print("Warning: No null space found, using identity projection")
            return torch.eye(K0.shape[0], device=K0.device, dtype=K0.dtype)
        
        # Extract null space eigenvectors
        U_null = U[:, null_mask]  # (hidden_size, num_null)
        
        # Projection matrix
        P = U_null @ U_null.T  # (hidden_size, hidden_size)
        
        return P
    
    def compute_from_random_samples(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer: int,
        hparams: AlphaEditHyperParams,
        device: torch.device,
        num_samples: int = 1000
    ) -> torch.Tensor:
        """
        Compute projection matrix from random data samples.
        
        Args:
            model: The model
            dataloader: DataLoader with diverse samples
            layer: Target layer
            hparams: Hyperparameters
            device: Computation device
            num_samples: Number of samples to collect
            
        Returns:
            Projection matrix P
        """
        collector = KCollector(model, device, hparams)
        
        K_list = []
        total = 0
        
        for images, _ in tqdm(dataloader, desc="Collecting K0"):
            if total >= num_samples:
                break
            
            k = collector.compute_ks(images, layer)  # (hidden_size, B)
            K_list.append(k.cpu())
            total += images.shape[0]
        
        K0 = torch.cat(K_list, dim=1).to(device)  # (hidden_size, num_samples)
        
        return self.compute_projection_matrix(K0)


class Editor:
    """
    Main class for AlphaEdit weight editing in ViT models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        hparams: AlphaEditHyperParams = None,
        log_dir: str = "logs"
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model = model.to(device)
        self.hparams = hparams if hparams else AlphaEditHyperParams()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.k_collector = KCollector(model, device, self.hparams)
        self.z_computer = ZComputer(model, device, self.hparams)
        self.projector = NullSpaceProjector(self.hparams.nullspace_threshold)
        
        # Cache for sequential edits
        self.cache_KKT = {}  # layer -> K_prev @ K_prev^T
        self.P = {}  # layer -> projection matrix
        
        # Edit history
        self.edit_history = []
    
    def precompute_projection(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 1000
    ):
        """
        Precompute null-space projection matrices for all target layers.
        """
        print("\nPrecomputing projection matrices...")
        
        for layer in tqdm(self.hparams.layers, desc="Layers"):
            P = self.projector.compute_from_random_samples(
                model=self.model,
                dataloader=dataloader,
                layer=layer,
                hparams=self.hparams,
                device=self.device,
                num_samples=num_samples
            )
            self.P[layer] = P
            
            # Initialize cache
            hidden_size = P.shape[0]
            self.cache_KKT[layer] = torch.zeros(
                (hidden_size, hidden_size),
                device=self.device,
                dtype=P.dtype
            )
        
        print(f"Projection matrices computed for layers: {self.hparams.layers}")
    
    def apply_edit(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        sample_indices: List[int] = None
    ) -> Dict[str, Any]:
        """
        Apply AlphaEdit to correct misclassified samples.
        
        Args:
            images: Misclassified images (B, C, H, W)
            true_labels: Correct labels (B,)
            sample_indices: Optional indices for logging
            
        Returns:
            Dictionary with edit information
        """
        images = images.to(self.device)
        true_labels = true_labels.to(self.device)
        
        if sample_indices is None:
            sample_indices = list(range(len(images)))
        
        edit_info = {
            'sample_indices': sample_indices,
            'num_samples': len(images),
            'layer_updates': {}
        }
        
        # Process each target layer
        num_layers = len(self.hparams.layers)
        
        # First compute all target Z vectors
        target_zs = {}
        for layer in self.hparams.layers:
            target_z = self.z_computer.compute_target_z(images, true_labels, layer)
            target_zs[layer] = target_z
        
        for i, layer in enumerate(self.hparams.layers):
            print(f"\nEditing layer {layer} ({i+1}/{num_layers})...")
            
            # Get K vectors: (hidden_size, B)
            K = self.k_collector.compute_ks(images, layer)
            
            # Get current output: (B, hidden_size)
            current_z = self.k_collector.compute_current_output(images, layer)
            
            # Target output
            target_z = target_zs[layer]
            
            # Residual: R = V_target - W @ K
            # In our case: R = target_z - current_z
            R = (target_z - current_z).T  # (hidden_size, B)
            
            # Distribute residual across remaining layers
            R = R / (num_layers - i)
            
            # Get projection matrix
            P = self.P.get(layer)
            if P is None:
                # Use identity if not precomputed
                P = torch.eye(K.shape[0], device=self.device, dtype=K.dtype)
            
            # Get cached K @ K^T from previous edits
            cache_KKT = self.cache_KKT.get(layer, torch.zeros_like(P))
            
            # Compute update using AlphaEdit formula:
            # Δ = R @ K^T @ P @ (K @ K^T @ P + cache_KKT @ P + λI)^{-1}
            
            KKT = K @ K.T  # (hidden_size, hidden_size)
            
            # A = K @ K^T @ P + cache_KKT @ P + λI
            A = KKT @ P + cache_KKT @ P + self.hparams.L2 * torch.eye(
                K.shape[0], device=self.device, dtype=K.dtype
            )
            
            # Solve: Δ @ A = R @ K^T @ P
            # => Δ = R @ K^T @ P @ A^{-1}
            RKT_P = R @ K.T @ P  # (hidden_size, hidden_size)
            
            try:
                delta = torch.linalg.solve(A.T, RKT_P.T).T  # (hidden_size, hidden_size)
            except Exception as e:
                print(f"Warning: Solve failed, using pseudoinverse: {e}")
                A_inv = torch.linalg.pinv(A)
                delta = RKT_P @ A_inv
            
            # Apply update to model weights
            module_name = self.hparams.rewrite_module_tmp.format(layer)
            target_module = self._get_module(self.model, module_name)
            
            # The weight matrix shape is (out_features, in_features)
            # delta is (hidden_size, hidden_size), we need to adjust
            with torch.no_grad():
                # For nn.Linear: y = x @ W^T + b
                # We want to add delta to the output, so we modify W
                # New output = old_output + delta @ K = x @ W^T + x @ delta^T
                # So W_new = W + delta^T (if delta acts on input)
                
                # Actually, for the output.dense layer:
                # Input: (B, 197, 3072) -> Output: (B, 197, 768)
                # Weight: (768, 3072)
                
                # We computed delta to modify the output at CLS position
                # This is an approximation - we apply a rank-B update
                
                old_weight = target_module.weight.data.clone()
                
                # Simplified update: scale delta to match weight dimensions
                if delta.shape != old_weight.shape:
                    # Project delta to appropriate dimensions
                    # delta: (768, 768), weight: (768, 3072)
                    # We compute a low-rank update: delta @ K_normalized
                    K_normalized = K / (K.norm(dim=0, keepdim=True) + 1e-8)
                    
                    # Expand to full weight dimensions
                    # This is a simplification - for ViT output.dense, input is 3072
                    in_features = old_weight.shape[1]
                    out_features = old_weight.shape[0]
                    
                    # Create a projection from hidden to input dimension
                    if in_features != delta.shape[1]:
                        # Use random projection (simplified)
                        proj = torch.randn(delta.shape[1], in_features, device=self.device)
                        proj = proj / proj.norm(dim=1, keepdim=True)
                        delta_proj = delta @ proj
                    else:
                        delta_proj = delta
                    
                    target_module.weight.data = old_weight + delta_proj
                else:
                    target_module.weight.data = old_weight + delta
                
                update_norm = torch.norm(target_module.weight.data - old_weight).item()
            
            # Update cache
            self.cache_KKT[layer] = cache_KKT + KKT
            
            # Record
            edit_info['layer_updates'][layer] = {
                'update_norm': update_norm,
                'K_norm': K.norm().item(),
                'R_norm': R.norm().item()
            }
            
            print(f"  Update norm: {update_norm:.6f}")
        
        # Record edit
        self.edit_history.append(edit_info)
        
        return edit_info
    
    def apply_batch_edit(
        self,
        dataloader: torch.utils.data.DataLoader,
        misclassified_info: Dict[str, Any],
        max_edits: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Apply edits to multiple misclassified samples.
        """
        indices = misclassified_info['indices'][:max_edits]
        
        results = []
        
        # Group by batches
        dataset = dataloader.dataset
        
        for idx in tqdm(indices, desc="Applying edits"):
            image, label = dataset[idx]
            image = image.unsqueeze(0)
            label = torch.tensor([label])
            
            result = self.apply_edit(
                images=image,
                true_labels=label,
                sample_indices=[idx]
            )
            results.append(result)
        
        return results
    
    def export_edit_log(self, filename: str = "edit_log.csv") -> str:
        """Export edit history to CSV."""
        if not self.edit_history:
            print("No edit history to export.")
            return None
        
        rows = []
        for edit in self.edit_history:
            for layer, info in edit['layer_updates'].items():
                rows.append({
                    'sample_indices': str(edit['sample_indices']),
                    'layer': layer,
                    'update_norm': info['update_norm'],
                    'K_norm': info['K_norm'],
                    'R_norm': info['R_norm']
                })
        
        df = pd.DataFrame(rows)
        csv_path = self.log_dir / filename
        df.to_csv(csv_path, index=False)
        
        print(f"Edit log exported to: {csv_path}")
        return str(csv_path)
    
    def save_edited_model(self, filepath: str = None) -> str:
        """Save the edited model."""
        if filepath is None:
            filepath = Path("checkpoints") / "vit_pathmnist_edited.pt"
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'edit_history': self.edit_history,
            'hparams': vars(self.hparams)
        }, filepath)
        
        print(f"Edited model saved to: {filepath}")
        return str(filepath)
    
    @staticmethod
    def _get_module(model, name):
        for part in name.split('.'):
            model = getattr(model, part)
        return model


def main():
    """Test editor functionality."""
    print("=" * 70)
    print("ViT Model Editing Pipeline - Weight Editor (AlphaEdit)")
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
    
    # Initialize editor with custom hyperparameters
    hparams = AlphaEditHyperParams(
        layers=[9, 10, 11],  # Edit later layers
        v_num_grad_steps=10,  # Reduced for demo
        L2=1e-3
    )
    
    editor = Editor(
        model=model,
        device=device,
        hparams=hparams
    )
    
    # Demo with random data
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1])
    
    print("\nApplying test edit...")
    result = editor.apply_edit(
        images=dummy_images,
        true_labels=dummy_labels,
        sample_indices=[0, 1]
    )
    
    print(f"\nEdit result: {result}")
    
    editor.export_edit_log()
    editor.save_edited_model()
    
    print("\n✓ Editor test complete!")


if __name__ == "__main__":
    main()
