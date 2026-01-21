"""
Editor Module for ViT Model Editing Pipeline (AlphaEdit Adaptation)
===================================================================
Applies targeted weight edits to ViT models using null-space projection.

Key AlphaEdit formulas adapted for ViT (matching original implementation):
- Covariance matrix: C = E[K @ K^T] = (K0 @ K0^T) / num_samples
- Projection matrix: P = Û @ Û^T (from SVD of C, using small singular values)
- Update formula: Δ = [P @ (K @ K^T + cache_c) + λI]^{-1} @ P @ K @ R^T
- Where R = Z_target - Z_current (residual to desired output)

IMPORTANT implementation details (matching AlphaEdit paper):
1. Covariance matrix MUST be normalized by sample count
2. P is multiplied OUTSIDE (KKT + cache_c): P @ (KKT + cache_c)
3. Solve form: Δ = A^{-1} @ B, not Δ = B @ A^{-1}
4. Cache update happens AFTER all layers are edited in a round

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


class HeadEditHyperParams:
    """Hyperparameters for Head Editing (classifier-only modification)."""

    def __init__(
        self,
        # Optimization
        num_steps: int = 50,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        # EWC Regularization
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 500,
        # Method
        closed_form: bool = False,
        # Regularization strength for closed-form
        reg_lambda: float = 1.0,
    ):
        self.num_steps = num_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.closed_form = closed_form
        self.reg_lambda = reg_lambda


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
            K vectors of shape (in_features, B) - transposed for matrix ops
            where in_features is the input dimension of the target module
        """
        self.model.eval()

        # Get the module name
        module_name = self.hparams.rewrite_module_tmp.format(layer)

        # Trace the input to target module
        with TraceDict(self.model, [module_name], retain_input=True) as traces:
            with torch.no_grad():
                images = images.to(self.device)
                _ = self.model(images)

                # Get input activations
                # For vit.encoder.layer.{}.output.dense:
                #   input shape: (B, 197, 3072) - from intermediate layer
                #   output shape: (B, 197, 768) - back to hidden size
                k_input = traces[module_name]['input']

        # Extract based on token position
        if self.hparams.fact_token == "cls":
            # Use CLS token (position 0)
            k = k_input[:, 0, :]  # (B, in_features)
        else:
            # Use mean of all tokens
            k = k_input.mean(dim=1)  # (B, in_features)

        # Transpose for matrix operations: (in_features, B)
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
        to near-zero eigenvalues of E[K @ K^T] (second moment / covariance).

        Args:
            K0: Preserved knowledge matrix of shape (in_features, num_samples)
                where in_features is the input dimension of the target module

        Returns:
            Projection matrix P of shape (in_features, in_features)
        """
        # Compute covariance (second moment): E[K @ K^T] = (K0 @ K0^T) / num_samples
        # IMPORTANT: Must normalize by sample count to match AlphaEdit original implementation
        num_samples = K0.shape[1]
        cov = (K0 @ K0.T) / num_samples  # (in_features, in_features)

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
        num_samples: int = 1000,
        return_sample_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Compute projection matrix from random data samples.

        Args:
            model: The model
            dataloader: DataLoader with diverse samples
            layer: Target layer
            hparams: Hyperparameters
            device: Computation device
            num_samples: Number of samples to collect
            return_sample_info: If True, also return info about samples used

        Returns:
            Tuple of (Projection matrix P, sample_info dict or None)
            sample_info contains: images, labels, batch_indices for evaluation
        """
        collector = KCollector(model, device, hparams)

        K_list = []
        total = 0

        # Track sample information if requested
        sample_images = [] if return_sample_info else None
        sample_labels = [] if return_sample_info else None
        batch_start_indices = [] if return_sample_info else None
        current_idx = 0

        for images, labels in tqdm(dataloader, desc="Collecting K0"):
            if total >= num_samples:
                break

            k = collector.compute_ks(images, layer)  # (hidden_size, B)
            K_list.append(k.cpu())

            if return_sample_info:
                # Store sample data for later evaluation
                sample_images.append(images.cpu())
                sample_labels.append(labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels))
                batch_start_indices.append(current_idx)
                current_idx += images.shape[0]

            total += images.shape[0]

        K0 = torch.cat(K_list, dim=1).to(device)  # (hidden_size, num_samples)
        P = self.compute_projection_matrix(K0)

        sample_info = None
        if return_sample_info:
            # Concatenate all collected samples
            all_images = torch.cat(sample_images, dim=0)[:num_samples]
            all_labels = torch.cat(sample_labels, dim=0)[:num_samples]
            sample_info = {
                'images': all_images,
                'labels': all_labels,
                'num_samples': min(total, num_samples),
                'layer': layer
            }

        return P, sample_info


class Editor:
    """
    Main class for AlphaEdit weight editing in ViT models.
    Supports multiple MedMNIST datasets with dataset-specific checkpoint naming.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        hparams: AlphaEditHyperParams = None,
        log_dir: str = "logs",
        dataset_name: str = "pathmnist"
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)
        self.hparams = hparams if hparams else AlphaEditHyperParams()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name

        # Components
        self.k_collector = KCollector(model, device, self.hparams)
        self.z_computer = ZComputer(model, device, self.hparams)
        self.projector = NullSpaceProjector(self.hparams.nullspace_threshold)

        # Cache for sequential edits
        self.cache_KKT = {}  # layer -> K_prev @ K_prev^T
        self.P = {}  # layer -> projection matrix

        # Edit history
        self.edit_history = []

        # Projection samples (FT-Train samples used for P matrix construction)
        self.projection_samples = None  # Will store {images, labels, num_samples}
    
    def precompute_projection(
        self,
        stats_loader: torch.utils.data.DataLoader,
        num_samples: int = 1000,
        track_samples: bool = True
    ):
        """
        Precompute null-space projection matrices for all target layers.

        IMPORTANT (4-Set Protocol):
        - stats_loader MUST be FT-Train loader (90% of official train)
        - This computes covariance K^T K for knowledge preservation
        - NEVER use Edit-Discovery loader here!

        Args:
            stats_loader: DataLoader for FT-Train set (for covariance computation)
            num_samples: Number of samples to collect for statistics
            track_samples: If True, store the samples used for later evaluation
        """
        print("\nPrecomputing projection matrices (using FT-Train for stats)...")

        # Track samples from the first layer only (they're the same for all layers)
        first_layer = True

        for layer in tqdm(self.hparams.layers, desc="Layers"):
            P, sample_info = self.projector.compute_from_random_samples(
                model=self.model,
                dataloader=stats_loader,
                layer=layer,
                hparams=self.hparams,
                device=self.device,
                num_samples=num_samples,
                return_sample_info=(track_samples and first_layer)
            )
            self.P[layer] = P

            # Store sample info from first layer
            if track_samples and first_layer and sample_info is not None:
                self.projection_samples = sample_info
                print(f"\n  Tracked {sample_info['num_samples']} FT-Train samples for projection matrix")
                first_layer = False

            # Initialize cache
            hidden_size = P.shape[0]
            self.cache_KKT[layer] = torch.zeros(
                (hidden_size, hidden_size),
                device=self.device,
                dtype=P.dtype
            )

        print(f"Projection matrices computed for layers: {self.hparams.layers}")

    def get_projection_samples(self) -> Optional[Dict[str, Any]]:
        """
        Get the FT-Train samples used for projection matrix computation.

        Returns:
            Dictionary with 'images', 'labels', 'num_samples' or None if not tracked
        """
        return self.projection_samples

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
        
        # Store K vectors for cache update after all layers are edited
        layer_K_vectors = {}

        for i, layer in enumerate(self.hparams.layers):
            print(f"\nEditing layer {layer} ({i+1}/{num_layers})...")

            # Get K vectors: (hidden_size, B)
            K = self.k_collector.compute_ks(images, layer)
            layer_K_vectors[layer] = K  # Store for cache update later

            # Get current output: (B, hidden_size)
            current_z = self.k_collector.compute_current_output(images, layer)

            # Target output
            target_z = target_zs[layer]

            # Residual: R = V_target - W @ K
            # In our case: R = target_z - current_z
            # R shape: (hidden_size, B) for matrix operations
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

            # ================================================================
            # AlphaEdit Update Formula (CORRECTED to match original paper):
            #
            # Δ = [P @ (K @ K^T + cache_c) + λI]^{-1} @ P @ K @ R^T
            #
            # Where:
            #   - P: null-space projection matrix
            #   - K: key vectors (hidden_size, B)
            #   - R: residual vectors (hidden_size, B)
            #   - cache_c: accumulated K @ K^T from previous edits
            #   - λ: L2 regularization
            # ================================================================

            KKT = K @ K.T  # (hidden_size, hidden_size)

            # A = P @ (K @ K^T + cache_c) + λI
            # NOTE: P is multiplied OUTSIDE (KKT + cache_c), not inside!
            A = P @ (KKT + cache_KKT) + self.hparams.L2 * torch.eye(
                K.shape[0], device=self.device, dtype=K.dtype
            )

            # B = P @ K @ R^T
            B = P @ K @ R.T  # (hidden_size, hidden_size)

            # Solve: A @ Δ = B  =>  Δ = A^{-1} @ B
            try:
                delta = torch.linalg.solve(A, B)  # (hidden_size, hidden_size)
            except Exception as e:
                print(f"Warning: Solve failed, using pseudoinverse: {e}")
                A_inv = torch.linalg.pinv(A)
                delta = A_inv @ B

            # Apply update to model weights
            module_name = self.hparams.rewrite_module_tmp.format(layer)
            target_module = self._get_module(self.model, module_name)

            # The weight matrix shape is (out_features, in_features)
            # delta is (in_features, in_features) after our fix
            # For vit.encoder.layer.{}.output.dense:
            #   weight shape: (768, 3072)
            #   delta shape: (3072, 3072)
            with torch.no_grad():
                old_weight = target_module.weight.data.clone()

                # Match delta shape to weight shape
                # Following AlphaEdit's upd_matrix_match_shape logic
                if delta.shape == old_weight.shape:
                    # Direct match
                    upd_matrix = delta
                elif delta.T.shape == old_weight.shape:
                    # Transposed match (for GPT-2/GPT-J style weights)
                    upd_matrix = delta.T
                else:
                    raise ValueError(
                        f"Update matrix shape {delta.shape} does not match "
                        f"weight shape {old_weight.shape} or its transpose. "
                        f"K shape: {K.shape}, module: {module_name}"
                    )

                target_module.weight.data = old_weight + upd_matrix
                update_norm = torch.norm(upd_matrix).item()

            # Record (but don't update cache yet!)
            edit_info['layer_updates'][layer] = {
                'update_norm': update_norm,
                'K_norm': K.norm().item(),
                'R_norm': R.norm().item()
            }

            print(f"  Update norm: {update_norm:.6f}")

        # ================================================================
        # Update cache AFTER all layers have been edited
        # This matches the original AlphaEdit implementation
        # ================================================================
        for layer in self.hparams.layers:
            K = layer_K_vectors[layer]
            KKT = K @ K.T
            cache_KKT = self.cache_KKT.get(layer, torch.zeros_like(KKT))
            self.cache_KKT[layer] = cache_KKT + KKT
        
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
            filepath = Path("checkpoints") / f"vit_{self.dataset_name}_edited.pt"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'edit_history': self.edit_history,
            'hparams': vars(self.hparams),
            'dataset_name': self.dataset_name
        }, filepath)

        print(f"Edited model saved to: {filepath}")
        return str(filepath)
    
    @staticmethod
    def _get_module(model, name):
        for part in name.split('.'):
            model = getattr(model, part)
        return model


class HeadEditor:
    """
    Head Editing: Modifies only the classification head to correct misclassifications.

    A simpler and faster alternative to AlphaEdit that:
    - Only modifies model.classifier (768 -> num_classes linear layer)
    - Uses EWC regularization to prevent catastrophic forgetting
    - Supports gradient-based optimization or closed-form solution
    - Supports multiple MedMNIST datasets with dataset-specific checkpoint naming
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        hparams: HeadEditHyperParams = None,
        log_dir: str = "logs",
        dataset_name: str = "pathmnist"
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)
        self.hparams = hparams if hparams else HeadEditHyperParams()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name

        # Fisher information for EWC
        self.fisher_weight = None
        self.fisher_bias = None
        self.original_weight = None
        self.original_bias = None

        # Edit history
        self.edit_history = []

    def compute_fisher_information(
        self,
        stats_loader: torch.utils.data.DataLoader,
        num_samples: int = None
    ):
        """
        Compute diagonal Fisher information matrix for classifier weights.
        Used for EWC regularization to preserve knowledge.

        IMPORTANT (4-Set Protocol):
        - stats_loader MUST be FT-Train loader (90% of official train)
        - This computes Fisher information for knowledge preservation
        - NEVER use Edit-Discovery loader here!

        Args:
            stats_loader: DataLoader for FT-Train set (for Fisher computation)
            num_samples: Number of samples to use (default: hparams.fisher_samples)
        """
        if num_samples is None:
            num_samples = self.hparams.fisher_samples

        print(f"\nComputing Fisher information from {num_samples} samples (FT-Train)...")
        self.model.eval()

        # Store original weights
        self.original_weight = self.model.classifier.weight.data.clone()
        self.original_bias = self.model.classifier.bias.data.clone()

        # Initialize Fisher accumulators
        self.fisher_weight = torch.zeros_like(self.model.classifier.weight)
        self.fisher_bias = torch.zeros_like(self.model.classifier.bias)

        count = 0
        for images, labels in tqdm(stats_loader, desc="Computing Fisher"):
            if count >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()
            outputs = self.model(images)

            # Use log-likelihood
            log_probs = torch.log_softmax(outputs.logits, dim=1)
            loss = nn.functional.nll_loss(log_probs, labels)
            loss.backward()

            # Accumulate squared gradients
            if self.model.classifier.weight.grad is not None:
                self.fisher_weight += self.model.classifier.weight.grad.data ** 2
            if self.model.classifier.bias.grad is not None:
                self.fisher_bias += self.model.classifier.bias.grad.data ** 2

            count += images.shape[0]

        # Normalize
        self.fisher_weight /= count
        self.fisher_bias /= count

        print(f"Fisher information computed (weight norm: {self.fisher_weight.norm():.4f})")

    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if self.fisher_weight is None or self.original_weight is None:
            return torch.tensor(0.0, device=self.device)

        weight_diff = self.model.classifier.weight - self.original_weight
        bias_diff = self.model.classifier.bias - self.original_bias

        ewc_loss = (
            (self.fisher_weight * weight_diff ** 2).sum() +
            (self.fisher_bias * bias_diff ** 2).sum()
        )

        return ewc_loss

    def apply_edit(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        sample_indices: List[int] = None
    ) -> Dict[str, Any]:
        """
        Apply gradient-based head editing with EWC regularization.

        Args:
            images: Misclassified images (B, C, H, W)
            true_labels: Correct labels (B,)
            sample_indices: Optional indices for logging

        Returns:
            Dictionary with edit information
        """
        if self.hparams.closed_form:
            return self.apply_closed_form_edit(images, true_labels, sample_indices)

        images = images.to(self.device)
        true_labels = true_labels.to(self.device)

        if sample_indices is None:
            sample_indices = list(range(len(images)))

        # Store pre-edit state
        pre_weight = self.model.classifier.weight.data.clone()
        pre_bias = self.model.classifier.bias.data.clone()

        # Freeze all parameters except classifier
        for name, param in self.model.named_parameters():
            param.requires_grad = 'classifier' in name

        # Get CLS token representations (frozen backbone)
        self.model.eval()
        with torch.no_grad():
            # Access ViT backbone
            vit_outputs = self.model.vit(images)
            cls_representations = vit_outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # Pre-edit predictions
        with torch.no_grad():
            pre_logits = self.model.classifier(cls_representations)
            pre_preds = pre_logits.argmax(dim=1)

        # Optimizer for classifier only
        optimizer = torch.optim.Adam(
            self.model.classifier.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # Optimization loop
        losses = []
        for step in range(self.hparams.num_steps):
            optimizer.zero_grad()

            # Forward through classifier
            logits = self.model.classifier(cls_representations)

            # Classification loss
            ce_loss = nn.CrossEntropyLoss()(logits, true_labels)

            # EWC regularization
            ewc_loss = self._compute_ewc_loss()

            # Total loss
            loss = ce_loss + self.hparams.ewc_lambda * ewc_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Restore all parameters to trainable
        for param in self.model.parameters():
            param.requires_grad = True

        # Post-edit predictions
        with torch.no_grad():
            post_logits = self.model.classifier(cls_representations)
            post_preds = post_logits.argmax(dim=1)

        # Compute update statistics
        weight_change = (self.model.classifier.weight.data - pre_weight).norm().item()
        bias_change = (self.model.classifier.bias.data - pre_bias).norm().item()

        # Check success
        corrected = (post_preds == true_labels).sum().item()
        total = len(true_labels)

        edit_info = {
            'sample_indices': sample_indices,
            'num_samples': total,
            'corrected': corrected,
            'success_rate': corrected / total,
            'weight_change_norm': weight_change,
            'bias_change_norm': bias_change,
            'final_loss': losses[-1] if losses else 0,
            'pre_predictions': pre_preds.cpu().tolist(),
            'post_predictions': post_preds.cpu().tolist(),
            'true_labels': true_labels.cpu().tolist(),
            'method': 'gradient'
        }

        self.edit_history.append(edit_info)

        print(f"  Corrected: {corrected}/{total} ({100*corrected/total:.1f}%)")
        print(f"  Weight change: {weight_change:.6f}, Bias change: {bias_change:.6f}")

        return edit_info

    def apply_closed_form_edit(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        sample_indices: List[int] = None
    ) -> Dict[str, Any]:
        """
        Apply closed-form least-squares head editing.

        Solves: W_new = argmin ||W @ X - Y||^2 + lambda * ||W - W_old||^2
        """
        images = images.to(self.device)
        true_labels = true_labels.to(self.device)

        if sample_indices is None:
            sample_indices = list(range(len(images)))

        # Store pre-edit state
        pre_weight = self.model.classifier.weight.data.clone()
        pre_bias = self.model.classifier.bias.data.clone()

        # Get CLS token representations
        self.model.eval()
        with torch.no_grad():
            vit_outputs = self.model.vit(images)
            cls_representations = vit_outputs.last_hidden_state[:, 0, :]  # (B, 768)

            # Pre-edit predictions
            pre_logits = self.model.classifier(cls_representations)
            pre_preds = pre_logits.argmax(dim=1)

        # Prepare matrices
        X = cls_representations.T  # (768, B)
        num_classes = self.model.classifier.weight.shape[0]
        Y = nn.functional.one_hot(true_labels, num_classes=num_classes).float().T  # (9, B)

        W_old = self.model.classifier.weight.data  # (9, 768)

        # Regularized least squares solution
        # W_new = (Y @ X.T + lambda * W_old) @ (X @ X.T + lambda * I)^{-1}
        lambda_reg = self.hparams.reg_lambda

        XXT = X @ X.T  # (768, 768)
        YXT = Y @ X.T  # (9, 768)

        A = XXT + lambda_reg * torch.eye(X.shape[0], device=self.device, dtype=X.dtype)
        B = YXT + lambda_reg * W_old

        try:
            W_new = torch.linalg.solve(A.T, B.T).T
        except Exception as e:
            print(f"Warning: Closed-form solve failed, using pseudoinverse: {e}")
            A_inv = torch.linalg.pinv(A)
            W_new = B @ A_inv

        # Apply update
        with torch.no_grad():
            self.model.classifier.weight.data = W_new

        # Post-edit predictions
        with torch.no_grad():
            post_logits = self.model.classifier(cls_representations)
            post_preds = post_logits.argmax(dim=1)

        # Compute statistics
        weight_change = (self.model.classifier.weight.data - pre_weight).norm().item()

        corrected = (post_preds == true_labels).sum().item()
        total = len(true_labels)

        edit_info = {
            'sample_indices': sample_indices,
            'num_samples': total,
            'corrected': corrected,
            'success_rate': corrected / total,
            'weight_change_norm': weight_change,
            'bias_change_norm': 0.0,  # Bias not modified in closed-form
            'final_loss': 0,
            'pre_predictions': pre_preds.cpu().tolist(),
            'post_predictions': post_preds.cpu().tolist(),
            'true_labels': true_labels.cpu().tolist(),
            'method': 'closed_form'
        }

        self.edit_history.append(edit_info)

        print(f"  Corrected: {corrected}/{total} ({100*corrected/total:.1f}%)")
        print(f"  Weight change: {weight_change:.6f}")

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
        dataset = dataloader.dataset

        # Collect all samples
        images_list = []
        labels_list = []

        for idx in indices:
            image, label = dataset[idx]
            images_list.append(image)
            labels_list.append(label)

        images = torch.stack(images_list)
        labels = torch.tensor(labels_list)

        # Apply edit to all samples at once
        result = self.apply_edit(
            images=images,
            true_labels=labels,
            sample_indices=indices
        )

        return [result]

    def export_edit_log(self, filename: str = "head_edit_log.csv") -> str:
        """Export edit history to CSV."""
        if not self.edit_history:
            print("No edit history to export.")
            return None

        rows = []
        for edit in self.edit_history:
            rows.append({
                'sample_indices': str(edit['sample_indices']),
                'num_samples': edit['num_samples'],
                'corrected': edit['corrected'],
                'success_rate': edit['success_rate'],
                'weight_change_norm': edit['weight_change_norm'],
                'bias_change_norm': edit['bias_change_norm'],
                'method': edit['method']
            })

        df = pd.DataFrame(rows)
        csv_path = self.log_dir / filename
        df.to_csv(csv_path, index=False)

        print(f"Head edit log exported to: {csv_path}")
        return str(csv_path)

    def save_edited_model(self, filepath: str = None) -> str:
        """Save the edited model."""
        if filepath is None:
            filepath = Path("checkpoints") / f"vit_{self.dataset_name}_head_edited.pt"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'edit_history': self.edit_history,
            'hparams': vars(self.hparams),
            'dataset_name': self.dataset_name,
            'fisher_weight': self.fisher_weight,
            'fisher_bias': self.fisher_bias,
            'original_weight': self.original_weight,
            'original_bias': self.original_bias
        }, filepath)

        print(f"Head-edited model saved to: {filepath}")
        return str(filepath)


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
    
    print("\n[OK] Editor test complete!")


if __name__ == "__main__":
    main()
