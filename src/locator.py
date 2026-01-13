"""
Locator Module for ViT Model Editing Pipeline (AlphaEdit Causal Tracing)
=========================================================================
Identifies important layers using causal tracing methodology from AlphaEdit.

Key methodology:
1. Corrupt input by adding noise to patch embeddings
2. Run corrupted forward pass (prediction degrades)
3. Restore clean activations at specific (patch, layer) positions
4. Measure prediction recovery to determine importance

Adapted from AlphaEdit's causal_trace.py for Vision Transformers.
"""

import os
import contextlib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from transformers import ViTForImageClassification


# =============================================================================
# Utility Classes (adapted from AlphaEdit's nethook.py)
# =============================================================================

class StopForward(Exception):
    """Exception to stop forward pass early."""
    pass


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v, clone, detach, retain_grad) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v, clone, detach, retain_grad) for v in x])
    else:
        return x


def get_module(model: nn.Module, name: str) -> nn.Module:
    """Finds the named module within the given model."""
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(f"Module {name} not found")


class Trace(contextlib.AbstractContextManager):
    """
    Single-layer activation tracer with edit capability.
    Adapted from AlphaEdit's nethook.py.
    """

    def __init__(
        self,
        module: nn.Module,
        layer: str = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Callable = None,
        stop: bool = False
    ):
        self.layer = layer
        self.retain_output = retain_output
        self.retain_input = retain_input
        self.clone = clone
        self.detach = detach
        self.retain_grad = retain_grad
        self.edit_output = edit_output
        self.stop = stop

        self.output = None
        self.input = None

        if layer is not None:
            module = get_module(module, layer)

        def retain_hook(m, inputs, output):
            if self.retain_input:
                self.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone, detach=detach, retain_grad=False
                )
            if self.edit_output is not None:
                output = invoke_with_optional_args(
                    self.edit_output, output=output, layer=self.layer
                )
            if self.retain_output:
                self.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                if retain_grad:
                    output = recursive_copy(self.output, clone=True, detach=False)
            if self.stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and type is not None and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    Multi-layer activation tracer with edit capability.
    Adapted from AlphaEdit's nethook.py.
    """

    def __init__(
        self,
        module: nn.Module,
        layers: List[str],
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Callable = None,
        stop: bool = False
    ):
        super().__init__()
        self.stop = stop

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and type is not None and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()


def invoke_with_optional_args(fn, *args, **kwargs):
    """Invokes a function with only the arguments it accepts."""
    import inspect
    argspec = inspect.getfullargspec(fn)

    # Simple case: if function accepts **kwargs, pass everything
    if argspec.varkw is not None:
        return fn(*args, **kwargs)

    # Filter kwargs to only include accepted parameters
    accepted_kwargs = {}
    for k, v in kwargs.items():
        if k in argspec.args or k in argspec.kwonlyargs:
            accepted_kwargs[k] = v

    return fn(*args, **accepted_kwargs)


# =============================================================================
# ViT-specific Layer Naming
# =============================================================================

def layername(model: nn.Module, num: int, kind: str = None) -> str:
    """
    Get the layer name for a ViT model.

    Args:
        model: ViT model
        num: Layer number (0-11 for ViT-B)
        kind: 'embed' for embedding layer, 'attn' for attention,
              'mlp' for MLP, None for full layer

    Returns:
        Layer name string
    """
    if kind == "embed":
        return "vit.embeddings"

    base = f"vit.encoder.layer.{num}"

    if kind is None:
        return base
    elif kind == "attn":
        return f"{base}.attention"
    elif kind == "mlp":
        return f"{base}.output"  # In ViT, output.dense is the MLP output
    else:
        return f"{base}.{kind}"


# =============================================================================
# Causal Tracing Core Functions (adapted from AlphaEdit)
# =============================================================================

def trace_with_patch(
    model: nn.Module,
    inp: Dict[str, torch.Tensor],
    states_to_patch: List[Tuple[int, str]],
    answers_t: torch.Tensor,
    tokens_to_mix: Tuple[int, int],
    noise: float = 0.1,
    uniform_noise: bool = False,
    replace: bool = False,
    trace_layers: List[str] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Runs a single causal trace for ViT models.

    Given a model and a batch input where batch size >= 2, runs inference
    while corrupting runs [1...n] and optionally restoring hidden states
    from the uncorrupted run [0].

    Args:
        model: ViT model
        inp: Input dict with 'pixel_values' of shape (B, C, H, W)
        states_to_patch: List of (token_index, layername) pairs to restore
        answers_t: Target class index for probability computation
        tokens_to_mix: Range of tokens to corrupt (begin, end) in the
                       sequence (including CLS token at position 0)
        noise: Noise level (std dev for Gaussian, or range for uniform)
        uniform_noise: If True, use uniform noise instead of Gaussian
        replace: If True, replace with noise; if False, add noise
        trace_layers: Optional list of layers to trace outputs from

    Returns:
        probs: Probability of target class after restoration
        all_traced: (optional) Traced activations if trace_layers provided
    """
    rs = np.random.RandomState(1)  # For reproducibility
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define noise function
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(output, layer):
        """
        Patching function that:
        1. Adds noise to embeddings for corrupted runs
        2. Restores clean activations at specified positions
        """
        if layer == embed_layername:
            # Corrupt the patch embeddings
            # output shape: (B, num_patches+1, hidden_dim) = (B, 197, 768)
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                # Generate noise for corrupted batch elements [1:]
                noise_data = noise_fn(
                    torch.from_numpy(
                        prng(output.shape[0] - 1, e - b, output.shape[2])
                    ).float()
                ).to(output.device)

                if replace:
                    output[1:, b:e] = noise_data
                else:
                    output[1:, b:e] = output[1:, b:e] + noise_data
            return output

        if layer not in patch_spec:
            return output

        # Restore clean activations from run [0] to corrupted runs [1:]
        h = untuple(output)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return output

    # Prepare layers to trace
    additional_layers = [] if trace_layers is None else trace_layers
    all_layers = [embed_layername] + list(patch_spec.keys()) + additional_layers

    # Run model with patching
    with torch.no_grad(), TraceDict(
        model,
        all_layers,
        edit_output=patch_rep
    ) as td:
        outputs = model(**inp)

    # Compute probability of target class
    # Average over corrupted runs (exclude clean run [0])
    probs = torch.softmax(outputs.logits[1:], dim=1).mean(dim=0)[answers_t]

    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers],
            dim=2
        )
        return probs, all_traced

    return probs


def trace_important_states(
    model: nn.Module,
    num_layers: int,
    inp: Dict[str, torch.Tensor],
    e_range: Tuple[int, int],
    answer_t: torch.Tensor,
    noise: float = 0.1,
    uniform_noise: bool = False,
    replace: bool = False,
    token_range: List[int] = None
) -> torch.Tensor:
    """
    Traces importance of each (token, layer) combination.

    For each token position and each layer, restores the clean activation
    and measures how much the prediction probability recovers.

    Args:
        model: ViT model
        num_layers: Number of transformer layers
        inp: Input dict with pixel_values
        e_range: Range of tokens to corrupt (e.g., image patches)
        answer_t: Target class index
        noise: Noise level
        uniform_noise: Use uniform instead of Gaussian noise
        replace: Replace vs add noise
        token_range: Specific tokens to analyze (None = all)

    Returns:
        Tensor of shape (num_tokens, num_layers) with recovery scores
    """
    ntoks = 197  # CLS + 196 patches for ViT-B/16
    table = []

    if token_range is None:
        token_range = range(ntoks)

    for tnum in tqdm(token_range, desc="Tracing tokens", leave=False):
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace
            )
            row.append(r)
        table.append(torch.stack(row))

    return torch.stack(table)


def trace_important_window(
    model: nn.Module,
    num_layers: int,
    inp: Dict[str, torch.Tensor],
    e_range: Tuple[int, int],
    answer_t: torch.Tensor,
    kind: str,
    window: int = 10,
    noise: float = 0.1,
    uniform_noise: bool = False,
    replace: bool = False,
    token_range: List[int] = None
) -> torch.Tensor:
    """
    Traces importance using a sliding window of layers.

    Instead of restoring a single layer, restores a window of layers
    centered at each position. Useful for analyzing MLP or attention
    specifically.

    Args:
        model: ViT model
        num_layers: Number of transformer layers
        inp: Input dict
        e_range: Range of tokens to corrupt
        answer_t: Target class index
        kind: 'attn' or 'mlp' to specify component type
        window: Size of layer window
        noise: Noise level
        uniform_noise: Use uniform noise
        replace: Replace vs add noise
        token_range: Specific tokens to analyze

    Returns:
        Tensor of shape (num_tokens, num_layers) with recovery scores
    """
    ntoks = 197
    table = []

    if token_range is None:
        token_range = range(ntoks)

    for tnum in tqdm(token_range, desc=f"Tracing {kind}", leave=False):
        row = []
        for layer in range(num_layers):
            # Create window of layers centered at current layer
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2),
                    min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace
            )
            row.append(r)
        table.append(torch.stack(row))

    return torch.stack(table)


# =============================================================================
# Noise Level Estimation
# =============================================================================

def collect_embedding_std(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 100
) -> float:
    """
    Estimate the standard deviation of patch embeddings.
    Used to calibrate noise level for causal tracing.

    Args:
        model: ViT model
        dataloader: DataLoader with images
        device: Computation device
        num_samples: Number of samples to use

    Returns:
        Standard deviation of embeddings
    """
    model.eval()
    all_embeds = []
    count = 0

    embed_layer = layername(model, 0, "embed")

    for images, _ in dataloader:
        if count >= num_samples:
            break

        images = images.to(device)

        with Trace(model, embed_layer) as t:
            with torch.no_grad():
                _ = model(pixel_values=images)
                all_embeds.append(t.output.cpu())

        count += images.shape[0]

    all_embeds = torch.cat(all_embeds, dim=0)
    noise_level = all_embeds.std().item()

    return noise_level


# =============================================================================
# Main Locator Class
# =============================================================================

class CausalTracer:
    """
    Causal tracing analysis for ViT models.
    Identifies important (token, layer) combinations for predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        num_layers: int = 12
    ):
        """
        Args:
            model: ViT model
            device: Computation device
            num_layers: Number of transformer layers (12 for ViT-B)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.device = device
        self.num_layers = num_layers
        self.model.eval()

    def calculate_hidden_flow(
        self,
        image: torch.Tensor,
        true_label: int,
        samples: int = 10,
        noise: float = 0.1,
        uniform_noise: bool = False,
        replace: bool = False,
        window: int = 10,
        kind: str = None,
        token_range: List[int] = None
    ) -> Dict[str, Any]:
        """
        Run causal tracing over all token/layer combinations.

        Args:
            image: Input image tensor (C, H, W)
            true_label: Target class index
            samples: Number of corrupted samples to average
            noise: Noise level (or 'auto' to estimate)
            uniform_noise: Use uniform noise
            replace: Replace vs add noise
            window: Window size for component-specific tracing
            kind: None for full layer, 'attn' or 'mlp' for components
            token_range: Specific tokens to trace (None = CLS only for efficiency)

        Returns:
            Dictionary with tracing results
        """
        # Prepare input: clean sample + corrupted copies
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        images = image.repeat(samples + 1, 1, 1, 1)
        inp = {"pixel_values": images}

        # Get base prediction
        with torch.no_grad():
            outputs = self.model(pixel_values=image)
            base_probs = torch.softmax(outputs.logits, dim=1)
            base_score = base_probs[0, true_label].item()
            predicted_class = outputs.logits.argmax(dim=1).item()

        # Check if prediction is correct
        if predicted_class != true_label:
            # Still run tracing, but note the prediction was wrong
            pass

        answer_t = true_label

        # Define corruption range
        # For ViT: position 0 is CLS, positions 1-196 are patches
        # We corrupt all patch tokens (similar to corrupting subject in text)
        e_range = (1, 197)  # Corrupt all image patches

        # Get corrupted (low) score - no restoration
        low_score = trace_with_patch(
            self.model, inp, [], answer_t, e_range,
            noise=noise, uniform_noise=uniform_noise
        ).item()

        # Run tracing
        if token_range is None:
            # Default: only trace CLS token (most important for classification)
            token_range = [0]

        if not kind:
            differences = trace_important_states(
                self.model,
                self.num_layers,
                inp,
                e_range,
                answer_t,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                token_range=token_range
            )
        else:
            differences = trace_important_window(
                self.model,
                self.num_layers,
                inp,
                e_range,
                answer_t,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                window=window,
                kind=kind,
                token_range=token_range
            )

        differences = differences.detach().cpu()

        return {
            'scores': differences,
            'low_score': low_score,
            'high_score': base_score,
            'answer': true_label,
            'predicted': predicted_class,
            'correct_prediction': predicted_class == true_label,
            'window': window,
            'kind': kind or '',
            'token_range': token_range
        }


class Locator:
    """
    Main class for layer localization using causal tracing.
    Implements AlphaEdit-style analysis for ViT models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        log_dir: str = "logs",
        num_layers: int = 12
    ):
        """
        Args:
            model: Fine-tuned ViT model
            device: Computation device
            log_dir: Directory for saving results
            num_layers: Number of transformer layers
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)
        self.model.eval()

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.num_layers = num_layers
        self.tracer = CausalTracer(model, device, num_layers)

        # Results storage
        self.trace_results = []
        self.noise_level = None

    def estimate_noise_level(
        self,
        dataloader: torch.utils.data.DataLoader,
        factor: float = 3.0,
        num_samples: int = 100
    ) -> float:
        """
        Estimate appropriate noise level from data.

        Args:
            dataloader: DataLoader with training/validation images
            factor: Multiplier for embedding std (default 3.0 as in AlphaEdit)
            num_samples: Number of samples for estimation

        Returns:
            Calibrated noise level
        """
        std = collect_embedding_std(
            self.model, dataloader, self.device, num_samples
        )
        self.noise_level = factor * std
        print(f"Estimated noise level: {self.noise_level:.4f} (std={std:.4f}, factor={factor})")
        return self.noise_level

    def analyze_sample(
        self,
        image: torch.Tensor,
        true_label: int,
        predicted_label: int,
        sample_idx: int,
        samples: int = 10,
        noise: float = None,
        analyze_components: bool = False,
        token_range: List[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single sample using causal tracing.

        Args:
            image: Input image tensor (C, H, W)
            true_label: Ground truth label
            predicted_label: Model's prediction
            sample_idx: Sample index for logging
            samples: Number of corrupted samples for averaging
            noise: Noise level (uses estimated if None)
            analyze_components: If True, also analyze attention and MLP separately
            token_range: Token positions to analyze (None = CLS only)

        Returns:
            Dictionary with analysis results
        """
        if noise is None:
            noise = self.noise_level if self.noise_level else 0.1

        print(f"\nAnalyzing sample {sample_idx}: true={true_label}, pred={predicted_label}")

        # Full layer analysis
        result = self.tracer.calculate_hidden_flow(
            image=image,
            true_label=true_label,
            samples=samples,
            noise=noise,
            token_range=token_range
        )

        # Extract layer importance scores
        scores = result['scores']  # Shape: (num_tokens, num_layers)

        # For CLS token analysis (most important for classification)
        if 0 in (token_range or [0]):
            cls_idx = (token_range or [0]).index(0)
            layer_scores = scores[cls_idx].numpy()
        else:
            # Average over analyzed tokens
            layer_scores = scores.mean(dim=0).numpy()

        # Normalize: (score - low_score) / (high_score - low_score)
        low = result['low_score']
        high = result['high_score']
        if high > low:
            normalized_scores = (layer_scores - low) / (high - low)
        else:
            normalized_scores = layer_scores

        # Create result record
        record = {
            'sample_idx': sample_idx,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct_prediction': result['correct_prediction'],
            'high_score': high,
            'low_score': low,
        }

        # Add per-layer scores
        for i in range(self.num_layers):
            record[f'layer_{i}_raw'] = float(layer_scores[i])
            record[f'layer_{i}_normalized'] = float(normalized_scores[i])

        # Find most important layers
        sorted_layers = sorted(
            enumerate(normalized_scores),
            key=lambda x: x[1],
            reverse=True
        )
        record['top_layer'] = sorted_layers[0][0]
        record['top_score'] = sorted_layers[0][1]
        record['top_3_layers'] = [l[0] for l in sorted_layers[:3]]

        # Optional: analyze attention and MLP separately
        if analyze_components:
            for kind in ['attn', 'mlp']:
                comp_result = self.tracer.calculate_hidden_flow(
                    image=image,
                    true_label=true_label,
                    samples=samples,
                    noise=noise,
                    kind=kind,
                    token_range=token_range
                )
                comp_scores = comp_result['scores']
                if 0 in (token_range or [0]):
                    cls_idx = (token_range or [0]).index(0)
                    comp_layer_scores = comp_scores[cls_idx].numpy()
                else:
                    comp_layer_scores = comp_scores.mean(dim=0).numpy()

                for i in range(self.num_layers):
                    record[f'layer_{i}_{kind}'] = float(comp_layer_scores[i])

        self.trace_results.append(record)

        return record

    def analyze_batch(
        self,
        images: List[torch.Tensor],
        true_labels: List[int],
        predicted_labels: List[int],
        sample_indices: List[int],
        samples: int = 10,
        noise: float = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple samples.
        """
        results = []

        for img, true_label, pred_label, idx in tqdm(
            zip(images, true_labels, predicted_labels, sample_indices),
            total=len(images),
            desc="Causal tracing"
        ):
            result = self.analyze_sample(
                image=img,
                true_label=true_label,
                predicted_label=pred_label,
                sample_idx=idx,
                samples=samples,
                noise=noise
            )
            results.append(result)

        return results

    def get_layer_statistics(self) -> pd.DataFrame:
        """
        Compute aggregate statistics across all analyzed samples.
        """
        if not self.trace_results:
            return pd.DataFrame()

        df = pd.DataFrame(self.trace_results)

        # Compute mean normalized score per layer
        stats = []
        for i in range(self.num_layers):
            col = f'layer_{i}_normalized'
            if col in df.columns:
                stats.append({
                    'layer_index': i,
                    'layer_name': f'vit.encoder.layer.{i}',
                    'mean_score': df[col].mean(),
                    'std_score': df[col].std(),
                    'max_score': df[col].max(),
                    'min_score': df[col].min()
                })

        return pd.DataFrame(stats).sort_values('mean_score', ascending=False)

    def get_top_layers(self, n: int = 3) -> List[int]:
        """
        Get the top N most important layer indices based on causal tracing.
        """
        stats = self.get_layer_statistics()
        if stats.empty:
            return list(range(self.num_layers - n, self.num_layers))

        return stats.head(n)['layer_index'].tolist()

    def export_results(self, filename: str = "causal_trace_results.csv") -> str:
        """
        Export all analysis results to CSV.
        """
        if not self.trace_results:
            print("No results to export.")
            return None

        df = pd.DataFrame(self.trace_results)
        csv_path = self.log_dir / filename
        df.to_csv(csv_path, index=False)
        print(f"Results exported to: {csv_path}")

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
        print("Causal Tracing - Layer Importance Summary")
        print("=" * 60)
        print("(Higher score = restoring this layer recovers prediction more)")
        print()

        max_score = stats['mean_score'].max()
        min_score = stats['mean_score'].min()
        score_range = max_score - min_score if max_score > min_score else 1

        for _, row in stats.iterrows():
            layer_idx = int(row['layer_index'])
            score = row['mean_score']
            normalized = (score - min_score) / score_range
            bar_len = int(40 * normalized)
            bar = "█" * bar_len
            print(f"Layer {layer_idx:2d}: {bar:<40} {score:.4f}")

        print("=" * 60)
        top_layers = self.get_top_layers(3)
        print(f"Recommended layers for editing: {top_layers}")

    def plot_trace_heatmap(
        self,
        result: Dict[str, Any],
        savepath: str = None
    ) -> None:
        """
        Plot a heatmap of causal tracing results.

        Args:
            result: Result from analyze_sample
            savepath: Path to save the plot (None to display)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        scores = result.get('scores')
        if scores is None:
            # Reconstruct from record
            scores = np.array([
                result.get(f'layer_{i}_normalized', 0)
                for i in range(self.num_layers)
            ])
            scores = scores.reshape(1, -1)
        else:
            scores = scores.numpy()

        fig, ax = plt.subplots(figsize=(10, 3), dpi=100)

        h = ax.pcolor(scores, cmap='Purples', vmin=result.get('low_score', 0))
        ax.set_xlabel('Layer')
        ax.set_ylabel('Token')
        ax.set_title(f"Causal Trace - Sample {result.get('sample_idx', '?')}")

        plt.colorbar(h, ax=ax, label='Recovery Score')

        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    """Test causal tracing functionality."""
    print("=" * 70)
    print("ViT Model Editing Pipeline - Causal Tracing Locator (AlphaEdit)")
    print("=" * 70)

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
        num_layers=12
    )

    # Demo with random image
    print("\nRunning causal tracing on dummy image...")
    dummy_image = torch.randn(3, 224, 224)

    result = locator.analyze_sample(
        image=dummy_image,
        true_label=0,
        predicted_label=1,
        sample_idx=0,
        samples=5,  # Reduced for demo
        noise=0.1,
        token_range=[0]  # Only CLS token
    )

    print(f"\nSample result:")
    print(f"  High score (clean): {result['high_score']:.4f}")
    print(f"  Low score (corrupted): {result['low_score']:.4f}")
    print(f"  Top layer: {result['top_layer']}")
    print(f"  Top 3 layers: {result['top_3_layers']}")

    locator.visualize_importance()
    locator.export_results()

    print("\n✓ Causal tracing test complete!")


if __name__ == "__main__":
    main()
