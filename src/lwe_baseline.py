import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluator import evaluate_comparative, evaluate_edit_samples


def _default_edit_layers(num_layers: int, width: int = 3) -> List[int]:
    if num_layers < width:
        return list(range(num_layers))
    start = num_layers - width
    return list(range(start, num_layers))


def _mlp_param_names(layer_idx: int) -> Tuple[str, str]:
    fc2 = f"vit.encoder.layer.{layer_idx}.output.dense.weight"
    fc1 = f"vit.encoder.layer.{layer_idx}.intermediate.dense.weight"
    return fc2, fc1


def _collect_inner_params(layers: List[int]) -> List[str]:
    names: List[str] = []
    for layer in sorted(layers, reverse=True):
        fc2, fc1 = _mlp_param_names(layer)
        names.extend([fc2, fc1])
    return names


class _L2Regularization(nn.Module):
    def __init__(self, source_model: nn.Module, inner_params: List[str]):
        super().__init__()
        self.source_weight = {}
        inner = set(inner_params)
        for name, param in source_model.named_parameters():
            if name in inner:
                self.source_weight[name] = param.detach().clone()

    def forward(self, target_model: nn.Module) -> torch.Tensor:
        loss = 0.0
        for name, param in target_model.named_parameters():
            if name in self.source_weight:
                loss = loss + 0.5 * torch.norm(param - self.source_weight[name], p=2) ** 2
        return loss


class _L1Regularization(nn.Module):
    def __init__(self, source_model: nn.Module, inner_params: List[str]):
        super().__init__()
        self.source_weight = {}
        inner = set(inner_params)
        for name, param in source_model.named_parameters():
            if name in inner:
                self.source_weight[name] = param.detach().clone()

    def forward(self, target_model: nn.Module) -> torch.Tensor:
        loss = 0.0
        for name, param in target_model.named_parameters():
            if name in self.source_weight:
                loss = loss + torch.norm(param - self.source_weight[name], p=1)
        return loss


class LWEFTEditor:
    def __init__(
        self,
        model: nn.Module,
        layers: List[int],
        edit_lr: float = 2e-5,
        max_edit_steps: int = 100,
        cloc: float = 1.0,
        l2_reg: bool = False,
    ):
        self.model = model
        self.layers = layers
        self.edit_lr = edit_lr
        self.max_edit_steps = max_edit_steps
        self.cloc = cloc
        self.inner_params = _collect_inner_params(layers)
        self.reg_loss = _L2Regularization(model, self.inner_params) if l2_reg else _L1Regularization(model, self.inner_params)

    def edit_sample(self, image: torch.Tensor, label: torch.Tensor) -> None:
        edit_model = copy.deepcopy(self.model).eval()
        opt_params = [
            {"params": p, "lr": self.edit_lr}
            for name, p in edit_model.named_parameters()
            if name in self.inner_params
        ]
        if len(opt_params) != len(self.inner_params):
            raise RuntimeError("LWE-FT target parameters not found. Check model architecture and layer indices.")

        opt_1st = torch.optim.SGD(opt_params, lr=self.edit_lr)
        opt = torch.optim.RMSprop(opt_params, lr=self.edit_lr)

        for step in range(self.max_edit_steps):
            output = edit_model(image)
            logits = output.logits
            cls_loss = F.cross_entropy(logits, label)
            acc = logits.argmax(dim=1).eq(label).float().mean().item()
            reg = self.reg_loss(edit_model)
            loss = cls_loss + self.cloc * reg

            if acc == 1.0 and cls_loss.item() <= 0.01:
                break

            loss.backward()
            if step == 0:
                opt_1st.step()
                opt_1st.zero_grad()
            else:
                opt.step()
                opt.zero_grad()

        self.model.load_state_dict(edit_model.state_dict())


class HPRDMaskPredictor(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, num_masks: int = 6, hidden_proj: int = 512):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.num_masks = num_masks
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_proj),
            nn.GELU(),
            nn.Linear(hidden_proj, hidden_proj),
            nn.GELU(),
            nn.Linear(hidden_proj, num_masks * mlp_dim),
            nn.Sigmoid(),
        )

    def forward(self, cls_feature: torch.Tensor) -> torch.Tensor:
        out = self.net(cls_feature)
        return out.view(cls_feature.size(0), self.num_masks, self.mlp_dim)


class LWEHPRDEditor:
    def __init__(
        self,
        model: nn.Module,
        layers: List[int],
        device: torch.device,
        sparsity: float = 0.1,
        edit_lr: float = 2e-5,
        max_edit_steps: int = 100,
    ):
        self.model = model
        self.layers = sorted(layers, reverse=True)
        self.device = device
        self.sparsity = sparsity
        self.edit_lr = edit_lr
        self.max_edit_steps = max_edit_steps

        config = model.config
        hidden_size = int(config.hidden_size)
        mlp_dim = int(config.intermediate_size)
        self.mask_predictor = HPRDMaskPredictor(hidden_size=hidden_size, mlp_dim=mlp_dim).to(device)

        self.inner_params = _collect_inner_params(self.layers)
        self._name_to_idx = {name: idx for idx, name in enumerate(self.inner_params)}

    def _extract_cls_feature(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(images, output_hidden_states=True)
        return outputs.hidden_states[-1][:, 0, :]

    def _build_pseudo_masks(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(images)
        loss = F.cross_entropy(outputs.logits, labels)
        loss.backward()

        pseudo_masks = []
        named_params = dict(self.model.named_parameters())
        for name in self.inner_params:
            grad = named_params[name].grad
            if grad is None:
                raise RuntimeError(f"Missing grad for parameter: {name}")
            if "intermediate.dense.weight" in name:
                importance = grad.abs().mean(dim=1)
            else:
                importance = grad.abs().mean(dim=0)
            pseudo_masks.append(importance)

        stacked = torch.stack(pseudo_masks, dim=0)
        k = max(1, int(self.sparsity * stacked.numel()))
        threshold = torch.topk(stacked.reshape(-1), k=k).values[-1]
        hard = (stacked >= threshold).float()
        return hard

    def train_mask_predictor(
        self,
        error_loader: DataLoader,
        epochs: int = 3,
        lr: float = 1e-4,
        max_batches_per_epoch: Optional[int] = None,
    ):
        self.model.eval()
        self.mask_predictor.train()
        opt = torch.optim.Adam(self.mask_predictor.parameters(), lr=lr)

        for epoch in range(epochs):
            running = 0.0
            num = 0
            pbar = tqdm(error_loader, desc=f"HPRD mask train {epoch+1}/{epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                cls_feat = self._extract_cls_feature(images).detach()
                pseudo_masks = self._build_pseudo_masks(images, labels).unsqueeze(0).repeat(images.size(0), 1, 1)

                pred_masks = self.mask_predictor(cls_feat)
                bce = F.binary_cross_entropy(pred_masks, pseudo_masks)
                sparse_penalty = pred_masks.mean()
                loss = bce + 0.1 * sparse_penalty

                opt.zero_grad()
                loss.backward()
                opt.step()

                running += loss.item()
                num += 1
                pbar.set_postfix({"loss": f"{running / max(1, num):.4f}"})

                if max_batches_per_epoch is not None and (batch_idx + 1) >= max_batches_per_epoch:
                    break

    @torch.no_grad()
    def _predict_hard_masks(self, image: torch.Tensor) -> torch.Tensor:
        self.mask_predictor.eval()
        cls_feat = self._extract_cls_feature(image)
        pred_masks = self.mask_predictor(cls_feat).mean(dim=0)
        k = max(1, int(self.sparsity * pred_masks.numel()))
        threshold = torch.topk(pred_masks.reshape(-1), k=k).values[-1]
        hard = (pred_masks >= threshold).float()
        return hard

    def edit_sample(self, image: torch.Tensor, label: torch.Tensor) -> None:
        edit_model = copy.deepcopy(self.model).eval()
        opt_params = [
            {"params": p, "lr": self.edit_lr}
            for name, p in edit_model.named_parameters()
            if name in self.inner_params
        ]
        if len(opt_params) != len(self.inner_params):
            raise RuntimeError("LWE-HPRD target parameters not found. Check model architecture and layer indices.")

        opt = torch.optim.RMSprop(opt_params, lr=self.edit_lr)
        hard_masks = self._predict_hard_masks(image).to(image.device)

        named_params = dict(edit_model.named_parameters())

        for _ in range(self.max_edit_steps):
            out = edit_model(image)
            cls_loss = F.cross_entropy(out.logits, label)
            acc = out.logits.argmax(dim=1).eq(label).float().mean().item()
            if acc == 1.0 and cls_loss.item() <= 0.01:
                break

            cls_loss.backward()

            for name in self.inner_params:
                param = named_params[name]
                if param.grad is None:
                    continue
                mask_idx = self._name_to_idx[name]
                cur_mask = hard_masks[mask_idx]
                if "intermediate.dense.weight" in name:
                    param.grad.mul_(cur_mask[:, None])
                else:
                    param.grad.mul_(cur_mask[None, :])

            opt.step()
            opt.zero_grad()

        self.model.load_state_dict(edit_model.state_dict())


@dataclass
class LWEBaselineConfig:
    method: str = "ft"
    edit_lr: float = 2e-5
    max_edit_steps: int = 100
    cloc: float = 1.0
    l2_reg: bool = False
    layer: Optional[int] = None
    sparsity: float = 0.1
    hprd_train_epochs: int = 3
    hprd_train_lr: float = 1e-4
    hprd_train_batches: Optional[int] = None
    hprd_ckpt_path: Optional[str] = None


def run_lwe_baseline(args, trainer, data_handler, dataloaders, misclassified: Dict[str, torch.Tensor]) -> Tuple[nn.Module, Optional[float]]:
    device = trainer.device
    model = trainer.model

    if args.max_edits is None:
        edit_indices = misclassified["indices"]
    else:
        edit_indices = misclassified["indices"][: args.max_edits]

    if len(edit_indices) == 0:
        print("No misclassified samples to edit for LWE baseline.")
        return model, None

    transform = trainer.get_transforms()
    error_dataset = data_handler.get_error_samples_dataset(error_indices=edit_indices, transform=transform)
    error_loader = DataLoader(
        error_dataset,
        batch_size=max(1, min(args.batch_size, len(error_dataset))),
        shuffle=True,
        num_workers=0,
        pin_memory=args.pin_memory if args.pin_memory is not None else torch.cuda.is_available(),
    )

    images = []
    labels = []
    for idx in edit_indices:
        image, label = data_handler.get_discovery_dataset(transform)[int(idx)]
        images.append(image)
        labels.append(label)

    edit_images = torch.stack(images).to(device)
    edit_labels = torch.tensor(labels, device=device)
    edit_indices_list = [int(i) for i in edit_indices]

    before = evaluate_edit_samples(
        model=model,
        images=edit_images,
        true_labels=edit_labels,
        sample_indices=edit_indices_list,
        device=device,
        desc="LWE Before",
        batch_size=args.batch_size,
    )
    print(
        f"LWE Edit accuracy BEFORE: {before['accuracy'] * 100:.1f}% "
        f"({before['num_correct']}/{before['num_total']})"
    )

    layer = args.lwe_layer
    if layer is None:
        layer_list = _default_edit_layers(model.config.num_hidden_layers, width=3)
    else:
        layer_list = [layer - 2, layer - 1, layer]

    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start is not None:
        start.record()

    if args.lwe_method == "ft":
        editor = LWEFTEditor(
            model=model,
            layers=layer_list,
            edit_lr=args.lwe_edit_lr,
            max_edit_steps=args.lwe_max_steps,
            cloc=args.lwe_cloc,
            l2_reg=args.lwe_l2_reg,
        )
    else:
        editor = LWEHPRDEditor(
            model=model,
            layers=layer_list,
            device=device,
            sparsity=args.lwe_sparsity,
            edit_lr=args.lwe_edit_lr,
            max_edit_steps=args.lwe_max_steps,
        )

        if args.lwe_hprd_ckpt:
            ckpt = torch.load(args.lwe_hprd_ckpt, map_location=device)
            editor.mask_predictor.load_state_dict(ckpt["mask_predictor"])
            print(f"Loaded HPRD mask predictor from: {args.lwe_hprd_ckpt}")
        else:
            editor.train_mask_predictor(
                error_loader=error_loader,
                epochs=args.lwe_hprd_epochs,
                lr=args.lwe_hprd_lr,
                max_batches_per_epoch=args.lwe_hprd_max_batches,
            )
            ckpt_path = Path(args.checkpoint_dir) / f"{args.model_short}_{args.dataset}_hprd_mask.pt"
            torch.save({"mask_predictor": editor.mask_predictor.state_dict()}, ckpt_path)
            print(f"Saved HPRD mask predictor to: {ckpt_path}")

    for idx in tqdm(range(edit_images.size(0)), desc=f"Applying LWE-{args.lwe_method.upper()} edits"):
        editor.edit_sample(edit_images[idx:idx + 1], edit_labels[idx:idx + 1])

    if end is not None:
        end.record()
        torch.cuda.synchronize()
        edit_seconds = start.elapsed_time(end) / 1000.0
    else:
        edit_seconds = None

    after = evaluate_edit_samples(
        model=model,
        images=edit_images,
        true_labels=edit_labels,
        sample_indices=edit_indices_list,
        device=device,
        desc="LWE After",
        batch_size=args.batch_size,
    )
    print(
        f"LWE Edit accuracy AFTER: {after['accuracy'] * 100:.1f}% "
        f"({after['num_correct']}/{after['num_total']})"
    )

    original_model = copy.deepcopy(model)
    finetuned_path = Path(args.checkpoint_dir) / f"{args.model_short}_{args.dataset}_finetuned.pt"
    if finetuned_path.exists():
        checkpoint = torch.load(finetuned_path, map_location=device)
        original_model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_comparative(
        model_orig=original_model,
        model_edit=model,
        test_loader=dataloaders["test"],
        device=device,
        results_dir=args.results_dir,
        set_name="Test Set",
    )

    save_path = Path(args.checkpoint_dir) / f"{args.model_short}_{args.dataset}_lwe.pt"
    trainer.save_checkpoint(filepath=save_path, is_best=False)
    print(f"Saved LWE model to: {save_path}")

    return model, edit_seconds
