from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import seaborn as sns

from backend.torch_bootstrap import prepare_torch_environment

prepare_torch_environment()

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader

from backend.config import (
    CLASS_NAMES,
    CLASS_WEIGHT_DR_BOOST,
    CLASS_WEIGHT_GLAUCOMA_BOOST,
    DEFAULT_WEIGHTS_PATH,
    MODEL_NAME,
    REPORTS_DIR,
)
from backend.models.model import EyeNetEnsemble, build_model
from backend.services.data_pipeline import (
    RetinalDataset,
    build_weighted_sampler,
    class_distribution,
    collect_samples,
    detect_pre_split_dataset,
    drop_filename_label_conflicts,
    stratified_split,
)

matplotlib.use("Agg")


@dataclass
class PhaseConfig:
    name: str
    epochs: int
    mode: str
    head_lr: float
    backbone_lr: float
    batch_size: int


def parse_args():
    parser = argparse.ArgumentParser(description="Train the EyeNet retinal ensemble")
    parser.add_argument("--data_dir", required=True, help="Dataset root directory")
    parser.add_argument("--split_mode", default="auto", choices=["auto", "flat", "split"])
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--robust_aug",
        action="store_true",
        help="Enable extra domain-robust training augmentations (JPEG artifacts, blur, mild crop, gamma). "
             "Off by default to preserve baseline behavior.",
    )
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument(
        "--mixup_in_warmup",
        action="store_true",
        help="Apply Mixup during the fusion warmup phase (off by default for small datasets).",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--output", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--cache_dir", default=None, help="Optional cache directory for preprocessed images")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--max_samples_per_class", type=int, default=0)
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable inverse-frequency weighting (and DR/Glaucoma boosts).",
    )
    parser.add_argument("--dr_boost", type=float, default=CLASS_WEIGHT_DR_BOOST)
    parser.add_argument("--glaucoma_boost", type=float, default=CLASS_WEIGHT_GLAUCOMA_BOOST)
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="If > 0, use focal loss with this gamma instead of plain cross-entropy.",
    )
    parser.add_argument(
        "--no_one_cycle",
        action="store_true",
        help="Use per-epoch CosineAnnealingLR instead of OneCycleLR (batch-stepped).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_samples(args) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    data_dir = Path(args.data_dir)
    use_split = args.split_mode == "split" or (
        args.split_mode == "auto" and detect_pre_split_dataset(data_dir)
    )

    if use_split:
        train_samples = collect_samples(data_dir / "train")
        val_samples = collect_samples(data_dir / "val")
        test_samples = collect_samples(data_dir / "test") if (data_dir / "test").is_dir() else []
    else:
        train_ratio = 1.0 - args.val_ratio - args.test_ratio
        all_samples = collect_samples(data_dir)
        if not all_samples:
            raise RuntimeError(f"No retinal images found in {data_dir}")
        train_samples, val_samples, test_samples = stratified_split(
            all_samples,
            train_ratio=train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    if not train_samples or not val_samples:
        raise RuntimeError("Training and validation samples are required.")

    train_samples = drop_filename_label_conflicts(train_samples)
    val_samples = drop_filename_label_conflicts(val_samples)
    test_samples = drop_filename_label_conflicts(test_samples)

    if args.max_samples_per_class > 0:
        rng = random.Random(args.seed)

        def cap(samples):
            by_class = {idx: [] for idx in range(len(CLASS_NAMES))}
            for sample in samples:
                by_class[sample[1]].append(sample)
            capped = []
            for idx in by_class:
                rng.shuffle(by_class[idx])
                capped.extend(by_class[idx][: args.max_samples_per_class])
            return capped

        train_samples = cap(train_samples)
        val_samples = cap(val_samples)
        test_samples = cap(test_samples)

    return train_samples, val_samples, test_samples


def build_loaders(args, train_samples, val_samples, test_samples, phase_batch_size):
    cache_root = Path(args.cache_dir) if args.cache_dir else None
    train_cache = cache_root / "train" if cache_root else None
    val_cache = cache_root / "val" if cache_root else None
    test_cache = cache_root / "test" if cache_root else None

    train_dataset = RetinalDataset(
        train_samples,
        training=True,
        robust_aug=bool(getattr(args, "robust_aug", False)),
        cache_dir=train_cache,
    )
    val_dataset = RetinalDataset(val_samples, training=False, cache_dir=val_cache)
    test_dataset = RetinalDataset(test_samples, training=False, cache_dir=test_cache) if test_samples else None

    train_sampler = build_weighted_sampler(train_samples)
    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": args.device == "cuda" and torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=phase_batch_size,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=phase_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=phase_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    return train_loader, val_loader, test_loader


def compute_class_weight_tensor(
    train_samples: Sequence[Tuple[str, int]],
    num_classes: int,
    device: torch.device,
    dr_boost: float,
    glaucoma_boost: float,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, y in train_samples:
        counts[int(y)] += 1.0
    counts = counts.clamp(min=1.0)
    w = counts.sum() / (num_classes * counts)
    w[0] = w[0] * dr_boost
    w[1] = w[1] * glaucoma_boost
    w = w / w.mean()
    return w.float().to(device)


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("class_weight", weight.clone().float())
        else:
            self.register_buffer("class_weight", torch.empty(0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.class_weight if self.class_weight.numel() > 0 else None
        ce = F.cross_entropy(
            logits,
            target,
            weight=w,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce).clamp(min=1e-8, max=1.0)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def mixup_batch(inputs, targets, alpha: float):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, targets, targets[index], float(lam)


def mixup_loss(criterion, logits, targets_a, targets_b, lam: float):
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)


def create_optimizer(model: EyeNetEnsemble, head_lr: float, backbone_lr: float, weight_decay: float):
    head_module = getattr(model, "fusion", None) or getattr(model, "fc", None)
    if head_module is None:
        raise AttributeError("Model is missing a recognized classifier head module.")
    head_parameters = [param for param in head_module.parameters() if param.requires_grad]

    backbone_modules = [model.resnet, getattr(model, "efficientnet", None), getattr(model, "effnet", None), model.densenet]
    backbone_parameters = []
    for backbone in backbone_modules:
        if backbone is None:
            continue
        backbone_parameters.extend(
            [param for param in backbone.parameters() if param.requires_grad]
        )

    parameter_groups = []
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": head_lr})
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": backbone_lr})

    return AdamW(parameter_groups, weight_decay=weight_decay)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    mixup_alpha: float,
    scheduler=None,
    scheduler_steps_per_batch: bool = False,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        mixed_images, targets_a, targets_b, lam = mixup_batch(images, labels, mixup_alpha)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(mixed_images)
            loss = mixup_loss(criterion, logits, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and scheduler_steps_per_batch:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tta: bool = False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        if use_tta:
            flipped_logits = model(torch.flip(images, dims=[3]))
            logits = (logits + flipped_logits) / 2.0

        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "preds": all_preds,
    }


def save_checkpoint(model, path: Path, payload: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"state_dict": model.state_dict(), **payload}
    torch.save(data, path)


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return checkpoint


def build_phase_plan(total_epochs: int, batch_size: int, head_lr: float, backbone_lr: float):
    total_epochs = max(1, total_epochs)
    # Longer fusion warmup before unfreezing backbones (helps small fundus sets).
    phase1 = max(1, int(round(total_epochs * 0.45)))
    phase2 = max(1, int(round(total_epochs * 0.35)))
    phase3 = total_epochs - phase1 - phase2
    # Ensure exact budget match.
    while phase1 + phase2 + phase3 > total_epochs and phase2 > 1:
        phase2 -= 1
    while phase1 + phase2 + phase3 > total_epochs and phase1 > 1:
        phase1 -= 1
    phase3 = max(0, total_epochs - phase1 - phase2)
    return [
        PhaseConfig("Fusion Warmup", phase1, "warmup", head_lr, 0.0, batch_size),
        PhaseConfig("Selective Unfreeze", phase2, "partial", head_lr * 0.4, backbone_lr, batch_size),
        PhaseConfig("Full Fine-Tune", phase3, "full", head_lr * 0.2, backbone_lr * 0.5, max(1, batch_size // 2)),
    ]


def prepare_model_for_phase(model: EyeNetEnsemble, phase: PhaseConfig):
    if phase.mode == "warmup":
        model.freeze_backbones()
    elif phase.mode == "partial":
        model.unfreeze_backbones_gradual(stage=1)
    else:
        model.unfreeze_backbones()


def report_dir_for_run() -> Path:
    return REPORTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")


def save_training_curves(history: Sequence[Dict[str, float]], out_dir: Path):
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    val_f1 = [row["val_macro_f1"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, train_losses, label="Train loss", color="#0a9396")
    axes[0].plot(epochs, val_losses, label="Validation loss", color="#ae2012")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="Validation macro F1", color="#ee9b00")
    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def save_confusion(labels, preds, out_path: Path, title: str):
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="crest",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    train_samples, val_samples, test_samples = resolve_samples(args)
    run_dir = report_dir_for_run()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[EyeNet] Model: {MODEL_NAME}")
    print(f"[EyeNet] Device: {device}")
    print(f"[EyeNet] Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | Test samples: {len(test_samples)}")
    print(f"[EyeNet] Train distribution: {class_distribution(train_samples)}")

    model = build_model(pretrained=True).to(device)
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        print(f"[EyeNet] Resumed from {args.resume} (best macro F1={checkpoint.get('best_macro_f1', 'n/a')})")

    num_classes = len(CLASS_NAMES)
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weight_tensor(
            train_samples,
            num_classes,
            device,
            dr_boost=args.dr_boost,
            glaucoma_boost=args.glaucoma_boost,
        )
        print(f"[EyeNet] Class weights: {class_weights.cpu().tolist()}")

    if args.focal_gamma > 0:
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    phase_plan = build_phase_plan(args.epochs, args.batch_size, args.head_lr, args.backbone_lr)
    history: List[Dict[str, float]] = []
    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    output_path = Path(args.output)
    last_output_path = output_path.with_name(f"{output_path.stem}_last{output_path.suffix}")
    global_epoch = 0

    for phase in phase_plan:
        prepare_model_for_phase(model, phase)
        train_loader, val_loader, _ = build_loaders(
            args,
            train_samples,
            val_samples,
            test_samples,
            phase.batch_size,
        )
        optimizer = create_optimizer(
            model=model,
            head_lr=phase.head_lr,
            backbone_lr=max(phase.backbone_lr, 1e-6),
            weight_decay=args.weight_decay,
        )
        total_steps = max(1, len(train_loader) * phase.epochs)
        use_one_cycle = not args.no_one_cycle
        if use_one_cycle:
            max_lrs = [g["lr"] for g in optimizer.param_groups]
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1e4,
            )
            sched_batch = True
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(phase.epochs, 1))
            sched_batch = False

        print(f"\n{'=' * 72}")
        print(f"[Phase] {phase.name} | epochs={phase.epochs} | batch_size={phase.batch_size}")
        print(f"{'=' * 72}")

        for _ in range(phase.epochs):
            global_epoch += 1
            epoch_start = time.time()

            effective_mixup = (
                0.0 if (phase.mode == "warmup" and not args.mixup_in_warmup) else args.mixup_alpha
            )

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                effective_mixup,
                scheduler=scheduler if sched_batch else None,
                scheduler_steps_per_batch=sched_batch,
            )
            val_metrics = evaluate(model, val_loader, criterion, device, use_tta=False)
            if not sched_batch:
                scheduler.step()

            epoch_record = {
                "epoch": global_epoch,
                "phase": phase.name,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
                "seconds": round(time.time() - epoch_start, 2),
            }
            history.append(epoch_record)

            print(
                f"Epoch {global_epoch:02d} | "
                f"train_loss={epoch_record['train_loss']:.4f} | "
                f"train_acc={epoch_record['train_accuracy']:.4f} | "
                f"val_loss={epoch_record['val_loss']:.4f} | "
                f"val_acc={epoch_record['val_accuracy']:.4f} | "
                f"val_f1={epoch_record['val_macro_f1']:.4f}"
            )

            save_checkpoint(
                model,
                last_output_path,
                {
                    "best_macro_f1": best_macro_f1,
                    "class_names": CLASS_NAMES,
                    "history": history,
                },
            )

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                epochs_without_improvement = 0
                save_checkpoint(
                    model,
                    output_path,
                    {
                        "best_macro_f1": best_macro_f1,
                        "class_names": CLASS_NAMES,
                        "history": history,
                    },
                )
            else:
                epochs_without_improvement += 1
                print(f"  No validation macro-F1 improvement for {epochs_without_improvement}/{args.patience} epoch(s)")

            if epochs_without_improvement >= args.patience:
                print("  Early stopping triggered on validation macro F1")
                break

        if epochs_without_improvement >= args.patience:
            break

    if output_path.exists():
        load_checkpoint(model, str(output_path), device)

    _, val_loader, test_loader = build_loaders(
        args,
        train_samples,
        val_samples,
        test_samples,
        args.batch_size,
    )
    final_val_metrics = evaluate(model, val_loader, criterion, device, use_tta=not args.disable_tta)

    summary = {
        "model": MODEL_NAME,
        "best_validation_macro_f1": round(best_macro_f1, 4),
        "final_validation_loss": round(final_val_metrics["loss"], 4),
        "final_validation_accuracy": round(final_val_metrics["accuracy"], 4),
        "final_validation_macro_f1": round(final_val_metrics["macro_f1"], 4),
        "weights_path": str(output_path),
        "reports_dir": str(run_dir),
        "class_names": CLASS_NAMES,
        "epochs_completed": global_epoch,
        "timestamp": datetime.now().isoformat(),
        "train_config": {
            "epochs_budget": args.epochs,
            "mixup_alpha": args.mixup_alpha,
            "mixup_in_warmup": args.mixup_in_warmup,
            "label_smoothing": args.label_smoothing,
            "class_weights": not args.no_class_weights,
            "dr_boost": args.dr_boost,
            "glaucoma_boost": args.glaucoma_boost,
            "focal_gamma": args.focal_gamma,
            "scheduler": "cosine_per_epoch" if args.no_one_cycle else "one_cycle_per_phase",
            "phase_split": "0.45_warmup_0.35_partial_0.20_full",
        },
    }

    save_training_curves(history, run_dir)
    save_confusion(
        final_val_metrics["labels"],
        final_val_metrics["preds"],
        run_dir / "validation_confusion_matrix.png",
        "Validation Confusion Matrix",
    )

    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, criterion, device, use_tta=not args.disable_tta)
        summary.update(
            {
                "test_loss": round(test_metrics["loss"], 4),
                "test_accuracy": round(test_metrics["accuracy"], 4),
                "test_macro_f1": round(test_metrics["macro_f1"], 4),
            }
        )
        save_confusion(
            test_metrics["labels"],
            test_metrics["preds"],
            run_dir / "test_confusion_matrix.png",
            "Test Confusion Matrix",
        )
        report = classification_report(
            test_metrics["labels"],
            test_metrics["preds"],
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        )
        with open(run_dir / "test_classification_report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    else:
        print("[EyeNet] Test split not available. Final metrics are validation-only.")

    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(run_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"\n[EyeNet] Training complete")
    print(f"[EyeNet] Best weights: {output_path}")
    print(f"[EyeNet] Last weights: {last_output_path}")
    print(f"[EyeNet] Reports: {run_dir}")
    print(f"[EyeNet] Best validation macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
