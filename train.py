"""
EyeNet — Production Training Pipeline
======================================
3-phase fine-tuning for retinal fundus classification.
ResNet50 + EfficientNetB0 + DenseNet121 Joint Feature Ensemble.

Usage:
  python train.py --data_dir ./data --epochs 30 --device cpu
  python train.py --data_dir ./data --epochs 30 --device cuda --resume ./weights/eyenet_ensemble_last.pth
"""

import argparse
import os
import random
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.models.model import build_model


# ════════════════════════════════════════════════════════════════════════════
# 0. REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════════════════

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING — must be IDENTICAL to inference/utils.py
# ════════════════════════════════════════════════════════════════════════════

CLASS_NAMES    = ["Diabetic Retinopathy", "Glaucoma", "Cataract", "Normal"]
FOLDER_NAMES   = ["diabetic_retinopathy", "glaucoma", "cataract", "normal"]
IMAGE_SIZE     = 256
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]


def crop_fundus_circle(img_bgr: np.ndarray) -> np.ndarray:
    """Crop the retinal circle from the black background."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = 8
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img_bgr.shape[1] - x, w + 2 * pad)
    h = min(img_bgr.shape[0] - y, h + 2 * pad)
    cropped = img_bgr[y:y + h, x:x + w]
    return cropped if cropped.size > 0 else img_bgr


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on L-channel in LAB space to enhance vessel/lesion contrast."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Unsharp masking for vessel detail enhancement."""
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=2)
    return cv2.addWeighted(img_bgr, 1.4, blur, -0.4, 0)


class FundusPreprocess:
    """
    OpenCV preprocessing applied BEFORE torchvision transforms.
    Steps: crop → CLAHE → unsharp mask → convert to RGB.
    Must be identical to inference preprocess_image() in utils.py.
    """
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        img_bgr = crop_fundus_circle(img_bgr)
        img_bgr = apply_clahe(img_bgr)
        img_bgr = apply_unsharp_mask(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb


# ════════════════════════════════════════════════════════════════════════════
# 2. TRANSFORMS
# ════════════════════════════════════════════════════════════════════════════

def get_train_transform() -> transforms.Compose:
    """Training: preprocessing + strong augmentation + normalize."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # — Strong augmentations (training only) ——————————————————————————
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0,
                                translate=(0.05, 0.05),
                                scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),   # after ToTensor
        # — Normalize ——————————————————————————————————————————————————
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """Validation/test: preprocessing only, no augmentation."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ════════════════════════════════════════════════════════════════════════════

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_flat_samples(root: str) -> list[tuple[str, int]]:
    """Collect (path, label) from a flat class-wise folder (no train/val split)."""
    samples = []
    for idx, folder in enumerate(FOLDER_NAMES):
        cls_dir = os.path.join(root, folder)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] Missing class dir: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                samples.append((os.path.join(cls_dir, fname), idx))
    return samples


def stratified_split(samples, train_r=0.70, val_r=0.15, seed=42):
    """Stratified split into train / val / test lists."""
    rng = random.Random(seed)
    by_cls: dict[int, list] = {}
    for s in samples:
        by_cls.setdefault(s[1], []).append(s)
    train, val, test = [], [], []
    for items in by_cls.values():
        rng.shuffle(items)
        n  = len(items)
        n1 = max(1, int(n * train_r))
        n2 = max(1, int(n * val_r))
        train += items[:n1]
        val   += items[n1:n1 + n2]
        test  += items[n1 + n2:]
    return train, val, test


class RetinalDataset(Dataset):
    """
    Loads retinal fundus images.
    Supports both:
      - a list of (path, label) tuples  (flat/auto-split mode)
      - a directory root with class subfolders (train/val/test mode)
    Applies FundusPreprocess (OpenCV) then torchvision transforms.
    """
    def __init__(self, source, transform, is_train: bool = False):
        self.transform   = transform
        self.is_train    = is_train
        self.preprocess  = FundusPreprocess()

        if isinstance(source, list):           # pre-built sample list
            self.samples = source
        else:                                   # directory root
            self.samples = collect_flat_samples(source)

        if not self.samples:
            raise RuntimeError(f"No images found in {source}")

        print(f"  Dataset: {len(self.samples)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Return black image as safe fallback
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), label

        # Steps 1-4: OpenCV preprocessing (must match inference)
        img_rgb = self.preprocess(img_bgr)

        # Steps 5: torchvision transforms (+ augmentation for train)
        tensor = self.transform(img_rgb)
        return tensor.float(), label


def compute_class_weights(dataset: RetinalDataset,
                           device: torch.device) -> torch.Tensor:
    """
    weight[i] = total_samples / (num_classes * samples_in_class_i)
    Handles class imbalance for weighted CrossEntropyLoss.
    """
    num_classes = len(CLASS_NAMES)
    counts = [0] * num_classes
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class counts: { {CLASS_NAMES[i]: counts[i] for i in range(num_classes)} }")
    print(f"  Class weights: { {CLASS_NAMES[i]: round(weights[i], 3) for i in range(num_classes)} }")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ════════════════════════════════════════════════════════════════════════════
# 4. TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:  # mixed precision (GPU only)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tta: bool = False):
    """
    Evaluation with optional TTA (horizontal flip average).
    Returns: loss, accuracy, f1_macro, all_preds, all_labels
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if use_tta:
            logits_orig  = model(imgs)
            logits_flip  = model(torch.flip(imgs, dims=[3]))
            logits       = (logits_orig + logits_flip) / 2.0
        else:
            logits = model(imgs)

        loss     = criterion(logits, labels)
        preds    = logits.argmax(1)

        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc      = correct / total
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, acc, f1_macro, all_preds, all_labels


def log_confusion_matrix(all_labels, all_preds, phase: str):
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n  [Confusion Matrix — {phase}]")
    header = f"{'':>24}" + "".join(f"{n[:7]:>10}" for n in CLASS_NAMES)
    print("  " + header)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i][:22]:>24}" + "".join(f"{v:>10}" for v in row))

    print(f"\n  [Classification Report — {phase}]")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES,
                                zero_division=0))


# ════════════════════════════════════════════════════════════════════════════
# 5. CHECKPOINT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, path: str, note: str = ""):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    tag = f" ({note})" if note else ""
    print(f"  [✓] Saved{tag}: {path}")


def load_checkpoint(model, path: str, device: torch.device):
    if not os.path.isfile(path):
        print(f"  [WARN] Checkpoint not found: {path}")
        return
    state = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  [Resume] {path} | missing={len(missing)}, unexpected={len(unexpected)}")


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_phase(
    phase_name: str,
    model, train_loader, val_loader, criterion,
    optimizer, scheduler, device,
    epochs: int,
    best_f1: float,
    no_improve: int,
    patience: int,
    weights_best: str,
    weights_last: str,
    use_scaler: bool = False,
) -> tuple[float, int]:
    """
    Run one training phase. Returns updated (best_f1, no_improve).
    """
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    print(f"\n{'='*60}")
    print(f"  {phase_name}  ({epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_f1, preds, labels = evaluate(
            model, val_loader, criterion, device, use_tta=False)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Ep {epoch:3d}/{epochs} | "
            f"T-loss {tr_loss:.4f} | T-acc {tr_acc:.3f} | "
            f"V-loss {val_loss:.4f} | V-acc {val_acc:.3f} | "
            f"V-F1(mac) {val_f1:.4f} | {elapsed:.0f}s"
        )

        # Per-class accuracy quick view
        f1_per = f1_score(labels, preds, average=None, zero_division=0)
        cls_str = " | ".join(f"{CLASS_NAMES[i][:4]}: {f1_per[i]:.2f}"
                              for i in range(len(CLASS_NAMES)))
        print(f"           Per-class F1 → {cls_str}")

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            save_checkpoint(model, weights_best,
                            note=f"best F1={best_f1:.4f}")
        else:
            no_improve += 1
            print(f"           No improve {no_improve}/{patience}")

        # Always save last
        save_checkpoint(model, weights_last, note="last")

        # Early stopping
        if no_improve >= patience:
            print(f"  [Early Stop] patience {patience} reached.")
            break

    # Phase-end confusion matrix
    log_confusion_matrix(labels, preds, phase_name)
    return best_f1, no_improve


def main():
    # ── CLI ────────────────────────────────────────────────────────────────
    p = argparse.ArgumentParser(description="EyeNet Training Pipeline")
    p.add_argument("--data_dir",   required=True,
                   help="Dataset root. Either flat (class folders directly) or "
                        "structured (train/ val/ test/ subdirs).")
    p.add_argument("--split_mode", default="auto",
                   choices=["auto", "flat", "split"],
                   help="'flat'=single dir with class folders (auto-split 70/15/15), "
                        "'split'=pre-split train/val/test dirs, 'auto'=detect.")
    p.add_argument("--epochs",     type=int, default=30,
                   help="Total epochs across all phases (default 30)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for phase 1 & 2 (phase 3 uses half)")
    p.add_argument("--device",     default="cpu",
                   choices=["cpu", "cuda"],
                   help="Training device")
    p.add_argument("--resume",     default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--patience",   type=int, default=7,
                   help="Early stopping patience (epochs)")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    set_seeds(args.seed)

    device = torch.device("cuda" if args.device == "cuda"
                          and torch.cuda.is_available() else "cpu")
    print(f"\n[EyeNet] Device: {device} | Seed: {args.seed}")

    # Weights paths
    weights_best = "./weights/eyenet_ensemble.pth"
    weights_last = "./weights/eyenet_ensemble_last.pth"
    os.makedirs("./weights", exist_ok=True)

    # ── DETECT DATASET STRUCTURE ───────────────────────────────────────────
    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")
    test_dir  = os.path.join(args.data_dir, "test")

    is_split = (os.path.isdir(train_dir) and os.path.isdir(val_dir))
    if args.split_mode == "auto":
        use_flat = not is_split
    elif args.split_mode == "flat":
        use_flat = True
    else:
        use_flat = False

    print(f"\n[EyeNet] Dataset mode: {'flat (auto-split)' if use_flat else 'pre-split'}")
    print("[EyeNet] Loading datasets...")

    if use_flat:
        all_samples = collect_flat_samples(args.data_dir)
        if not all_samples:
            print(f"[ERROR] No images found in {args.data_dir}")
            sys.exit(1)
        train_s, val_s, test_s = stratified_split(all_samples, seed=args.seed)
        print(f"  Auto-split → train={len(train_s)} | val={len(val_s)} | test={len(test_s)}")
        train_ds = RetinalDataset(train_s, get_train_transform(), is_train=True)
        val_ds   = RetinalDataset(val_s,   get_val_transform(),   is_train=False)
        test_samples = test_s
    else:
        train_ds = RetinalDataset(train_dir, get_train_transform(), is_train=True)
        val_ds   = RetinalDataset(val_dir,   get_val_transform(),   is_train=False)
        test_samples = None

    # ── Class weights from TRAINING SET only (no leakage) ──────────────────
    class_weights = compute_class_weights(train_ds, device)

    # ── LOSS FUNCTION ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1,
    )

    # ── DATA LOADERS ───────────────────────────────────────────────────────
    num_workers = 0 if device.type == "cpu" else 4  # safe default for Windows

    def make_loader(ds, bs, shuffle):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))

    train_loader_16 = make_loader(train_ds, args.batch_size, shuffle=True)
    train_loader_8  = make_loader(train_ds, max(1, args.batch_size // 2), shuffle=True)
    val_loader      = make_loader(val_ds,   args.batch_size, shuffle=False)

    # ── MODEL ──────────────────────────────────────────────────────────────
    print("\n[EyeNet] Building EyeNetEnsemble (pretrained=True)...")
    model = build_model(pretrained=True).to(device)

    if args.resume:
        load_checkpoint(model, args.resume, device)

    # ── PHASE BUDGETS ──────────────────────────────────────────────────────
    total_epochs = args.epochs
    p1_ep = min(5,  total_epochs // 6)
    p2_ep = min(10, total_epochs // 3)
    p3_ep = total_epochs - p1_ep - p2_ep
    print(f"\n[EyeNet] Phase budgets: P1={p1_ep} | P2={p2_ep} | P3={p3_ep}")

    use_scaler = (device.type == "cuda")
    best_f1   = 0.0
    no_improve = 0

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1 — Fusion Head Warmup
    # ════════════════════════════════════════════════════════════════════
    model.freeze_backbones()
    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    best_f1, no_improve = run_phase(
        "PHASE 1 — Fusion Head Warmup",
        model, train_loader_16, val_loader, criterion,
        opt1, scheduler=None, device=device,
        epochs=p1_ep,
        best_f1=best_f1, no_improve=no_improve,
        patience=args.patience,
        weights_best=weights_best, weights_last=weights_last,
        use_scaler=use_scaler,
    )

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2 — Gradual Backbone Unfreeze (last blocks only)
    # ════════════════════════════════════════════════════════════════════
    model.unfreeze_backbones_gradual(stage=1)
    opt2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=p2_ep)
    best_f1, no_improve = run_phase(
        "PHASE 2 — Gradual Unfreeze (last blocks)",
        model, train_loader_16, val_loader, criterion,
        opt2, scheduler=sch2, device=device,
        epochs=p2_ep,
        best_f1=best_f1, no_improve=no_improve,
        patience=args.patience,
        weights_best=weights_best, weights_last=weights_last,
        use_scaler=use_scaler,
    )

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3 — Full Fine-tuning
    # ════════════════════════════════════════════════════════════════════
    model.unfreeze_backbones()
    opt3 = torch.optim.Adam(model.parameters(), lr=1e-5)
    sch3 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt3, T_0=5, T_mult=2
    )
    best_f1, no_improve = run_phase(
        "PHASE 3 — Full Fine-tuning",
        model, train_loader_8, val_loader, criterion,
        opt3, scheduler=sch3, device=device,
        epochs=p3_ep,
        best_f1=best_f1, no_improve=no_improve,
        patience=args.patience,
        weights_best=weights_best, weights_last=weights_last,
        use_scaler=use_scaler,
    )

    # ════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION ON TEST SET (with TTA)
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  FINAL TEST SET EVALUATION (TTA enabled)")
    print(f"{'='*60}")

    # Load best weights for final test
    load_checkpoint(model, weights_best, device)
    model.eval()

    if os.path.isdir(test_dir):
        test_ds     = RetinalDataset(test_dir, get_val_transform(), is_train=False)
        test_loader = make_loader(test_ds, args.batch_size, shuffle=False)
        _, test_acc, test_f1, preds, labels = evaluate(
            model, test_loader, criterion, device, use_tta=True
        )
        print(f"\n  Test Accuracy : {test_acc:.4f}")
        print(f"  Test F1 Macro : {test_f1:.4f}")
        log_confusion_matrix(labels, preds, "FINAL TEST (TTA)")
    else:
        print(f"  [SKIP] test/ dir not found: {test_dir}")

    print(f"\n[EyeNet] ✓ Training complete. Best Val F1 Macro: {best_f1:.4f}")
    print(f"         Best weights: {weights_best}")
    print(f"         Last weights: {weights_last}\n")


from training_pipeline import main as upgraded_main


if __name__ == "__main__":
    upgraded_main()
    raise SystemExit
