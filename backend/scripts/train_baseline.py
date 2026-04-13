from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from backend.config import CLASS_NAMES, CLASS_SLUGS, NUM_CLASSES, REPORTS_DIR
from backend.services.data_pipeline import (
    RetinalDataset,
    collect_samples,
    drop_filename_label_conflicts,
    stratified_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fast baseline trainer (ResNet18).")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples_per_class", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / total, correct / total, macro_f1, labels_all, preds_all


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_samples = collect_samples(args.data_dir)
    if args.max_samples_per_class > 0:
        limited = []
        rng = random.Random(args.seed)
        by_class = {idx: [] for idx in range(NUM_CLASSES)}
        for sample in all_samples:
            by_class[sample[1]].append(sample)
        for idx in range(NUM_CLASSES):
            rng.shuffle(by_class[idx])
            limited.extend(by_class[idx][: args.max_samples_per_class])
        all_samples = limited
        print(f"Using capped dataset: {len(all_samples)} samples")
    train_samples, val_samples, test_samples = stratified_split(
        all_samples,
        train_ratio=1.0 - args.val_ratio - args.test_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_samples = drop_filename_label_conflicts(train_samples)
    val_samples = drop_filename_label_conflicts(val_samples)
    test_samples = drop_filename_label_conflicts(test_samples)

    train_ds = RetinalDataset(train_samples, training=True)
    val_ds = RetinalDataset(val_samples, training=False)
    test_ds = RetinalDataset(test_samples, training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = 0.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_macro_f1": round(val_f1, 4),
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, criterion, device)
    report = classification_report(
        test_labels,
        test_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "class_slugs": CLASS_SLUGS,
            "best_val_macro_f1": best_f1,
            "history": history,
        },
        output_path,
    )

    run_dir = REPORTS_DIR / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "weights": str(output_path),
        "best_val_macro_f1": round(best_f1, 4),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "test_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
