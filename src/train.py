# src/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from utils import check_data_layout, ensure_dir, get_device, set_seed


def build_transforms(img_size: int = 224):
    # Standard ImageNet normalization works fine for a baseline.
    return {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    }


def build_model(num_classes: int = 2) -> nn.Module:
    # Pretrained ResNet18 baseline
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train baseline classifier (ResNet18) on chest X-ray dataset.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to data/raw containing train/val/test.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Device: {device}")

    splits = check_data_layout(args.data_dir)
    tfms = build_transforms(args.img_size)

    train_ds = datasets.ImageFolder(splits["train"], transform=tfms["train"])
    val_ds = datasets.ImageFolder(splits["val"], transform=tfms["val"])

    print("[INFO] class_to_idx:", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = build_model(num_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = ensure_dir("results/checkpoints")
    metrics_dir = ensure_dir("results/metrics")
    run_start = time()

    history = {"train": [], "val": [], "args": vars(args), "class_to_idx": train_ds.class_to_idx}

    best_val_acc = -1.0
    best_path = ckpt_dir / "model_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)

        print(f"[EPOCH {epoch}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        history["train"].append({"epoch": epoch, "loss": tr_loss, "acc": tr_acc})
        history["val"].append({"epoch": epoch, "loss": va_loss, "acc": va_acc})

        # Save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "args": vars(args),
            }, best_path)

    history["best_val_acc"] = best_val_acc
    history["runtime_sec"] = round(time() - run_start, 2)

    with open(metrics_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[DONE] Best val acc: {best_val_acc:.4f}")
    print(f"[SAVED] Checkpoint: {best_path}")
    print(f"[SAVED] History: results/metrics/train_history.json")


if __name__ == "__main__":
    main()
