# src/evaluate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from utils import check_data_layout, ensure_dir, get_device


def build_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline classifier on test set.")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/model_best.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    splits = check_data_layout(args.data_dir)

    tfm = build_transform(args.img_size)
    test_ds = datasets.ImageFolder(splits["test"], transform=tfm)

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_to_idx = ckpt.get("class_to_idx", None)

    model = build_model(num_classes=len(test_ds.classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        # probability of class 1 (usually PNEUMONIA if mapped to 1)
        y_prob.extend(probs[:, 1].tolist())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # AUROC only makes sense if both classes exist in y_true
    auroc = None
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = None

    metrics = {
        "accuracy": float(acc),
        "auroc": auroc,
        "confusion_matrix": cm,
        "num_test_samples": len(y_true),
        "dataset_classes": test_ds.classes,
        "class_to_idx_from_ckpt": class_to_idx,
    }

    ensure_dir("results/metrics")
    out_path = Path("results/metrics/test_metrics.json")
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[RESULTS]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
