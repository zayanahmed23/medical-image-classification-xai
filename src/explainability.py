# src/explainability.py
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from utils import check_data_layout, ensure_dir, get_device


# -------- Grad-CAM core --------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Forward hook to capture activations
        self.target_layer.register_forward_hook(self._forward_hook)
        # Backward hook to capture gradients
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] has shape [B, C, H, W]
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Returns a CAM heatmap for a single image tensor x (shape [1, 3, H, W]).
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        # Global-average-pool gradients over spatial dims -> weights per channel
        grads = self.gradients  # [1, C, H, W]
        acts = self.activations  # [1, C, H, W]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        cam = (weights * acts).sum(dim=1, keepdim=False)  # [1, H, W]
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


# -------- Helpers --------
def build_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor [3,H,W] -> uint8 BGR image for OpenCV overlay.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # RGB uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def overlay_cam(bgr_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay heatmap on BGR image.
    """
    h, w = bgr_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(bgr_img, 1 - alpha, heatmap, alpha, 0)
    return blended


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Collect predictions and labels for the whole set.
    Returns lists: y_true, y_pred, probs_class1
    """
    model.eval()
    y_true, y_pred, y_prob1 = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob1.extend(probs[:, 1].tolist())

    return y_true, y_pred, y_prob1


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations.")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/model_best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "false_positives", "true_positives", "false_negatives"])
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] Device: {device}")

    splits = check_data_layout(args.data_dir)
    tfm = build_transform(args.img_size)

    ds = datasets.ImageFolder(splits[args.split], transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device.type == "cuda"))

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(num_classes=len(ds.classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("[INFO] Classes:", ds.classes)
    print("[INFO] class_to_idx:", ds.class_to_idx)

    # Choose target layer: last conv block of ResNet18
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    # Build index list depending on mode
    y_true, y_pred, _ = collect_predictions(model, loader, device)

    idxs = list(range(len(ds)))
    if args.mode == "false_positives":
        # true NORMAL (0) predicted PNEUMONIA (1)
        idxs = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == 0 and p == 1]
    elif args.mode == "true_positives":
        # true PNEUMONIA (1) predicted PNEUMONIA (1)
        idxs = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == 1 and p == 1]
    elif args.mode == "false_negatives":
        # true PNEUMONIA (1) predicted NORMAL (0)
        idxs = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == 1 and p == 0]

    if len(idxs) == 0:
        raise RuntimeError(f"No samples found for mode={args.mode}. Try another mode/split.")

    # Take first N (deterministic). You can randomize later if you want.
    selected = idxs[: args.num_samples]

    out_dir = ensure_dir(Path("results/visualizations/gradcam") / args.split / args.mode)
    print(f"[INFO] Saving to: {out_dir}")

    for k, idx in enumerate(selected, start=1):
        img_tensor, true_label = ds[idx]  # img_tensor is normalized [3,H,W]
        x = img_tensor.unsqueeze(0).to(device)

        # Predicted class for this image
        with torch.no_grad():
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())

        # Generate CAM for predicted class (standard usage)
        cam = gradcam(x, class_idx=pred)

        bgr = denormalize(img_tensor)
        overlay = overlay_cam(bgr, cam, alpha=args.alpha)

        true_name = ds.classes[true_label]
        pred_name = ds.classes[pred]

        out_path = out_dir / f"{k:02d}_idx{idx}_true-{true_name}_pred-{pred_name}.png"
        cv2.imwrite(str(out_path), overlay)

    print(f"[DONE] Saved {len(selected)} Grad-CAM overlays.")


if __name__ == "__main__":
    main()
