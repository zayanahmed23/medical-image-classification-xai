# src/faithfulness_deletion.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from utils import check_data_layout, ensure_dir, get_device


# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # [B, C, H, W]

    def cam_for_class(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        x: [1,3,H,W]
        returns cam [H,W] in [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        grads = self.gradients  # [1,C,H,W]
        acts = self.activations  # [1,C,H,W]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        cam = (weights * acts).sum(dim=1)[0]  # [H,W]
        cam = torch.relu(cam).cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


# --------------- Helpers ----------------
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
def pneumonia_prob(model: nn.Module, x: torch.Tensor, pneumonia_idx: int) -> float:
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return float(probs[0, pneumonia_idx].item())


def topk_mask(cam: np.ndarray, topk_percent: float) -> np.ndarray:
    """
    cam: [H,W] in [0,1]
    returns mask [H,W] boolean where True indicates "delete these pixels"
    """
    assert 0 < topk_percent < 100
    thresh = np.percentile(cam, 100 - topk_percent)
    return cam >= thresh


def apply_deletion(img_tensor: torch.Tensor, del_mask: np.ndarray, fill: str = "mean") -> torch.Tensor:
    """
    img_tensor: [3,H,W] normalized tensor
    del_mask: [H,W] boolean
    fill:
      - "mean": fill with 0 in normalized space (≈ dataset mean after normalization)
      - "black": fill with very low value (-2.0) in normalized space
    """
    x = img_tensor.clone()
    if fill == "mean":
        fill_val = 0.0
    elif fill == "black":
        fill_val = -2.0
    else:
        raise ValueError("fill must be 'mean' or 'black'")

    mask_t = torch.from_numpy(del_mask).to(x.device)  # [H,W]
    for c in range(3):
        x[c][mask_t] = fill_val
    return x


def denormalize_to_bgr(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: [3,H,W] normalized
    -> BGR uint8 for saving quick previews
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def save_mask_preview(out_path: Path, original: torch.Tensor, masked: torch.Tensor, cam: np.ndarray, del_mask: np.ndarray):
    """
    Saves a 3-panel preview: original | masked | CAM heatmap
    """
    orig_bgr = denormalize_to_bgr(original)
    masked_bgr = denormalize_to_bgr(masked)

    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    # visualize deletion mask as white overlay
    mask_vis = (del_mask.astype(np.uint8) * 255)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    # stack: original, masked, heatmap+mask
    heat_plus = cv2.addWeighted(heat, 0.75, mask_vis, 0.25, 0)

    panel = np.hstack([orig_bgr, masked_bgr, heat_plus])
    cv2.imwrite(str(out_path), panel)


def pick_examples(ds: datasets.ImageFolder, y_true: List[int], y_pred: List[int],
                  pneumonia_idx: int, normal_idx: int) -> List[Tuple[str, int]]:
    """
    Returns a list of (tag, idx) for the 6 selected categories based on dataset indices.
    We'll pick the FIRST match we find for each tag to keep it deterministic.
    """
    idxs = list(range(len(ds)))

    def first_idx(cond):
        for i in idxs:
            if cond(i):
                return i
        return None

    # TP (PNEUMONIA->PNEUMONIA): 2 samples
    tp1 = first_idx(lambda i: y_true[i] == pneumonia_idx and y_pred[i] == pneumonia_idx)
    tp2 = first_idx(lambda i: y_true[i] == pneumonia_idx and y_pred[i] == pneumonia_idx and i != tp1)

    # FP (NORMAL->PNEUMONIA): 2 samples
    fp1 = first_idx(lambda i: y_true[i] == normal_idx and y_pred[i] == pneumonia_idx)
    fp2 = first_idx(lambda i: y_true[i] == normal_idx and y_pred[i] == pneumonia_idx and i != fp1)

    # Random: 2 samples (any)
    r1 = 0 if len(ds) > 0 else None
    r2 = 1 if len(ds) > 1 else None

    chosen = [
        ("tp_01", tp1),
        ("tp_02", tp2),
        ("fp_01", fp1),
        ("fp_02", fp2),
        ("random_01", r1),
        ("random_02", r2),
    ]

    # Filter missing
    return [(tag, idx) for tag, idx in chosen if idx is not None]


@torch.no_grad()
def get_preds(model: nn.Module, ds: datasets.ImageFolder, device: torch.device, batch_size: int = 32):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Deletion faithfulness test for Grad-CAM.")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/model_best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--topk", type=float, default=25.0, help="Top-k percent CAM pixels to delete (e.g., 25).")
    parser.add_argument("--fill", type=str, default="mean", choices=["mean", "black"],
                        help="Fill value for deleted pixels in normalized space.")
    parser.add_argument("--save_previews", action="store_true", help="Save side-by-side preview images.")
    parser.add_argument("--out_json", type=str, default="results/metrics/faithfulness_deletion.json")
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] Device: {device}")

    splits = check_data_layout(args.data_dir)
    tfm = build_transform(224)
    ds = datasets.ImageFolder(splits[args.split], transform=tfm)

    # class indices from dataset
    class_to_idx = ds.class_to_idx
    if "PNEUMONIA" not in class_to_idx or "NORMAL" not in class_to_idx:
        raise RuntimeError(f"Expected classes NORMAL and PNEUMONIA, found: {ds.classes}")

    pneumonia_idx = class_to_idx["PNEUMONIA"]
    normal_idx = class_to_idx["NORMAL"]

    # load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(num_classes=len(ds.classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Grad-CAM setup: last block
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    # get predictions (needed to pick TP/FP examples deterministically)
    y_true, y_pred = get_preds(model, ds, device, batch_size=32)

    selected = pick_examples(ds, y_true, y_pred, pneumonia_idx, normal_idx)
    print("[INFO] Selected examples:", selected)

    out_json_path = Path(args.out_json)
    ensure_dir(out_json_path.parent)

    preview_dir = Path("results/visualizations/faithfulness_deletion") / args.split
    if args.save_previews:
        ensure_dir(preview_dir)

    results: Dict[str, Any] = {
        "split": args.split,
        "topk_percent_deleted": args.topk,
        "fill": args.fill,
        "checkpoint": args.checkpoint,
        "class_to_idx": class_to_idx,
        "items": []
    }

    # Run deletion test per selected image
    for tag, idx in selected:
        img_tensor, true_label = ds[idx]
        x = img_tensor.unsqueeze(0).to(device)

        # baseline prob for pneumonia
        p_before = pneumonia_prob(model, x, pneumonia_idx)

        # predicted class (for reference)
        with torch.no_grad():
            pred = int(torch.argmax(model(x), dim=1).item())

        # CAM for predicted class (standard)
        cam = gradcam.cam_for_class(x, class_idx=pred)  # [H,W]
        
        # Resize CAM (e.g., 7x7) to input size (224x224) so we can mask the image
        H, W = img_tensor.shape[1], img_tensor.shape[2]  # 224, 224
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)


        del_mask = topk_mask(cam, args.topk)  # boolean [H,W]
        masked_tensor = apply_deletion(img_tensor.to(device), del_mask, fill=args.fill)
        x_masked = masked_tensor.unsqueeze(0)

        p_after = pneumonia_prob(model, x_masked, pneumonia_idx)
        drop = p_before - p_after

        item = {
            "tag": tag,
            "dataset_index": idx,
            "true_label": ds.classes[true_label],
            "pred_label": ds.classes[pred],
            "p_pneumonia_before": p_before,
            "p_pneumonia_after": p_after,
            "confidence_drop": drop,
            "deleted_fraction": float(del_mask.mean()),
        }
        results["items"].append(item)

        if args.save_previews:
            out_path = preview_dir / f"{tag}_idx{idx}_true-{ds.classes[true_label]}_pred-{ds.classes[pred]}.png"
            save_mask_preview(out_path, img_tensor, masked_tensor.detach().cpu(), cam, del_mask)

        print(f"[DONE] {tag} idx={idx} true={ds.classes[true_label]} pred={ds.classes[pred]} "
              f"p_before={p_before:.4f} p_after={p_after:.4f} drop={drop:.4f}")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[SAVED] {out_json_path}")
    if args.save_previews:
        print(f"[SAVED] previews to {preview_dir}")


if __name__ == "__main__":
    main()
