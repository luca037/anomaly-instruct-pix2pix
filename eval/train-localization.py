"""
Train a per-object anomaly-detection segmentor (DRAEM ``DiscriminativeSubNetwork``)
on synthetic ip2p-generated defects.

Ported from O2MAG `eval/train-localization.py`. One segmentor is trained per
object: input = 3-channel image, target = 1-channel anomaly mask.

Training data:
  * synthetic defects + their **stored** on-disk Otsu masks from
    ``<generated_data_path>/<object>/test/<defect>`` + ``ground_truth/<defect>``
    (the masks written by ``eval/generate_defects.py`` — reused as-is), and
  * real MVTec ``<object>/train/good`` images as the normal class (ip2p only
    generates defects, so there are no synthetic goods).

Differences vs. upstream:
  * Adds ``--device`` (repo convention: default ``cuda:1``, never ``cuda:0``).
  * Loss is FocalLoss + SSIM on the mask head.
  * Saves the best-by-val-focal-loss checkpoint as ``<object>.pckl``.
  * Prints train loss + val loss per epoch and saves ``<object>_loss.jpg``.

Usage
-----
    uv run eval/train-localization.py \
        --mvtec_path /home/luca_piai/big_disk/datasets/mvtec \
        --generated_data_path /home/luca_piai/big_disk/datasets/generated \
        --checkpoint_path eval/checkpoints/localization \
        --categories cable screw transistor leather hazelnut pill tile \
        --device cuda:1
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from unet_utils.data_loader import (
    AnomalyLocalizationTrainDataset,
    AnomalyLocalizationTestDataset,
)
from unet_utils.loss import FocalLoss, SSIMLoss
from unet_utils.model_unet import DiscriminativeSubNetwork

CATEGORIES = [  # noqa: E501
    "cable", "screw", "transistor", "leather", "hazelnut", "pill", "tile",
    "carpet", "capsule", "wood", "metal_nut",
]

def _object_defect_types(args, obj_name):
    """Discover defect-type names from the generated ``test/`` folder."""
    test_root = os.path.join(args.generated_data_path, obj_name, "test")
    return [
        d
        for d in sorted(os.listdir(test_root))
        if os.path.isdir(os.path.join(test_root, d))
    ] if os.path.isdir(test_root) else CATEGORIES[:1]


def test_focal_loss(args, obj_name, model, anomaly_names):
    """Evaluate the SAME composite loss used in training (FocalLoss + 0.1·SSIM)
    on the real MVTec test split for one object, so train and val losses are
    directly comparable."""
    model.eval()
    dataset = AnomalyLocalizationTestDataset(args, obj_name, anomaly_names)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
    criterion = FocalLoss(gamma=2.0, alpha=-1.0, beta=1.0, reduction="mean")
    ssim = SSIMLoss()

    loss_sum = 0.0
    total = 0
    with torch.no_grad():
        for image, mask, _defect in dataloader:
            image = image.to(args.device)
            mask = mask.to(args.device)
            pred = model(image)
            step_loss = criterion(pred, mask) + 0.1 * ssim(pred, mask)
            loss_sum += step_loss.item() * image.size(0)
            total += image.size(0)
    return loss_sum / total if total > 0 else float("nan")


def _plot_losses(train_losses, val_losses, run_name, checkpoint_path):
    if not train_losses:
        return
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="val loss", marker="s", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("focal loss")
    ax.set_title(f"{run_name} - train / val focal loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(checkpoint_path, f"{run_name}_loss.jpg")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved loss plot to {out_path}")


def train_on_device(obj_names, args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    for obj_name in obj_names:
        print(obj_name)
        run_name = obj_name
        train_dataset = AnomalyLocalizationTrainDataset(args, obj_name)
        anomaly_names = _object_defect_types(args, obj_name)
        print(f"  train samples: {len(train_dataset)} | defects: {anomaly_names}")

        model = DiscriminativeSubNetwork(in_channels=3, out_channels=1)
        model = model.to(args.device)

        focal = FocalLoss(gamma=2.0, alpha=-1.0, beta=1.0, reduction="mean")
        ssim = SSIMLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [int(args.epochs * 0.8), int(args.epochs * 0.9)], gamma=0.2
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.num_workers,
        )

        best_val = float("inf")
        train_losses = []
        val_losses = []
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            n_seen = 0
            print(f"  Epoch: {epoch}", end=" ")
            for image, mask in dataloader:
                image = image.to(args.device)
                mask = mask.to(args.device)
                pred = model(image)
                loss = focal(pred, mask) + 0.1 * ssim(pred, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * image.size(0)
                n_seen += image.size(0)
            train_loss = running_loss / n_seen
            train_losses.append(train_loss)
            print(f"  train loss: {train_loss:.4f}", end=" ")

            scheduler.step()
            val_loss = test_focal_loss(args, obj_name, model, anomaly_names)
            val_losses.append(val_loss)
            print(f"  val loss: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(args.checkpoint_path, run_name + ".pckl"),
                )
            print(f"  best so far: {best_val:.4f}")

        _plot_losses(train_losses, val_losses, run_name, args.checkpoint_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train per-object anomaly-detection segmentors."
    )
    parser.add_argument("--device", type=str, default="cuda:1", help="e.g. cuda:1")
    parser.add_argument(
        "--mvtec_path", type=str, required=True, help="Path to real MVTec dataset"
    )
    parser.add_argument(
        "--generated_data_path",
        type=str,
        required=True,
        help="Path to generated defect images",
    )
    parser.add_argument(
        "--checkpoint_path", default="./eval/checkpoints/localization", type=str
    )
    parser.add_argument("--categories", type=str, nargs="+", default=CATEGORIES)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--clip_bad_json",
        type=str,
        default=None,
        help="Optional path to clip_bad.json (from eval/clip_filter.py); "
        "synthetic images listed as BAD are excluded from training.",
    )
    args = parser.parse_args()

    train_on_device(args.categories, args)


if __name__ == "__main__":
    main()
