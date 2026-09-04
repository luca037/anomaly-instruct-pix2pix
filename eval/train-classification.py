"""
Train a per-object defect-type classifier (ResNet-34) on synthetic ip2p images.

Adapted from O2MAG `eval/train-classification.py`. One fresh ResNet-34 is
trained per object: its final ``fc`` layer is sized to that object's number of
defect classes (the defect subdirectories under
``<generated_data_path>/<object>/test/``).

Differences vs. upstream:
  * Adds ``--device`` (repo convention: default ``cuda:1``, never ``cuda:0``).
  * Training data is read from the MVTec-mirrored layout written by
    ``eval/generate_defects.py``  (``<generated_data_path>/<object>/test/<defect>/``).
  * Uses ``--categories`` to pick objects (default: the 7 classes we evaluate:
    cable, screw, transistor, leather, hazelnut, pill, tile).
  * Fixed a bug where validation accuracy was reported only for the last batch
    of each class (now accumulated across the whole test set).

Usage
-----
    uv run eval/train-classification.py \
        --mvtec_path /home/luca_piai/big_disk/datasets/mvtec \
        --generated_data_path /home/luca_piai/big_disk/datasets/generated \
        --checkpoint_path eval/checkpoints/classification \
        --categories cable screw transistor leather hazelnut pill tile \
        --device cuda:1
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet34
from unet_utils.data_loader import MVTec_classification_test, MVTec_classification_train

CATEGORIES = ["cable", "screw", "transistor", "leather", "hazelnut", "pill", "tile", "carpet", "capsule", "wood", "metal_nut"]


def test(args, obj_name, model, anomaly_names):
    """Evaluate loss + accuracy on the real MVTec test split for one object."""
    model.eval()
    dataset = MVTec_classification_test(args, obj_name, anomaly_names)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(args.device)
            label = label.to(args.device)
            y_pred = model(image)
            loss_sum += criterion(y_pred, label).item() * label.size(0)
            prediction = torch.argmax(y_pred, 1)
            correct += (prediction == label).sum().item()
            total += label.size(0)
    acc = correct / total if total > 0 else 0.0
    val_loss = loss_sum / total if total > 0 else 0.0
    print(f"  val loss: {val_loss:.4f}  Accuracy: {acc:.4f} ({correct}/{total})")
    return acc, val_loss


def _plot_losses(train_losses, val_losses, run_name, checkpoint_path):
    """Save a train/val loss-curve plot for one object under the checkpoints dir."""
    if not train_losses:
        return
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="val loss", marker="s", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{run_name} - train / val loss")
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
        dataset = MVTec_classification_train(args, obj_name)
        class_num = dataset.class_num()
        anomaly_names = dataset.return_anomaly_names()
        print(f"  classes ({class_num}): {anomaly_names}")

        model = resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT, progress=True)
        model.fc = nn.Linear(model.fc.in_features, class_num)
        model = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [int(args.epochs * 0.8), int(args.epochs * 0.9)], gamma=0.2
        )
        criterion = nn.CrossEntropyLoss()
        dataloader = DataLoader(
            dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers
        )

        max_acc = 0.0
        train_losses = []
        val_losses = []
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            n_seen = 0
            print(f"  Epoch: {epoch}", end=" ")
            for image, label in dataloader:
                image = image.to(args.device)
                label = label.to(args.device)
                y_pred = model(image)
                loss = criterion(y_pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * label.size(0)
                n_seen += label.size(0)
            train_loss = running_loss / n_seen
            train_losses.append(train_loss)
            print(f"  train loss: {train_loss:.4f}", end=" ")

            scheduler.step()
            acc, val_loss = test(args, obj_name, model, anomaly_names)
            val_losses.append(val_loss)
            if acc > max_acc:
                max_acc = acc
                torch.save(
                    model.state_dict(),
                    os.path.join(args.checkpoint_path, run_name + ".pckl"),
                )
            print(f"  best so far: {max_acc:.4f}")

        _plot_losses(train_losses, val_losses, run_name, args.checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Train per-object defect classifiers.")
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
        "--checkpoint_path", default="./eval/checkpoints/classification", type=str
    )
    parser.add_argument("--categories", type=str, nargs="+", default=CATEGORIES)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_on_device(args.categories, args)


if __name__ == "__main__":
    main()
