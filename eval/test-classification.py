"""
Evaluate the per-object defect classifiers trained by `train-classification.py`
on the real MVTec test set.

Adapted from O2MAG `eval/test-classification.py`. Loads each object's checkpoint
(``<checkpoint_path>/<object>.pckl``), builds the matching test dataset from real
MVTec ``<mvtec_path>/<object>/test/<defect>/`` images, and reports per-object
top-1 accuracy across all defect classes.

Usage
-----
    uv run eval/test-classification.py \
        --mvtec_path /home/luca_piai/big_disk/datasets/mvtec \
        --generated_data_path /home/luca_piai/big_disk/datasets/generated \
        --checkpoint_path eval/checkpoints/classification \
        --categories cable screw transistor leather hazelnut pill tile \
        --device cuda:1
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet34
from unet_utils.data_loader import MVTec_classification_test, MVTec_classification_train

CATEGORIES = ["cable", "screw", "transistor", "leather", "hazelnut", "pill", "tile"]


def test(args, obj_name, model, anomaly_names):
    """Evaluate loss + accuracy on the real MVTec test split for one object.

    Reports overall accuracy plus per-defect (per-class) accuracy over the
    validation (real MVTec) set.
    """
    model.eval()
    dataset = MVTec_classification_test(args, obj_name, anomaly_names)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    num_classes = len(anomaly_names)
    per_correct = [0] * num_classes
    per_total = [0] * num_classes

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
            for lbl, pred in zip(label.tolist(), prediction.tolist()):
                per_total[lbl] += 1
                if lbl == pred:
                    per_correct[lbl] += 1
    acc = correct / total if total > 0 else 0.0
    val_loss = loss_sum / total if total > 0 else 0.0
    print(f"  val loss: {val_loss:.4f}  Accuracy: {acc:.4f} ({correct}/{total})")
    per_defect = {}
    for i, name in enumerate(anomaly_names):
        c = per_correct[i]
        t = per_total[i]
        acc_c = c / t if t > 0 else 0.0
        per_defect[name] = (acc_c, c, t)
        print(f"    {name:12s}: {acc_c:.4f} ({c}/{t})")
    return acc, val_loss, per_defect


def eval_train_set(args, obj_name, model):
    """Compute loss + accuracy on the generated train set (reports 'train loss')."""
    train_dataset = MVTec_classification_train(args, obj_name)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for idx in range(
            train_dataset.length
        ):  # unique images only (skip the 5x oversample)
            image, label = train_dataset[idx]
            image = image.unsqueeze(0).to(args.device)
            label = torch.tensor([label], device=args.device)
            y_pred = model(image)
            loss_sum += criterion(y_pred, label).item()
            correct += (y_pred.argmax(1) == label).sum().item()
            total += 1
    acc = correct / total if total > 0 else 0.0
    train_loss = loss_sum / total if total > 0 else 0.0
    print(f"  train loss: {train_loss:.4f}  train acc: {acc:.4f} ({correct}/{total})")
    return train_loss, acc


def test_on_device(obj_names, args):
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {args.checkpoint_path}")

    results = {}
    for obj_name in obj_names:
        print(obj_name)
        run_name = obj_name
        # The train dataset defines the class set + ordering; reuse it so the
        # fc output dim and label indices match those used at training time.
        train_dataset = MVTec_classification_train(args, obj_name)
        class_num = train_dataset.class_num()
        anomaly_names = train_dataset.return_anomaly_names()

        model = resnet34(
            weights=torchvision.models.ResNet34_Weights.DEFAULT, progress=True
        )
        model.fc = nn.Linear(model.fc.in_features, class_num)
        model = model.to(args.device)
        model.load_state_dict(
            torch.load(os.path.join(args.checkpoint_path, run_name + ".pckl"))
        )

        train_loss, train_acc = 0, 0#eval_train_set(args, obj_name, model)
        acc, val_loss, per_defect = test(args, obj_name, model, anomaly_names)
        results[obj_name] = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": acc,
            "per_defect": per_defect,
        }

    print("\nClassification summary per object:")
    for obj_name, r in results.items():
        print(
            f"  {obj_name}: "
            f"train loss={r['train_loss']:.4f} train acc={r['train_acc']:.4f} "
            f"| val loss={r['val_loss']:.4f} val acc={r['val_acc']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate per-object defect classifiers."
    )
    parser.add_argument("--device", type=str, default="cuda:1", help="e.g. cuda:1")
    parser.add_argument(
        "--mvtec_path", type=str, required=True, help="Path to real MVTec dataset"
    )
    parser.add_argument(
        "--generated_data_path",
        type=str,
        required=True,
        help="Path to generated defect images (defines classes)",
    )
    parser.add_argument(
        "--checkpoint_path", default="eval/checkpoints/classification", type=str
    )
    parser.add_argument("--categories", type=str, nargs="+", default=CATEGORIES)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    test_on_device(args.categories, args)


if __name__ == "__main__":
    main()
