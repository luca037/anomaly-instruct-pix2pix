"""
Shared constants and helpers for the eval pipeline.

Keeps behaviour identical to the pre-refactor scripts; just centralises
duplicated constants and argparse / plotting helpers so each
train/test script can expose a thin ``parse_args()`` like
``finetune_instruct_pix2pix.py``.
"""

import os

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    "cable",
    "screw",
    "transistor",
    "leather",
    "hazelnut",
    "pill",
    "tile",
    "carpet",
    "capsule",
    "wood",
    "metal_nut",
]

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
LOCALIZE_SIZE = 256

DEFAULT_CLASSIFICATION_CKPT = "eval/checkpoints/classification"
DEFAULT_LOCALIZATION_CKPT = "eval/checkpoints/localization"


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def add_common_args(parser, ckpt_default=None):
    """Add the 4 args that appear in every eval script."""
    parser.add_argument("--device", type=str, default="cuda:1", help="e.g. cuda:1")
    parser.add_argument("--mvtec_path", type=str, required=True, help="Path to real MVTec dataset")
    parser.add_argument("--generated_path", type=str, required=True, help="Path to generated defect images")
    parser.add_argument("--checkpoint_path", type=str, default=ckpt_default)
    parser.add_argument("--categories", type=str, nargs="+", default=CATEGORIES)


def add_training_args(parser):
    """Add the training-related args shared by train-* and partly by test-*."""
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--clip_bad_json",
        type=str,
        default=None,
        help="Optional path to clip_bad.json (from eval/clip_filter.py); images listed as BAD are excluded from the synthetic train set.",
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_losses(train_losses, val_losses, run_name, checkpoint_path, ylabel="loss", title=None):
    """Save a train/val loss-curve plot for one object under the checkpoints dir."""
    if not train_losses:
        return
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="val loss", marker="s", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else f"{run_name} - train / val {ylabel}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(checkpoint_path, f"{run_name}_loss.jpg")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved loss plot to {out_path}")
