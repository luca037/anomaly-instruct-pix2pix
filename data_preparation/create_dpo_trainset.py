#!/usr/bin/env python
# coding=utf-8
"""Creates the DPO preference metadata.jsonl for pill.

Walks ``TRAIN_PATH/<defect>/{winner,loser}/<img>`` and pairs each winner
with its loser and the corresponding MVTec good image (matched by filename).
The edit prompt is looked up from ``PROMPTS``.

Usage:
    python create_dpo_trainset.py
    python create_dpo_trainset.py --train_path /path/to/dpo_train_images/pill \
        --mvtec_path /path/to/mvtec/pill --output metadata.jsonl
"""

import argparse
import json
import os

# Settings.
TRAIN_PATH = "/home/luca_piai/big_disk/datasets/dpo_train_images/pill/"
MVTEC_PATH = "/home/luca_piai/big_disk/datasets/mvtec/pill/"

PREFIX = "<style1>"
PROMPTS = {
    "crack": PREFIX + "add a crack to the pill",
    "contaminant": PREFIX + "add an embedded contaminant to the pill",
    "hairline": PREFIX + "add a hairline crack to the pill",
    "hole": PREFIX + "add an insect bore hole to the pill",
    "edge": PREFIX + "a pill with a chipped edge",
}


# CLI arguments.


def parse_args():
    """Parses command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Create DPO train set.")
    parser.add_argument(
        "--train_path",
        type=str,
        default=TRAIN_PATH,
        help="Path to DPO train images (winner/loser).",
    )
    parser.add_argument(
        "--mvtec_path", type=str, default=MVTEC_PATH, help="Path to MVTec pill dataset."
    )
    parser.add_argument(
        "--output", type=str, default="metadata.jsonl", help="Output metadata.jsonl path."
    )
    parser.add_argument("--prefix", type=str, default=PREFIX, help="Prompt prefix.")
    return parser.parse_args()


# Main.


def main():
    """Walks the winner/loser folders and writes metadata.jsonl."""
    args = parse_args()

    # Allow overriding the prefix via CLI without editing PROMPTS.
    prompts = {k: v.replace(PREFIX, args.prefix) for k, v in PROMPTS.items()}

    defects_dirs = [
        d for d in os.listdir(args.train_path) if os.path.isdir(os.path.join(args.train_path, d))
    ]

    entries = []
    for defect in defects_dirs:
        if defect not in prompts:
            print(f"Warning: No prompt for defect {defect}, skipping.")
            continue
        winners = os.path.join(args.train_path, defect, "winner")
        losers = os.path.join(args.train_path, defect, "loser")
        if not os.path.isdir(winners) or not os.path.isdir(losers):
            print(f"Warning: Missing winner/loser folder for defect {defect}")
            continue
        for img in os.listdir(winners):
            win_path = os.path.join(winners, img)
            lose_path = os.path.join(losers, img)
            original = os.path.join(args.mvtec_path, "train", "good", img)
            if not os.path.isfile(win_path) or not os.path.isfile(lose_path):
                print(f"Warning: Missing file for image {img} in defect {defect}")
            entries.append(
                {
                    "winner_image": win_path,
                    "loser_image": lose_path,
                    "edit_prompt": prompts[defect],
                    "original_image": original,
                }
            )

    with open(args.output, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Created {args.output} with {len(entries)} entries.")


if __name__ == "__main__":
    main()
