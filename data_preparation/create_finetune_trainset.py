#!/usr/bin/env python
# coding=utf-8
"""Creates the fine-tuning training set for InstructPix2Pix.

Merges the MIRAGE MVTec and VisA generation tracking CSVs into a single
``metadata.jsonl`` with ``original_image, edited_image, edit_prompt`` rows.
Each CSV row is a ``(object, defect_type)`` pair; a percentage of rows per
pair can be retained via ``--mvtec_perc`` / ``--visa_perc``. The cable
``defect_synonym`` is cleaned to remove the leading instruction prefix.

Usage:
    python create_finetune_trainset.py --download --mvtec_perc 1.0 --visa_perc 1.0
"""

import argparse
import csv
import json
import os
import urllib.request

# Settings.
MVTEC_PATH = "/home/luca_piai/big_disk/datasets/mvtec/"
VISA_PATH = "/home/luca_piai/big_disk/datasets/visa/"
MIRAGE_PATH = "/home/luca_piai/big_disk/datasets/mirage"

TRAINSET_PATH = "/home/luca_piai/big_disk/datasets/train_set/"

MVTEC_CATEGORIES = ["cable", "hazelnut", "leather", "pill", "screw", "tile", "transistor"]
VISA_CATEGORIES = [None]

VISA_URL = "https://huggingface.co/datasets/visualanom/mirage_mvtec_visa/resolve/main/visa/generation_tracking.csv"
MVTEC_URL = "https://huggingface.co/datasets/visualanom/mirage_mvtec_visa/resolve/main/mvtec/generation_tracking.csv"

VISA_CSV = "visa_generation_tracking.csv"
MVTEC_CSV = "mvtec_generation_tracking_augmented.csv"


# Helpers.


def download_file(url, local_path):
    """Downloads a file if it does not already exist."""
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}...")
        urllib.request.urlretrieve(url, local_path)
    print(f"File {local_path} already exists, skipping download.")


def process_csv(csv_path, dataset_path, categories, mirage_path, ds_name, perc):
    """Parses a generation tracking CSV and builds training entries.

    Args:
        csv_path: Path to the tracking CSV.
        dataset_path: Root of the source dataset (MVTec or VisA).
        categories: List of categories to keep. Empty or [None] means skip.
        mirage_path: Root of the MIRAGE dataset.
        ds_name: Subfolder inside MIRAGE (``mvtec`` or ``visa``).
        perc: Fraction of rows to keep per (object, defect) pair.

    Returns:
        List of dicts with ``original_image, edited_image, edit_prompt``.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = [row for row in reader]

    # Group rows by (object, defect_type) where defect_type is the prefix
    # before ":" in defect_synonym.
    rows_dict = {}
    for row in all_rows:
        obj = row.get("object")
        defect = row.get("defect_synonym").split(":")[0]
        if (obj, defect) not in rows_dict:
            rows_dict[(obj, defect)] = [row]
        else:
            rows_dict[(obj, defect)].append(row)

    filtered_rows = []
    for _, v in rows_dict.items():
        end = int(len(v) * perc)
        filtered_rows.extend(v[:end])

    entries = []
    for row in filtered_rows:
        obj_category = row.get("object", "")
        if not categories or obj_category in categories:
            base_img = row.get("base_img_path", "")
            img = row.get("img_path", "")
            prompt = row.get("defect_synonym", "")

            # Adjust edited_image path: from "/transistor/test/general/..." to
            # "transistor/images/general_0000.png". Mirrors MIRAGE layout.
            img = os.path.join(img.split("/")[0], "images", os.path.basename(img))

            base_img_full = os.path.join(dataset_path, base_img.lstrip("/"))
            img_full = os.path.join(mirage_path, ds_name, img.lstrip("/"))

            entries.append(
                {
                    "original_image": base_img_full,
                    "edited_image": img_full,
                    "edit_prompt": prompt,
                }
            )
    return entries


def clean_cable_defect_prompt(mvtec_csv):
    """Removes the instruction prefix from cable defect_synonym entries.

    The MVTec cable prompts start with "Please generate an image... . ";
    this keeps only the part after the first ". ".

    Args:
        mvtec_csv: Path to the MVTec tracking CSV to clean in place.
    """
    with open(mvtec_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        if row.get("object") == "cable":
            prompt = row.get("defect_synonym", "")
            split = prompt.split(". ")
            if len(split) >= 2:
                row["defect_synonym"] = "".join(split[1:])

    with open(mvtec_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# CLI arguments.


def parse_args():
    """Parses command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Create train set.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download mvtec csv file and visa csv file from MIRAGE dataset repository.",
    )
    parser.add_argument("--mvtec_path", type=str, default=MVTEC_PATH, help="Path to MVTec dataset.")
    parser.add_argument("--visa_path", type=str, default=VISA_PATH, help="Path to VisA dataset.")
    parser.add_argument(
        "--mirage_path", type=str, default=MIRAGE_PATH, help="Path to MIRAGE dataset."
    )
    parser.add_argument(
        "--trainset_path", type=str, default=TRAINSET_PATH, help="Output directory for train set."
    )
    parser.add_argument(
        "--mvtec_categories",
        type=str,
        nargs="*",
        default=MVTEC_CATEGORIES,
        help="MVTec categories to include. Use 'None' to skip.",
    )
    parser.add_argument(
        "--visa_categories",
        type=str,
        nargs="*",
        default=VISA_CATEGORIES,
        help="VisA categories to include. Use 'None' to skip.",
    )
    parser.add_argument(
        "--mvtec_perc",
        type=float,
        default=1.0,
        help="How many samples are retained (1.0 means all, 0.5 means half discarded).",
    )
    parser.add_argument(
        "--visa_perc",
        type=float,
        default=1.0,
        help="How many samples are retained (1.0 means all, 0.5 means half discarded).",
    )
    return parser.parse_args()


# Main.


def main():
    """Creates metadata.jsonl from the tracking CSVs."""
    args = parse_args()

    # Handle "None" string passed via CLI for VISA/MVTec categories.
    mvtec_categories = (
        [None if c == "None" else c for c in args.mvtec_categories] if args.mvtec_categories else []
    )
    visa_categories = (
        [None if c == "None" else c for c in args.visa_categories] if args.visa_categories else []
    )

    if args.download:
        download_file(VISA_URL, VISA_CSV)
        download_file(MVTEC_URL, MVTEC_CSV)
        print("Cleaning MVTEC cable prompt...")
        clean_cable_defect_prompt(MVTEC_CSV)

    assert 0.0 <= args.mvtec_perc <= 1.0, "mvtec_perc must be between 0.0 and 1.0"
    assert 0.0 <= args.visa_perc <= 1.0, "visa_perc must be between 0.0 and 1.0"

    all_entries = []
    if not len(mvtec_categories) or mvtec_categories[0] is not None:
        print("Processing MVTEC data...")
        all_entries.extend(
            process_csv(
                MVTEC_CSV,
                args.mvtec_path,
                mvtec_categories,
                args.mirage_path,
                ds_name="mvtec",
                perc=args.mvtec_perc,
            )
        )
    if not len(visa_categories) or visa_categories[0] is not None:
        print("Processing VISA data...")
        all_entries.extend(
            process_csv(
                VISA_CSV,
                args.visa_path,
                visa_categories,
                args.mirage_path,
                ds_name="visa",
                perc=args.visa_perc,
            )
        )

    os.makedirs(args.trainset_path, exist_ok=True)
    out_path = os.path.join(args.trainset_path, "metadata.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Created {out_path} with {len(all_entries)} entries.")


if __name__ == "__main__":
    main()
