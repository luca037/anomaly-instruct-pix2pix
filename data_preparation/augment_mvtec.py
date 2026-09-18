#!/usr/bin/env python
# coding=utf-8
"""Augments the defect_synonym column in the MVTec generation tracking CSV.

This script increases prompt variance by asking DeepSeek to generate ``n``
paraphrases for each unique defect prompt (e.g. 9 paraphrases + the original
gives 10 prompts per defect, 100 per object). The paraphrases keep the
original defect-type prefix (``"hole: ..."``) and are then distributed
round-robin over the duplicated CSV rows.

Requirements:
    Python 3.x, ``openai`` (``pip install openai``), and a valid
    ``DEEPSEEK_API_KEY`` environment variable.

Usage:
    export DEEPSEEK_API_KEY="your-key"
    python augment_mvtec.py --generate --variations 9  # API + CSV
    python augment_mvtec.py                            # CSV only, reuse JSON
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

from openai import OpenAI

# Settings
INPUT_CSV = "csv/mvtec_generation_tracking.csv"
OUTPUT_CSV = "csv/mvtec_generation_tracking_augmented.csv"
JSON_FILE = "json/augmented_prompts.json"


# Helpers.


def get_unique_prompts(csv_path):
    """Extracts the set of unique defect_synonym prompts per object.

    Args:
        csv_path: Path to the tracking CSV with ``object`` and
            ``defect_synonym`` columns.

    Returns:
        Dict mapping ``object`` to a set of unique prompts.
    """
    unique_prompts = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row["object"]
            prompt = row["defect_synonym"]
            if obj not in unique_prompts:
                unique_prompts[obj] = set()
            unique_prompts[obj].add(prompt)
    return unique_prompts


def generate_prompts(n):
    """Generates paraphrases for each unique defect prompt via DeepSeek.

    The original CSV is scanned for unique prompts, then the DeepSeek API
    (accessed through the OpenAI client) is asked for ``n`` semantically
    equivalent variations per prompt. The defect-type prefix before ``": "``
    is preserved. Results are saved incrementally to ``JSON_FILE`` so the
    process can be resumed if interrupted.

    Args:
        n: Number of variations to generate per unique prompt.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY environment variable not set. Please set it to run the API.")
        return

    # Initialize the OpenAI client for the DeepSeek endpoint.
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    unique_prompts = get_unique_prompts(INPUT_CSV)
    augmented_prompts = {}

    # Resume from a previous JSON if it exists.
    if os.path.exists(JSON_FILE):
        print(f"Loading existing {JSON_FILE}")
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            augmented_prompts = json.load(f)

    for obj, prompts in unique_prompts.items():
        if obj not in augmented_prompts:
            augmented_prompts[obj] = {}

        for original_prompt in prompts:
            # Split the defect-type prefix from the description.
            if ": " in original_prompt:
                defect_type, desc = original_prompt.split(": ", 1)
            else:
                defect_type = ""
                desc = original_prompt

            print(f"Generating variations for: ({obj}, {defect_type})")

            system_prompt = "You are a helpful assistant that generates variations of sentences with the same meaning but different words."
            user_prompt = (
                f"Generate exactly {n} variations of the following description. "
                f"Explain using other words but keep the exact same meaning. "
                f"Do not include any prefix like numbers or bullet points, "
                f"just return the {n} variations separated by a newline.\n\n"
                f"Description: {desc}"
            )

            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                )

                variations = response.choices[0].message.content.strip().split("\n")

                clean_variations = []
                for v in variations:
                    v = v.strip()
                    if v:
                        # Remove markdown, dashes, or numbering.
                        v = re.sub(r"^(\d+\.|-|\*)\s*", "", v)
                        v = v.strip("\"'")
                        if defect_type:
                            clean_variations.append(f"{defect_type}: {v}")
                        else:
                            clean_variations.append(v)

                clean_variations = clean_variations[:n]

                if original_prompt not in augmented_prompts[obj]:
                    augmented_prompts[obj][original_prompt] = clean_variations
                else:
                    augmented_prompts[obj][original_prompt].extend(clean_variations)

                # Save incrementally after each successful call.
                with open(JSON_FILE, "w", encoding="utf-8") as f:
                    json.dump(augmented_prompts, f, indent=4)

            except Exception as e:
                print(f"Error generating for {original_prompt}: {e}")

    print(f"Successfully generated prompts and saved to {JSON_FILE}")


def create_augmented_csv():
    """Creates the augmented CSV by cycling through paraphrases.

    Reads ``JSON_FILE`` and replaces the ``defect_synonym`` column in
    ``INPUT_CSV`` round-robin per ``(object, original_prompt)`` pair. The
    pool for each pair is ``[original] + variations``.
    """
    if not os.path.exists(JSON_FILE):
        print(
            f"Error: {JSON_FILE} not found. Please run with --generate first to create the prompts."
        )
        return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        augmented_prompts = json.load(f)

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found.")
        return

    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    obj_prompt_counters = defaultdict(int)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            obj = row["object"]
            orig_prompt = row["defect_synonym"]
            variations = augmented_prompts.get(obj, {}).get(orig_prompt, [])
            pool = [orig_prompt] + variations
            idx = obj_prompt_counters[(obj, orig_prompt)] % len(pool)
            row["defect_synonym"] = pool[idx]
            obj_prompt_counters[(obj, orig_prompt)] += 1
            writer.writerow(row)

    print(f"Created {OUTPUT_CSV} with distributed augmented prompts.")

    unique_prompts = get_unique_prompts(OUTPUT_CSV)
    print("\nUnique prompts per object category")
    for o, p in unique_prompts.items():
        print(f"{o}, {len(p)}")


# Main.


def parse_args():
    """Parses command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Augment MVTec generation tracking prompts.")
    parser.add_argument(
        "--generate", action="store_true", help="Generate new prompts using the DeepSeek API."
    )
    parser.add_argument(
        "--variations", type=int, default=0, help="How many variations per unique prompt."
    )
    return parser.parse_args()


def main():
    """Runs prompt generation (optional) and CSV augmentation."""
    args = parse_args()

    if args.generate and args.variations > 0:
        print("Starting prompt generation...")
        generate_prompts(args.variations)

    print("Creating augmented CSV...")
    create_augmented_csv()


if __name__ == "__main__":
    main()
