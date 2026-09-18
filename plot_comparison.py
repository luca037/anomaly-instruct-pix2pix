"""
Plot a model-comparison grid from a results.json file.

This script is decoupled from generation: run ``generate.py compare`` to
produce images and ``results.json``, then run this script to tweak the plot
without re-generating.

Grid layout:
  Row 0: Original input images (with the prompt below each one).
  Rows 1…N: Outputs from each model, one row per model.
  Columns: One column per (image, prompt) pair from the test set.

Use ``--start`` / ``--end`` to slice columns when the full grid is too wide.

Usage:
    uv run plot_comparison.py --results results.json --category hazelnut --output hazelnut_grid.jpg
    uv run plot_comparison.py --results results.json --category pill --start 0 --end 4 --output pill_first4.jpg
"""

import argparse
import json
import textwrap

import matplotlib.pyplot as plt
from PIL import Image

# Helpers.


def load_image(path):
    """Loads an image, returning a blank placeholder on failure.

    Args:
        path: Path to the image file.

    Returns:
        PIL image in RGB mode, or a white 512x512 placeholder if loading fails.
    """
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not load image {path}. Using blank placeholder. Error: {e}")
        return Image.new("RGB", (512, 512), color="white")


# CLI arguments.


def parse_args():
    """Parses command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Plot a model-comparison grid from results.json.")
    parser.add_argument("--results", type=str, required=True, help="Path to results.json")
    parser.add_argument(
        "--category", type=str, required=True, help="Category to plot (e.g. hazelnut, pill)"
    )
    parser.add_argument(
        "--output", type=str, default="model_comparison_grid.jpg", help="Output image path"
    )
    parser.add_argument("--start", type=int, default=0, help="Starting index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Ending index (exclusive)")
    return parser.parse_args()


# Main.


def main():
    """Builds and saves the comparison grid."""
    args = parse_args()

    # Load the results JSON produced by ``generate.py compare``.
    print(f"Loading data from {args.results}...")
    with open(args.results, "r") as f:
        data = json.load(f)

    if args.category not in data:
        raise ValueError(f"Category '{args.category}' not found. Available: {list(data.keys())}")

    category = data[args.category]

    # Slice inputs/prompts so a narrower grid can be produced.
    inputs = category.get("inputs", [])[args.start : args.end]
    prompts = category.get("prompts", [])[args.start : args.end]
    outputs = category.get("outputs", {})

    if not inputs:
        raise ValueError("No input images found for the given range.")

    model_names = list(outputs.keys())
    num_cols = len(inputs)
    num_rows = 1 + len(model_names)

    print(f"Generating grid: {num_rows} rows x {num_cols} columns...")

    # Width scales with columns; height with models.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5.5))

    # Ensure 2D indexing works when there is a single column.
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    for col_idx in range(num_cols):
        # Row 0: original input with its prompt underneath.
        ax_input = axes[0, col_idx]
        img_input = load_image(inputs[col_idx])
        ax_input.imshow(img_input)

        # Wrap long prompts to avoid overlap.
        raw_prompt = prompts[col_idx]
        wrapped_prompt = textwrap.fill(f'"{raw_prompt}"', width=40)

        if col_idx == 0:
            ax_input.set_ylabel("Original Input", fontsize=16, fontweight="bold", labelpad=20)

        ax_input.set_title(f"Image {col_idx + 1}", fontsize=14, pad=10)
        ax_input.set_xlabel(wrapped_prompt, fontsize=16, style="italic", labelpad=10)
        ax_input.set_xticks([])
        ax_input.set_yticks([])

        # Rows 1…N: one row per model.
        for row_offset, model_name in enumerate(model_names):
            row_idx = row_offset + 1
            ax_model = axes[row_idx, col_idx]

            # Use the original (un-sliced) index so --start is handled correctly.
            try:
                img_path = outputs[model_name][col_idx + args.start]
                img_model = load_image(img_path)
            except IndexError:
                print(
                    f"Warning: Missing output for model '{model_name}' at index {col_idx + args.start}."
                )
                img_model = Image.new("RGB", (512, 512), color="white")

            ax_model.imshow(img_model)

            if col_idx == 0:
                ax_model.set_ylabel(model_name, fontsize=18, fontweight="bold", labelpad=20)

            ax_model.set_xticks([])
            ax_model.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, hspace=0.1, wspace=0.05)

    print(f"Saving comparison grid to {args.output}...")
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()

    print("Done!")


if __name__ == "__main__":
    main()
