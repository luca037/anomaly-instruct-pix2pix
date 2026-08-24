"""
Unified image generation script — supports multiple model backends.

Currently supported backends (selectable via ``--backend``):

  • **ip2p**  – InstructPix2Pix (``timbrooks/instruct-pix2pix``)
                Pass a fine-tuned UNet directory with ``--weights_path``
                or omit it for the vanilla baseline.

  • **flux2** – FLUX.2 Klein 4B (``black-forest-labs/FLUX.2-klein-4B``)
                Pass a LoRA weights directory with ``--weights_path``
                or omit it for the vanilla baseline.

Adding a new backend is straightforward: define a loader function and
register it in the ``BACKENDS`` dictionary.

This script provides two subcommands:

  1. compare  – Generate images for model comparison (one model per run).
                Generation is incremental: pairs that already have an output
                on disk are skipped, so you can add new prompts to the test
                set without re-generating everything.

  2. heatmap  – Generate N images with a single model and (optionally) plot
                a difference-heatmap grid.  Two mutually exclusive modes:
                  • --test_set : batch mode, iterates over test_set.json
                  • --prompt   : quick one-off with a CLI prompt and a
                                 predefined default image for the category.

Usage examples
==============

# ── compare (InstructPix2Pix) ────────────────────────────────────────
# Fine-tuned model:
    uv run generate.py compare \\
        --backend ip2p \\
        --test_set ./data_preparation/test_set.json \\
        --category hazelnut \\
        --model_id mvtec_7 \\
        --weights_path /path/to/checkpoint/unet \\
        --device cuda:1

# Vanilla baseline (omit --weights_path):
    uv run generate.py compare \\
        --backend ip2p \\
        --test_set ./data_preparation/test_set.json \\
        --category pill \\
        --model_id vanilla \\
        --device cuda:1

# ── compare (FLUX.2 Klein) ───────────────────────────────────────────
    uv run generate.py compare \\
        --backend flux2 \\
        --test_set ./data_preparation/test_set.json \\
        --category hazelnut \\
        --model_id flux2_lora_v1 \\
        --weights_path /path/to/lora/weights \\
        --device cuda:1

# ── heatmap ──────────────────────────────────────────────────────────
# Batch mode – all pairs from the test set:
    uv run generate.py heatmap \\
        --backend ip2p \\
        --test_set ./data_preparation/test_set.json \\
        --category hazelnut \\
        --weights_path /path/to/checkpoint/unet \\
        --device cuda:1

# Quick test – single prompt, default image, FLUX.2 backend:
    uv run generate.py heatmap \\
        --backend flux2 \\
        --prompt "add a crack" \\
        --category hazelnut \\
        --weights_path /path/to/lora/weights \\
        --device cuda:1

# Disable the heatmap row:
    uv run generate.py heatmap \\
        --backend ip2p \\
        --prompt "add a crack" \\
        --category hazelnut \\
        --device cuda:1 \\
        --no-heatmap
"""

import argparse
import json
import os
import textwrap

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler,
)
from tqdm import tqdm


###
### Constants
###

# Default input images used by the "heatmap --prompt" mode.
# Each category maps to a single representative "good" image that is
# used when you just want to quickly test a prompt from the CLI without
# looking up the exact path every time.
DEFAULT_IMAGES = {
    # MVTec dataset
    "bottle": "/home/luca_piai/big_disk/datasets/mvtec/bottle/test/good/001.png",
    "cable": "/home/luca_piai/big_disk/datasets/mvtec/cable/test/good/001.png",
    "capsule": "/home/luca_piai/big_disk/datasets/mvtec/capsule/test/good/001.png",
    "carpet": "/home/luca_piai/big_disk/datasets/mvtec/carpet/test/good/001.png",
    "grid": "/home/luca_piai/big_disk/datasets/mvtec/grid/test/good/001.png",
    "hazelnut": "/home/luca_piai/big_disk/datasets/mvtec/hazelnut/test/good/011.png",
    "leather": "/home/luca_piai/big_disk/datasets/mvtec/leather/test/good/001.png",
    "metal_nut": "/home/luca_piai/big_disk/datasets/mvtec/metal_nut/test/good/001.png",
    "pill": "/home/luca_piai/big_disk/datasets/mvtec/pill/test/good/013.png",
    "screw": "/home/luca_piai/big_disk/datasets/mvtec/screw/test/good/001.png",
    "tile": "/home/luca_piai/big_disk/datasets/mvtec/tile/test/good/001.png",
    "toothbrush": "/home/luca_piai/big_disk/datasets/mvtec/toothbrush/test/good/001.png",
    "transistor": "/home/luca_piai/big_disk/datasets/mvtec/transistor/test/good/003.png",
    "wood": "/home/luca_piai/big_disk/datasets/mvtec/wood/test/good/001.png",
    "zipper": "/home/luca_piai/big_disk/datasets/mvtec/zipper/test/good/001.png",
}

# Base directory where "compare" stores generated images.
# Each model gets its own sub-folder: ./output/<model_id>/
OUTPUT_BASE_PATH = "./output/"


###
### Backend registry
###
#
# To add a new model backend:
#   1. Write a loader function:  def _load_<name>(weights_path, device) -> pipeline
#   2. Register it:              BACKENDS["<name>"] = {"loader": _load_<name>, "description": "..."}
#
# The loader receives *weights_path* (str | None) and *device* (str).
# It must return a pipeline object whose __call__ accepts (prompt=, image=).
#


def _load_ip2p(weights_path, device):
    """Load the InstructPix2Pix pipeline.

    Args:
        weights_path: Path to a fine-tuned UNet directory, or None for the
                      vanilla InstructPix2Pix baseline.
        device: Torch device string (e.g. "cuda:1").

    Returns:
        A ready-to-use StableDiffusionInstructPix2PixPipeline.
    """
    base_model_id = "timbrooks/instruct-pix2pix"
    dtype = torch.bfloat16

    if weights_path is None:
        print("[ip2p] Loading vanilla InstructPix2Pix baseline...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        print(f"[ip2p] Loading custom UNet from {weights_path}...")
        trained_unet = UNet2DConditionModel.from_pretrained(
            weights_path, torch_dtype=dtype,
        )
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            base_model_id,
            unet=trained_unet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
    )
    pipe.to(device)
    return pipe


def _load_flux2(weights_path, device):
    """Load the FLUX.2 Klein 4B pipeline.

    Args:
        weights_path: Path to a LoRA weights directory/safetensors file,
                      or None for the vanilla FLUX.2 Klein baseline.
        device: Torch device string (e.g. "cuda:1").

    Returns:
        A ready-to-use Flux2KleinPipeline.
    """
    from diffusers import Flux2KleinPipeline

    base_model_id = "black-forest-labs/FLUX.2-klein-4B"
    dtype = torch.bfloat16

    print("[flux2] Loading base FLUX.2 Klein pipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    )

    if weights_path is None:
        print("[flux2] Using vanilla baseline (no LoRA).")
    else:
        print(f"[flux2] Injecting LoRA weights from {weights_path}...")
        pipe.load_lora_weights(weights_path)
        pipe.fuse_lora()

    pipe.to(device)
    return pipe


# ── Registry ──────────────────────────────────────────────────────────
# Each entry maps a backend name to its loader function and a short
# human-readable description shown in --help.
BACKENDS = {
    "ip2p": {
        "loader": _load_ip2p,
        "description": "InstructPix2Pix  (--weights_path = UNet dir)",
    },
    "flux2": {
        "loader": _load_flux2,
        "description": "FLUX.2 Klein 4B  (--weights_path = LoRA dir)",
    },
}


def load_pipeline(backend, weights_path, device):
    """Load a diffusion pipeline by backend name.

    Args:
        backend: Key into the BACKENDS registry (e.g. "ip2p", "flux2").
        weights_path: Backend-specific path to fine-tuned weights, or None
                      for the vanilla baseline.
        device: Torch device string.

    Returns:
        A ready-to-use pipeline.
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Available: {list(BACKENDS.keys())}"
        )
    loader_fn = BACKENDS[backend]["loader"]
    return loader_fn(weights_path, device)


###
### Shared helpers
###


def generate_image(pipe, image_path, prompt, steps):
    """Generate a single edited image from an input image and a prompt.

    The input is resized to 512×512 before being passed to the pipeline.
    Each backend uses its own default inference parameters.

    Args:
        pipe: A loaded pipeline (any backend).
        image_path: Path to the source image on disk.
        prompt: The editing instruction (e.g. "add a crack").
        steps: The number of inference steps to use.
    Returns:
        A PIL Image with the edited result.
    """
    init_image = Image.open(image_path).convert("RGB").resize((512, 512))
    output = pipe(
        prompt=prompt,
        image=init_image,
        num_inference_steps=steps,
    ).images[0]
    return output


def plot_grid(image_paths, prompts, output_filename="grid.png",
              use_heatmap=False):
    """Plot images in a grid with an optional heatmap row.

    Row 1: The images (Input + Generated outputs).
    Row 2 (optional): Difference heatmaps highlighting what changed.

    Args:
        image_paths: List of paths. First element is the original input;
                     the rest are generated images.
        prompts: Prompts used for the generated images.
        output_filename: Where to save the final grid image.
        use_heatmap: Whether to add a second row with heatmaps.
    """
    num_images = len(image_paths)
    if num_images == 0:
        print("Error: No images provided.")
        return

    num_rows = 2 if use_heatmap else 1
    fig, axes = plt.subplots(
        num_rows, num_images,
        figsize=(num_images * 5, 5 * num_rows + (2 if use_heatmap else 1)),
        squeeze=False,
    )

    input_img_np = None

    for i, img_path in enumerate(image_paths):
        ax_img = axes[0, i]

        # Safely load the image
        try:
            img = Image.open(img_path).convert("RGB").resize((512, 512))
            img_np = np.array(img).astype(np.float32) / 255.0
        except Exception as e:
            print(f"Warning: Could not load {img_path}. "
                  f"Using a placeholder. ({e})")
            img = Image.new("RGB", (512, 512), color="lightgray")
            img_np = np.zeros((512, 512, 3), dtype=np.float32)

        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # First column is always the original input image.
        if i == 0:
            input_img_np = img_np  # cache for heatmap subtraction later
            ax_img.set_title("Input Image", fontsize=16, fontweight="bold",
                             pad=15)
            if use_heatmap:
                # Show a blank (all-zero) heatmap under the input as
                # a visual anchor for the row.
                ax_heat = axes[1, i]
                ax_heat.imshow(np.zeros((512, 512)), cmap="jet",
                               vmin=0, vmax=1)
                ax_heat.set_xticks([])
                ax_heat.set_yticks([])
        else:
            # Generated output columns.
            ax_img.set_title(f"Generated Image {i}", fontsize=14, pad=15)

            # prompts list is offset by 1 because index 0 is the input.
            prompt_idx = i - 1
            wrapped_prompt = ""
            if prompt_idx < len(prompts):
                raw_prompt = prompts[prompt_idx]
                wrapped_prompt = textwrap.fill(f'"{raw_prompt}"', width=45)

            if use_heatmap:
                ax_heat = axes[1, i]
                # Mean absolute pixel difference across RGB channels.
                # 0 = identical to input, higher = more change.
                diff = np.abs(img_np - input_img_np).mean(axis=2)
                ax_heat.set_title("Heatmap", fontsize=14, pad=15)
                ax_heat.imshow(
                    diff, cmap="jet", vmin=0,
                    vmax=np.max(diff) if np.max(diff) > 0 else 1,
                )
                ax_heat.set_xticks([])
                ax_heat.set_yticks([])

            # Always place the prompt below the generated image (first row).
            if wrapped_prompt:
                ax_img.set_xlabel(wrapped_prompt, fontsize=12,
                                  style="italic", labelpad=12)

    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.15 if use_heatmap else 0.2,
        hspace=0.2 if use_heatmap else 0,
    )
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grid saved to {output_filename}")


###
### Subcommand: compare
###

def cmd_compare(args):
    """Generate images for model comparison (incremental).

    This function:
      1. Reads the test set JSON (source of truth for inputs/prompts).
      2. Loads (or creates) results.json and syncs the current inputs/prompts.
      3. Checks which (image, prompt) pairs already have a generated output
         on disk for the given model_id.
      4. Generates only the missing ones and updates results.json.
    """

    # Load test set (source of truth for inputs/prompts)
    with open(args.test_set, "r") as f:
        test_set = json.load(f)

    if args.category not in test_set:
        raise ValueError(
            f"Category '{args.category}' not found in {args.test_set}. "
            f"Available: {list(test_set.keys())}"
        )

    inputs = test_set[args.category]["inputs"]
    prompts = test_set[args.category]["prompts"]

    if len(inputs) != len(prompts):
        raise ValueError("'inputs' and 'prompts' must be the same length.")

    # Load existing results.json if it exists, otherwise start fresh.
    results_path = args.results
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Always overwrite inputs/prompts from the test set so that newly
    # added prompts are picked up without manual JSON editing.
    if args.category not in results:
        results[args.category] = {}
    results[args.category]["inputs"] = inputs
    results[args.category]["prompts"] = prompts

    if "outputs" not in results[args.category]:
        results[args.category]["outputs"] = {}

    # Get (or initialize) the output list for this model.
    # Pad with None so its length matches the current number of pairs;
    # this handles the case where new prompts were added to the test set.
    model_outputs = results[args.category]["outputs"].get(args.model_id, [])
    while len(model_outputs) < len(inputs):
        model_outputs.append(None)

    # Build the list of indices that still need generation:
    # either the slot is empty (None) or the file was deleted from disk.
    to_generate = [
        i for i in range(len(inputs))
        if model_outputs[i] is None or not os.path.exists(model_outputs[i])
    ]

    if not to_generate:
        print("All pairs already generated. Nothing to do.")
        # Persist anyway: the inputs/prompts might have been updated
        # even if no new images needed generating.
        results[args.category]["outputs"][args.model_id] = model_outputs
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        return

    print(f"Need to generate {len(to_generate)}/{len(inputs)} images.")

    # Load the (heavy) diffusion pipeline only when there is actual work.
    pipe = load_pipeline(args.backend, args.weights_path, args.device)

    output_dir = os.path.join(OUTPUT_BASE_PATH, args.model_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")

    # Generate each missing pair and record the output path.
    for i in tqdm(to_generate, desc="Generating"):
        file_name = f"{args.category}_{i:03d}.png"
        save_path = os.path.join(os.path.abspath(output_dir), file_name)

        output_image = generate_image(pipe, inputs[i], prompts[i], args.steps)
        output_image.save(save_path)
        model_outputs[i] = save_path

    # Write the updated results back to disk.
    results[args.category]["outputs"][args.model_id] = model_outputs
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Done! Results saved to {results_path}")


###
### Subcommand: heatmap
###

def cmd_heatmap(args):
    """Generate images + optional heatmap grids.

    Two modes (mutually exclusive via argparse):
      • --test_set : iterate over every (image, prompt) pair in the JSON.
      • --prompt   : use a single CLI prompt with the default image for
                     the chosen category (see DEFAULT_IMAGES).

    For each pair, *num_images* variants are generated and saved.  Then a
    grid plot is created with the input on the left and the outputs to its
    right, optionally with a heatmap row underneath.
    """

    # Resolve the list of (input_image_path, prompt) pairs depending on
    # which mode the user selected.
    if args.test_set:
        with open(args.test_set, "r") as f:
            test_set = json.load(f)
        if args.category not in test_set:
            raise ValueError(
                f"Category '{args.category}' not found in {args.test_set}. "
                f"Available: {list(test_set.keys())}"
            )
        pairs = list(zip(
            test_set[args.category]["inputs"],
            test_set[args.category]["prompts"],
        ))
    elif args.prompt:
        if args.category not in DEFAULT_IMAGES:
            raise ValueError(
                f"No default image for category '{args.category}'. "
                f"Available: {list(DEFAULT_IMAGES.keys())}"
            )
        pairs = [(DEFAULT_IMAGES[args.category], args.prompt)]
    else:
        raise ValueError("Provide either --test_set or --prompt.")

    # Load the pipeline once and reuse it for every pair.  This is the
    # main advantage over the old tmp.py approach which spawned a new
    # process (and re-loaded the model) for each pair.
    pipe = load_pipeline(args.backend, args.weights_path, args.device)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for pair_idx, (input_path, prompt) in enumerate(pairs):
        print(f"\n[{pair_idx + 1}/{len(pairs)}] prompt='{prompt}'")

        img_paths = []
        tag = prompt.replace(" ", "_")  # filesystem-safe version of prompt

        # Generate N variant images for this (input, prompt) pair.
        for i in range(args.num_images):
            output_image = generate_image(pipe, input_path, prompt, args.steps)
            out_name = f"{args.category}_{tag}_{pair_idx}_{i}.png"
            out_path = os.path.join(output_dir, out_name)
            output_image.save(out_path)
            img_paths.append(out_path)

        # Create the comparison grid: [input, gen_0, gen_1, …]
        grid_name = f"{args.category}_{tag}_{pair_idx}_grid.jpg"
        grid_path = os.path.join(output_dir, grid_name)

        plot_grid(
            [input_path] + img_paths,
            [prompt] * len(img_paths),
            output_filename=grid_path,
            use_heatmap=args.heatmap,
        )

    print("\nAll done!")


###
### CLI
###

def _add_common_args(parser):
    """Add arguments shared by all subcommands (backend, weights, device)."""
    backends_help = ", ".join(
        f"{name} ({info['description']})" for name, info in BACKENDS.items()
    )
    parser.add_argument(
        "--backend", type=str, required=True,
        choices=list(BACKENDS.keys()),
        help=f"Model backend to use. Available: {backends_help}",
    )
    parser.add_argument(
        "--weights_path", type=str, default=None,
        help="Path to fine-tuned weights (UNet dir for ip2p, LoRA dir for "
             "flux2). Omit for the vanilla baseline.",
    )
    parser.add_argument(
        "--device", type=str, required=True,
        help="CUDA device (e.g. cuda:1)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with multiple model backends.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── compare ───────────────────────────────────────────────────────
    p_cmp = subparsers.add_parser(
        "compare",
        help="Generate images for model comparison (incremental).",
    )
    _add_common_args(p_cmp)
    p_cmp.add_argument("--test_set", type=str, required=True,
                       help="Path to test_set.json")
    p_cmp.add_argument("--results", type=str, default="results.json",
                       help="Path to results.json (default: results.json)")
    p_cmp.add_argument("--category", type=str, required=True,
                       help="Object category (e.g. hazelnut, pill)")
    p_cmp.add_argument("--steps", type=int, default=20,
                        help="Number of inference steps (default: 20)")
    p_cmp.add_argument("--model_id", type=str, required=True,
                       help="Name for this model run")
    p_cmp.set_defaults(func=cmd_compare)

    # ── heatmap ───────────────────────────────────────────────────────
    p_heat = subparsers.add_parser(
        "heatmap",
        help="Generate images + optional heatmap grid.",
    )
    _add_common_args(p_heat)
    mode = p_heat.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test_set", type=str,
                      help="Path to test_set.json (batch mode)")
    mode.add_argument("--prompt", type=str,
                      help="Single prompt (quick-test mode)")

    p_heat.add_argument("--category", type=str, required=True,
                        help="Object category (e.g. hazelnut, pill)")
    p_heat.add_argument("--output_dir", type=str,
                        default="./output_heatmap/",
                        help="Output directory (default: ./output_heatmap/)")
    p_heat.add_argument("--num_images", type=int, default=3,
                        help="Images to generate per prompt (default: 3)")
    p_heat.add_argument("--steps", type=int, default=20,
                        help="Number of inference steps (default: 20)")
    p_heat.add_argument("--heatmap", action="store_true", default=True,
                        help="Enable heatmap row (default: on)")
    p_heat.add_argument("--no-heatmap", action="store_false", dest="heatmap",
                        help="Disable heatmap row")
    p_heat.set_defaults(func=cmd_heatmap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
