"""
Generate synthetic defect images with a fine-tuned InstructPix2Pix (ip2p) model.

For every <object, defect_type> pair in the input JSON this script:
  1. Pulls clean images from the MVTec ``train/good`` folder
     (cycles through them when you ask for more images than exist).
  2. Applies the simple defect prompt for that defect type with the ip2p model
     (reuses ``generate._load_ip2p`` + ``generate.generate_image``).
  3. Writes the results under::

         <output_dir>/<object>/test/<defect_type>/<000.png ...>
         <output_dir>/<object>/ground_truth/<defect_type>/<000_mask.png ...>

      Both subtrees mirror the MVTec layout, so the output can be fed straight
      into ``eval/compute_kid.py`` (whose ``--real_path`` points at the real
      mvtec tree and whose ``--generated_path`` points at ``<output_dir>``).
      Masks are a binarized version (Otsu) of the heatmap diff used in
      ``generate.py``: ``mean(|generated - input|)`` over RGB channels.

Usage
-----
    uv run eval/generate_defects.py \
        --device cuda:1 \
        --weights_path /path/to/finetuned/unet \
        --input_json eval/defect_prompts.json \
        --output_dir /home/luca_piai/big_disk/datasets/generated \
        --num_images 20
"""

import argparse
import json
import os
import sys

import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Make the repo root importable so we can reuse the pipeline helpers from
# generate.py (run as:  uv run eval/generate_defects.py ...).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from generate import _load_ip2p, generate_image  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
MVTEC_ROOT = "/home/luca_piai/big_disk/datasets/mvtec"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic defect images with a fine-tuned ip2p model."
    )
    parser.add_argument("--device", type=str, required=True, help='e.g. "cuda:1"')
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the fine-tuned ip2p UNet checkpoint directory.",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="eval/defect_prompts.json",
        help="JSON mapping <object> -> {<defect_type>: <prompt>}.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root dir for generated images (creates <object>/test/<defect_type>/).",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=20,
        help="Number of images to generate per defect type.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of images generated per pipeline call (batched inference). "
        "Increase to speed up generation if GPU memory allows.",
    )
    parser.add_argument(
        "--mvtec_path",
        type=str,
        default=MVTEC_ROOT,
        help="MVTec root; clean images from <mvtec_path>/<cat>/train/good/.",
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Skip binary mask (Otsu diff) computation and saving.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Compile the pipeline's UNet (and VAE) with torch.compile before "
        "generation. The first generation pays the (multi-second) compile cost, "
        "then repeated inference is faster. Off by default to avoid surprises.",
    )
    return parser.parse_args()


def list_clean_images(mvtec_path, object_name):
    good_dir = os.path.join(mvtec_path, object_name, "train", "good")
    if not os.path.isdir(good_dir):
        return []
    return [
        os.path.join(good_dir, f)
        for f in sorted(os.listdir(good_dir))
        if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(good_dir, f))
    ]


def compute_defect_mask(input_path, generated_pil):
    """Binary defect mask (PIL 'L', uint8 0/255) via Otsu on the diff.

    Mirrors the heatmap in ``generate.py``: the difference is the mean
    absolute pixel change over RGB channels between the generated image and
    the (resized to 512x512) input, normalized to [0, 1]. Otsu's method then
    binarizes that diff into a foreground-defect mask.
    """
    input_img = Image.open(input_path).convert("RGB").resize((512, 512))
    gen_img = generated_pil.convert("RGB").resize((512, 512))
    input_np = np.array(input_img, dtype=np.float32) / 255.0
    gen_np = np.array(gen_img, dtype=np.float32) / 255.0
    diff = np.abs(gen_np - input_np).mean(axis=2)  # (512, 512) in [0, 1]
    diff_uint8 = np.clip(diff * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(mask, mode="L")


def _apply_torch_compile(pipe):
    """Compile the heavy pipeline components with ``torch.compile``.

    Compiling speeds up repeated inference (the first call pays the multi-second
    compile cost). Each component is wrapped defensively so a failure on one
    (e.g. an unsupported op) doesn't abort the whole run.
    """
    for comp_name in ("unet", "vae"):
        comp = getattr(pipe, comp_name, None)
        if comp is None:
            continue
        try:
            setattr(pipe, comp_name, torch.compile(comp, mode="default"))
            print(f"[compile] torch.compile applied to pipe.{comp_name}")
        except Exception as e:  # pragma: no cover - environment dependent
            print(
                f"[compile] WARNING: could not compile pipe.{comp_name}: {e}. "
                f"Continuing without it."
            )
    return pipe


def _warmup(pipe, args, prompts):
    """Run one throwaway generation so torch.compile triggers (and any errors
    surface) *before* the timed generation loop.

    Picks the first object that has clean images and its first defect prompt.
    The result is discarded; nothing is written to disk.
    """
    for object_name, defects in prompts.items():
        clean_images = list_clean_images(args.mvtec_path, object_name)
        if not clean_images:
            continue
        defect_type, prompt = next(iter(defects.items()))
        print(f"[warmup] compiling on {object_name}/{defect_type} ...")
        try:
            generate_image(pipe, clean_images[0], prompt, 20)
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[warmup] WARNING: warm-up generation failed ({e}); continuing.")
        break


def main():
    args = parse_args()

    # Free convolution/matmul throughput wins (no accuracy impact at bf16).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    with open(args.input_json, "r") as f:
        prompts = json.load(f)

    pipe = _load_ip2p(args.weights_path, args.device)

    if args.torch_compile:
        print("[compile] Enabling torch.compile for the pipeline components...")
        pipe = _apply_torch_compile(pipe)
        _warmup(pipe, args, prompts)

    for object_name, defects in prompts.items():
        clean_images = list_clean_images(args.mvtec_path, object_name)
        if not clean_images:
            print(f"[skip] No clean images found for {object_name} under train/good/.")
            continue

        for defect_type, prompt in defects.items():
            out_dir = os.path.join(args.output_dir, object_name, "test", defect_type)
            os.makedirs(out_dir, exist_ok=True)
            mask_dir = os.path.join(
                args.output_dir, object_name, "ground_truth", defect_type
            )
            if not args.no_mask:
                os.makedirs(mask_dir, exist_ok=True)

            # Resume support: if images were already generated for this
            # <object>/<defect_type>, continue from the index right after the
            # highest existing one instead of overwriting. This lets an
            # interrupted run pick up where it left off (total target stays
            # ``args.num_images``).
            existing_idx = [
                int(os.path.splitext(f)[0])
                for f in os.listdir(out_dir)
                if f.lower().endswith(IMAGE_EXTS)
                and os.path.isfile(os.path.join(out_dir, f))
                and os.path.splitext(f)[0].isdigit()
            ]
            start_idx = (max(existing_idx) + 1) if existing_idx else 0
            n_new = max(0, args.num_images - start_idx)

            print(f"[{object_name}/{defect_type}] prompt={prompt}")
            if start_idx > 0:
                print(
                    f"  resuming from index {start_idx} "
                    f"({n_new} new image(s) to reach {args.num_images})"
                )
            print(f"  -> {n_new} new image(s) into {out_dir}")

            # Precompute the text-conditioning ONCE per defect (every image for a
            # defect shares the same prompt) so we don't re-run the text encoder
            # for every single image. The embeddings are then broadcast across the
            # batch via .repeat().
            enc = pipe._encode_prompt(
                prompt,
                device=args.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            prompt_embeds, neg_embeds = enc[0], enc[1]

            indices = list(range(start_idx, args.num_images))
            for b_start in tqdm(
                range(0, len(indices), args.batch_size),
                desc=f"{object_name}/{defect_type}",
                leave=False,
            ):
                batch_idx = indices[b_start : b_start + args.batch_size]
                B = len(batch_idx)
                # Load + resize the clean inputs (mirrors generate_image's
                # preprocessing) for this batch.
                clean_paths = [clean_images[i % len(clean_images)] for i in batch_idx]
                batch_pils = [
                    Image.open(p).convert("RGB").resize((512, 512)) for p in clean_paths
                ]
                out = pipe(
                    prompt_embeds=prompt_embeds.repeat(B, 1, 1),
                    negative_prompt_embeds=neg_embeds.repeat(B, 1, 1),
                    image=batch_pils,
                    num_inference_steps=20,
                ).images
                for j, i in enumerate(batch_idx):
                    generated = out[j]
                    gen_path = os.path.join(out_dir, f"{i:03d}.png")
                    generated.save(gen_path)

                    if not args.no_mask:
                        mask = compute_defect_mask(clean_paths[j], generated)
                        mask.save(os.path.join(mask_dir, f"{i:03d}_mask.png"))

            print(f"  saved {n_new} new image(s) to {out_dir}")


if __name__ == "__main__":
    main()
