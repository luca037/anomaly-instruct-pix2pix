"""
CLIP-based quality filter for generated synthetic defects.

Adapted from MIRAGE's ``src/CLIP-selection/CLIP.py``. Scans a generated
dataset (``<generated>/<obj>/test/<defect>/<i>.png``), pairs each generated
image with its clean MVTec source, and runs OpenCLIP to decide whether the
anomaly was actually rendered. Bad indices are collected in a single JSON
file:

    <generated_data_path>/clip_bad.json
    {
        "hazelnut/crack": ["001", "005"],
        "hazelnut/cut": ["003"]
    }

Training loaders can read this file and skip the listed indices; no images or
masks are moved or copied.

Run filtering:
    uv run eval/clip_filter.py \
        --device cuda:1 \
        --generated_data_path /home/luca_piai/big_disk/datasets/generated \
        --mvtec_path /home/luca_piai/big_disk/datasets/mvtec \
        --input_json eval/defect_prompts.json

Run overview (no CLIP needed):
    uv run eval/clip_filter.py --overview \
        --generated_data_path /home/luca_piai/big_disk/datasets/generated
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _import_open_clip():
    """Lazy import so ``--overview`` can run without open_clip installed."""
    try:
        import open_clip

        return open_clip
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "clip_filter.py filtering mode requires open_clip. "
            "Install it with:\n    uv pip install open-clip-torch"
        ) from exc
DEFAULT_MODEL = "ViT-bigG-14"
DEFAULT_PRETRAINED = "laion2b_s39b_b160k"


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP-based generation quality filter for synthetic defects."
    )
    parser.add_argument("--device", type=str, default="cuda:1", help="e.g. cuda:1")
    parser.add_argument(
        "--generated_data_path",
        type=str,
        required=True,
        help="Root of the generated dataset (output of generate_defects.py).",
    )
    parser.add_argument(
        "--mvtec_path",
        type=str,
        default="/home/luca_piai/big_disk/datasets/mvtec",
        help="MVTec root; clean sources from <mvtec>/<obj>/train/good/.",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="eval/defect_prompts.json",
        help="JSON mapping <object> -> {<defect>: <prompt>}.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output bad-index JSON. Defaults to <generated_data_path>/clip_bad.json.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="OpenCLIP model name."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=DEFAULT_PRETRAINED,
        help="OpenCLIP pretrained weights tag.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["cable", "screw", "transistor", "leather", "hazelnut", "pill", "tile", "carpet", "capsule", "wood", "metal_nut"],
        help="Limit filtering to these object categories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate indices already listed in the output JSON.",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Only print a summary of the output JSON; do not load CLIP or "
        "score any images.",
    )
    return parser.parse_args()


def setup_model(model_name, pretrained, device, seed=12345):
    """Load OpenCLIP model, tokenizer and image preprocess."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    open_clip = _import_open_clip()
    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def clip_score(
    img1_path, img2_path, prompt1, prompt2, model, preprocess, tokenizer, device
):
    """Return CLIP softmax probabilities for (img1, img2) vs (prompt1, prompt2).

    Layout:
      img1  = generated anomaly image
      img2  = clean normal image
      prompt1 = anomaly prompt
      prompt2 = normal prompt

    Returns ``(probs_per_image, probs_per_text)`` where:
      probs_per_image[direction=0] -> for each text, which image wins.
      probs_per_text[direction=1]  -> for each image, which text wins.
    """
    img1 = preprocess(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(device)
    images = torch.cat([img1, img2], dim=0)
    text_tokens = tokenizer([prompt1, prompt2]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        logits = similarity * model.logit_scale.exp()
        probs_per_image = logits.softmax(dim=0)
        probs_per_text = logits.softmax(dim=1)

    return probs_per_image, probs_per_text


def is_bad_generation(p, p1):
    """MIRAGE criterion.

    The anomaly image must align with the anomaly prompt (and vice versa).
    """
    return bool(
        (p[0][0] < p[1][0]) and (p[1][1] < p[0][1]) and (p1[0][0] < p1[0][1])
    )


def list_clean_images(mvtec_path, object_name):
    """Sorted list of clean MVTec train/good images.

    Same sort order that ``generate_defects.py`` uses, so the index-to-source
    mapping stays consistent.
    """
    good_dir = os.path.join(mvtec_path, object_name, "train", "good")
    if not os.path.isdir(good_dir):
        return []
    return sorted(
        os.path.join(good_dir, f)
        for f in os.listdir(good_dir)
        if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(good_dir, f))
    )


def list_generated_indices(test_dir):
    """Return generated image filenames sorted by their numeric index."""
    if not os.path.isdir(test_dir):
        return []
    files = [
        f
        for f in os.listdir(test_dir)
        if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(test_dir, f))
    ]

    def _key(fname):
        base = os.path.splitext(fname)[0]
        try:
            return int(base)
        except ValueError:
            return base

    return sorted(files, key=_key)


def run_overview(args):
    """Print a per-object, per-defect summary of the bad-index JSON."""
    if args.output_json is None:
        args.output_json = os.path.join(args.generated_data_path, "clip_bad.json")

    bad = {}
    if os.path.isfile(args.output_json):
        with open(args.output_json, "r") as f:
            bad = json.load(f)

    print(f"CLIP filter overview: {args.output_json}\n")
    grand_bad = 0
    grand_total = 0
    for obj_name in args.categories:
        obj_gen_root = os.path.join(args.generated_data_path, obj_name, "test")
        if not os.path.isdir(obj_gen_root):
            continue

        defects = sorted(
            d
            for d in os.listdir(obj_gen_root)
            if os.path.isdir(os.path.join(obj_gen_root, d))
        )
        if not defects:
            continue

        print(obj_name)
        obj_bad = 0
        obj_total = 0
        for defect in defects:
            key = f"{obj_name}/{defect}"
            total = len(list_generated_indices(os.path.join(obj_gen_root, defect)))
            n_bad = len(bad.get(key, []))
            obj_bad += n_bad
            obj_total += total
            pct = (100.0 * n_bad / total) if total > 0 else 0.0
            print(f"  {defect:12s}: {n_bad:4d} bad / {total:4d} total ({pct:5.1f}%)")

        obj_pct = (100.0 * obj_bad / obj_total) if obj_total > 0 else 0.0
        print(
            f"  {'object total':12s}: {obj_bad:4d} bad / {obj_total:4d} total "
            f"({obj_pct:5.1f}%)\n"
        )
        grand_bad += obj_bad
        grand_total += obj_total

    grand_pct = (100.0 * grand_bad / grand_total) if grand_total > 0 else 0.0
    print(
        f"grand total: {grand_bad} bad / {grand_total} total ({grand_pct:.1f}%)"
    )


def main():
    args = parse_args()
    if args.output_json is None:
        args.output_json = os.path.join(args.generated_data_path, "clip_bad.json")

    if args.overview:
        run_overview(args)
        return

    with open(args.input_json, "r") as f:
        prompts = json.load(f)

    # Load existing bad-index set, if any, to support resumable filtering.
    bad = {}
    if os.path.isfile(args.output_json) and not args.overwrite:
        with open(args.output_json, "r") as f:
            bad = json.load(f)
        n_existing = sum(len(v) for v in bad.values())
        print(f"[resume] loaded {args.output_json} ({n_existing} bad entries)")

    print(f"[model] loading {args.model}/{args.pretrained} on {args.device} ...")
    model, preprocess, tokenizer = setup_model(
        args.model, args.pretrained, args.device
    )

    objects = (
        args.categories
        if args.categories
        else sorted(
            d
            for d in os.listdir(args.generated_data_path)
            if os.path.isdir(os.path.join(args.generated_data_path, d))
        )
    )

    for obj_name in objects:
        if obj_name not in prompts:
            print(f"[skip] {obj_name}: not in {args.input_json}")
            continue

        obj_prompts = prompts[obj_name]
        obj_gen_root = os.path.join(args.generated_data_path, obj_name, "test")
        if not os.path.isdir(obj_gen_root):
            print(f"[skip] {obj_name}: no generated test/ dir")
            continue

        clean_images = list_clean_images(args.mvtec_path, obj_name)
        if not clean_images:
            print(f"[skip] {obj_name}: no clean images in MVTec train/good")
            continue
        n_clean = len(clean_images)

        defects = sorted(
            d
            for d in os.listdir(obj_gen_root)
            if os.path.isdir(os.path.join(obj_gen_root, d))
        )
        for defect in defects:
            if defect not in obj_prompts:
                print(f"[skip] {obj_name}/{defect}: no prompt in {args.input_json}")
                continue

            key = f"{obj_name}/{defect}"
            existing_bad = set(bad.get(key, []))

            anomaly_fragment = obj_prompts[defect].replace("add ", "")
            prompt_anomaly = (
                f"This is a damaged {obj_name} image with {anomaly_fragment}."
            )
            prompt_normal = (
                f"This is an intact {obj_name} image without any damage."
            )

            print(prompt_anomaly)

            test_dir = os.path.join(obj_gen_root, defect)
            files = list_generated_indices(test_dir)

            new_bad = []
            for fname in tqdm(files, desc=f"CLIP {key}", leave=False):
                idx = os.path.splitext(fname)[0]
                if not args.overwrite and idx in existing_bad:
                    continue

                gen_path = os.path.join(test_dir, fname)
                # Re-derive the clean source using the same deterministic mapping
                # generate_defects.py uses: clean_images[i % len(clean_images)].
                clean_path = clean_images[int(idx) % n_clean]

                p, p1 = clip_score(
                    gen_path,
                    clean_path,
                    prompt_anomaly,
                    prompt_normal,
                    model,
                    preprocess,
                    tokenizer,
                    args.device,
                )
                if is_bad_generation(p, p1):
                    new_bad.append(idx)
                    print(f"[BAD] {key}/{idx}")

            if new_bad:
                merged = sorted(
                    set(bad.get(key, []) + new_bad),
                    key=lambda x: int(x) if x.isdigit() else x,
                )
                bad[key] = merged
                with open(args.output_json, "w") as f:
                    json.dump(bad, f, indent=2)
                print(f"  {key}: +{len(new_bad)} bad -> {len(merged)} total")

    print(f"[done] wrote {args.output_json}")


if __name__ == "__main__":
    main()
