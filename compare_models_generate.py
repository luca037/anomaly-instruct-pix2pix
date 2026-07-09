import argparse
import json
import os
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from tqdm import tqdm


DEVICE = "cuda:1"
BASE_MVTEC_PATH = "/home/luca_piai/big_disk/datasets/mvtec/"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned InstructPix2Pix models.")
    parser.add_argument(
        "--json_path", 
        type=str, 
        required=True, 
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--category", 
        type=str, 
        required=True, 
        help="Name of the model/category (used for the output folder and JSON key)."
    )
    parser.add_argument(
        "--unet_path", 
        type=str, 
        default=None, 
        help="Path to the trained UNet directory (e.g., checkpoint-1000/unet)."
    )
    parser.add_argument(
        "--baseline", 
        action="store_true", 
        help="Skip UNet replacement and test the vanilla InstructPix2Pix model."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Safety check: ensure we have a UNet path if we aren't running the baseline
    if not args.baseline and not args.unet_path:
        raise ValueError("You must provide --unet_path unless the --baseline flag is used.")

    # 1. Load the JSON file
    print(f"Loading JSON data from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    input_paths = data.get("input", [])
    prompts = data.get("prompt", [])

    if not input_paths or len(input_paths) != len(prompts):
        raise ValueError("The 'input' and 'prompt' lists must exist and be the same length.")

    # Initialize the list for this specific category
    data["output"][args.category] = []

    # 2. Setup the output directory
    output_dir = os.path.join("output", args.category)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")

    # 3. Load the Model Pipeline
    base_model_id = "timbrooks/instruct-pix2pix"
    weight_dtype = torch.bfloat16

    if args.baseline:
        print("Loading Vanilla Baseline InstructPix2Pix...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            base_model_id,
            torch_dtype=weight_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
    else:
        print(f"Loading custom UNet from {args.unet_path}...")
        trained_unet = UNet2DConditionModel.from_pretrained(args.unet_path, torch_dtype=weight_dtype)
        
        print("Injecting UNet into base pipeline...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            base_model_id,
            unet=trained_unet,
            torch_dtype=weight_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )

    # Swap scheduler and move to GPU
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)

    # 4. The Generation Loop
    print(f"Starting generation for {len(input_paths)} images...")
    
    for i, (img_path, prompt) in enumerate(tqdm(zip(input_paths, prompts), total=len(input_paths))):
        # Format the output filename (test_good_000.png)
        file_name = img_path.replace('/', '_')
        save_path = os.path.join(output_dir, file_name)

        # Load and prep the image
        img_path = os.path.join(BASE_MVTEC_PATH, img_path)
        init_image = Image.open(img_path).convert("RGB").resize((512, 512))

        # Generate
        out_image = pipe(
            prompt=prompt,
            image=init_image,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7.5
        ).images[0]

        # Save image and record path
        out_image.save(save_path)
        data["output"][args.category].append(save_path)

    # 5. Save the updated JSON file
    print("Updating JSON file...")
    with open(args.json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Successfully finished testing category: '{args.category}'!")

if __name__ == "__main__":
    main()