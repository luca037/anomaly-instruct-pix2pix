import argparse
import json
import os
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from tqdm import tqdm


###
### Settings
###

BASE_MVTEC_PATH = "/home/luca_piai/big_disk/datasets/mvtec/"
OUTPUT_BASE_PATH = "./output/"
OUTPUT_JSON = "results.json"


###
### Functions
###

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
        help="Category/class to evaluate (e.g., 'hazelnut', 'pill', etc.)."
    )
    parser.add_argument(
        "--id", 
        type=str, 
        required=True, 
        help="Name of the model (used for the output folder and JSON key)."
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
    parser.add_argument(
        "--device", 
        type=str,
        required=True,
        help="Device to run the model on (e.g., 'cuda:0', 'cpu')."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Safety check: ensure we have a UNet path if we aren't running the baseline
    if not args.baseline and not args.unet_path:
        raise ValueError("You must provide --unet_path unless the --baseline flag is used.")

    # Load the JSON file
    print(f"Loading JSON data from {args.json_path}...")
    # Create a copy of input data, if necessary
    if not os.path.exists(OUTPUT_JSON):
        with open(args.json_path, 'r') as f:
            input_data = json.load(f)
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(input_data, f)

    with open(OUTPUT_JSON, 'r') as f:
        output_data = json.load(f)

    input_paths = output_data[args.category].get("inputs", [])
    prompts = output_data[args.category].get("prompts", [])

    if not input_paths or len(input_paths) != len(prompts):
        raise ValueError("The 'inputs' and 'prompts' lists must exist and be the same length.")

    # Initialize the list for this specific category
    output_data[args.category]['outputs'] = {args.id : []}

    # Setup the output directory
    output_dir = os.path.join(OUTPUT_BASE_PATH, args.id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")

    # Load the Model Pipeline
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
    pipe.to(args.device)

    # The Generation Loop
    print(f"Starting generation for {len(input_paths)} images...")
    for i, (img_path, prompt) in enumerate(tqdm(zip(input_paths, prompts), total=len(input_paths))):
        file_name = f"{args.category}_{i:03d}.png"
        save_path = os.path.join(os.path.abspath(output_dir), file_name)

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
        output_data[args.category]['outputs'][args.id].append(save_path)

    # Update results JSON file
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully finished testing category: '{args.category}'!")

if __name__ == "__main__":
    main()