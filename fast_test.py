import os
import argparse

import torch
import numpy as np

from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler

import textwrap
import matplotlib.pyplot as plt

# Default paths mapping
category_paths = {
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
    "zipper": "/home/luca_piai/big_disk/datasets/mvtec/zipper/test/good/001.png"
}


def plot_grid(image_paths, prompts, output_filename="grid.png", use_heatmap=False):
    """
    Plots a grid. 
    Row 1: The images (Input + Generated outputs).
    Row 2 (optional): The difference heatmaps (highlighting what the model changed).
    
    Args:
        image_paths (list): List of paths. The first element is the original input image.
                            The subsequent elements are the generated images.
        prompts (list): List of text prompts used for the generated images. 
        output_filename (str): Where to save the final image.
        use_heatmap (bool): Whether to plot a second row with heatmaps.
    """
    num_images = len(image_paths)
    if num_images == 0:
        print("Error: No images provided.")
        return

    num_rows = 2 if use_heatmap else 1

    # Create the figure. 
    # squeeze=False ensures 'axes' is always a 2D array
    fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images * 5, 5 * num_rows + (2 if use_heatmap else 1)), squeeze=False)
    
    input_img_np = None

    for i, img_path in enumerate(image_paths):
        ax_img = axes[0, i]
        
        # Safely load the image
        try:
            img = Image.open(img_path).convert("RGB")
            # Force resize to 512x512 just in case, so numpy subtraction doesn't crash
            img = img.resize((512, 512))
            img_np = np.array(img).astype(np.float32) / 255.0
        except Exception as e:
            print(f"Warning: Could not load {img_path}. Using a placeholder. ({e})")
            img = Image.new('RGB', (512, 512), color='lightgray')
            img_np = np.zeros((512, 512, 3), dtype=np.float32)

        # Plot the top row (Actual Images)
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # First column is the Input Image
        if i == 0:
            input_img_np = img_np
            
            ax_img.set_title("Input Image", fontsize=16, fontweight='bold', pad=15)
            
            if use_heatmap:
                ax_heat = axes[1, i]
                # Heatmap for the input is just empty (zero difference)
                ax_heat.imshow(np.zeros((512, 512)), cmap='jet', vmin=0, vmax=1)
                ax_heat.set_xticks([])
                ax_heat.set_yticks([])
        else:
            # Subsequent columns are the Generated Images
            ax_img.set_title(f"Generated Image {i}", fontsize=14, pad=15)
            
            prompt_idx = i - 1
            wrapped_prompt = ""
            if prompt_idx < len(prompts):
                raw_prompt = prompts[prompt_idx]
                wrapped_prompt = textwrap.fill(f'"{raw_prompt}"', width=45)
            
            if use_heatmap:
                ax_heat = axes[1, i]
                # Calculate the Heatmap (Mean absolute difference across RGB channels)
                diff = np.abs(img_np - input_img_np).mean(axis=2)
                
                # Plot the heatmap
                ax_heat.set_title("Heatmap", fontsize=14, pad=15)
                im = ax_heat.imshow(diff, cmap='jet', vmin=0, vmax=np.max(diff) if np.max(diff) > 0 else 1)
                
                # Add the prompt text below the heatmap
                if wrapped_prompt:
                    ax_heat.set_xlabel(wrapped_prompt, fontsize=12, style='italic', labelpad=15)

                # Clean up heatmap axes
                ax_heat.set_xticks([])
                ax_heat.set_yticks([])
            else:
                if wrapped_prompt:
                    ax_img.set_xlabel(wrapped_prompt, fontsize=12, style='italic', labelpad=12)

    # Adjust layout to ensure the text at the bottom isn't cut off
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15 if use_heatmap else 0.2, hspace=0.2 if use_heatmap else 0) 
    
    # Save the plot
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Success! Grid saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run InstructPix2Pix and generate comparison grids.")
    parser.add_argument("--category", type=str, choices=list(category_paths.keys()), default="hazelnut",
                        help="Category to use for the input image (hazelnut or pill)")
    parser.add_argument("--prompt", type=str, default="scratched imprint",
                        help="Prompt for the image generation")
    parser.add_argument("--device", type=str, required=True, 
                        help="Device to run the model on (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the output grid image")
    parser.add_argument("--heatmap", action="store_true", default=True,
                        help="Enable the heatmap row in the output grid (default: True)")
    parser.add_argument("--no-heatmap", action="store_false", dest="heatmap",
                        help="Disable the heatmap row in the output grid")
    
    args = parser.parse_args()

    input_image_path = category_paths[args.category]

    output_filename = args.output_path
    if output_filename is None:
        output_filename = f"{args.prompt.replace(' ', '_')}_grid.jpg"

    base_model_id = "timbrooks/instruct-pix2pix" 
    checkpoint_unet_path = "/home/luca_piai/big_disk/mirage-anomaly-model-fine_tune_pix2pix/checkpoint-2000/unet"

    print(f"Loading custom UNet from {checkpoint_unet_path}...")
    
    # Load trained UNet
    trained_unet = UNet2DConditionModel.from_pretrained(
        checkpoint_unet_path, 
        torch_dtype=torch.bfloat16
    )

    # Load the full pipeline, but inject your custom UNet
    print("Loading base pipeline and injecting UNet...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        base_model_id,
        unet=trained_unet,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.to(args.device)

    # Swap the scheduler for better InstructPix2Pix results
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Load and resize the base image
    init_image = Image.open(input_image_path).convert("RGB").resize((512, 512))
    print(f"Applying prompt: '{args.prompt}'...")

    # Generate images
    img_paths = []
    # Output directory for individual images based on output_filename
    out_dir = os.path.dirname(output_filename) or "."
    for i in range(3):
        output_image = pipe(
            prompt=args.prompt,
            image=init_image,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
        ).images[0]

        out_img = os.path.join(out_dir, f"{args.prompt.replace(' ', '_')}_{i}.png")
        output_image.save(out_img)
        img_paths.append(out_img)

    plot_grid(
        [input_image_path] + img_paths, 
        [args.prompt] * len(img_paths), 
        output_filename=output_filename, 
        use_heatmap=args.heatmap
    )

if __name__ == "__main__":
    main()