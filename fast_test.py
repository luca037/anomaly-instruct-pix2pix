import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler
import numpy as np

import textwrap
import matplotlib.pyplot as plt


def plot_single_row_grid_heat(image_paths, prompts, output_filename="single_row_grid.png"):
    """
    Plots a 2-row grid. 
    Row 1: The images (Input + Generated outputs).
    Row 2: The difference heatmaps (highlighting what the model changed).
    
    Args:
        image_paths (list): List of paths. The first element is the original input image.
                            The subsequent elements are the generated images.
        prompts (list): List of text prompts used for the generated images. 
        output_filename (str): Where to save the final image.
    """
    num_images = len(image_paths)
    if num_images == 0:
        print("Error: No images provided.")
        return

    # Create the figure. 2 rows, N columns. 
    # squeeze=False ensures 'axes' is always a 2D array, even if num_images=1
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 5, 10), squeeze=False)
    
    input_img_np = None

    for i, img_path in enumerate(image_paths):
        ax_img = axes[0, i]
        ax_heat = axes[1, i]
        
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
            
            # Heatmap for the input is just empty (zero difference)
            ax_heat.imshow(np.zeros((512, 512)), cmap='jet', vmin=0, vmax=1)
            #ax_heat.set_title("Baseline (No Change)", fontsize=14, pad=15)
        else:
            # Subsequent columns are the Generated Images
            ax_img.set_title(f"Generated Image {i}", fontsize=14, pad=15)
            
            # Calculate the Heatmap (Mean absolute difference across RGB channels)
            diff = np.abs(img_np - input_img_np).mean(axis=2)
            
            # Plot the heatmap
            # cmap='jet' maps 0 (no difference) to blue, and higher differences to yellow/red
            ax_heat.set_title("Heatmap", fontsize=14, pad=15)
            im = ax_heat.imshow(diff, cmap='jet', vmin=0, vmax=np.max(diff) if np.max(diff) > 0 else 1)
            #fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            
            # Add the prompt text below the heatmap
            prompt_idx = i - 1
            if prompt_idx < len(prompts):
                raw_prompt = prompts[prompt_idx]
                wrapped_prompt = textwrap.fill(f'"{raw_prompt}"', width=45)
                ax_heat.set_xlabel(wrapped_prompt, fontsize=12, style='italic', labelpad=15)

        # Clean up heatmap axes
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])

    # Adjust layout to ensure the text at the bottom isn't cut off
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.2) 
    
    # Save the plot
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Success! Grid saved to {output_filename}")

def plot_single_row_grid(image_paths, prompts, output_filename="single_row_grid.png"):
    """
    Plots a 1-row grid where the first image is the input, and subsequent images
    are generated outputs with their associated prompts.
    
    Args:
        image_paths (list): List of paths. The first element is the original input image.
                            The subsequent elements are the generated images.
        prompts (list): List of text prompts used for the generated images. 
                        (Expected length: len(image_paths) - 1)
        output_filename (str): Where to save the final image.
    """
    num_images = len(image_paths)
    if num_images == 0:
        print("Error: No images provided.")
        return

    # Create the figure. Width scales based on the number of images.
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 6))
    
    # Ensure axes is always iterable (in case only 1 image is passed by mistake)
    if num_images == 1:
        axes = [axes]

    for i, img_path in enumerate(image_paths):
        ax = axes[i]
        
        # Safely load the image (or a placeholder if it's missing)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load {img_path}. Using a placeholder. ({e})")
            img = Image.new('RGB', (512, 512), color='lightgray')

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # First column is always the Input Image
        if i == 0:
            ax.set_title("Input Image", fontsize=16, fontweight='bold', pad=15)
        else:
            # Subsequent columns are the Generated Images
            ax.set_title(f"Generated Image {i}", fontsize=14, pad=15)
            
            # Fetch the corresponding prompt (safely handling index math)
            prompt_idx = i - 1
            if prompt_idx < len(prompts):
                raw_prompt = prompts[prompt_idx]
                wrapped_prompt = textwrap.fill(f'"{raw_prompt}"', width=45)
                ax.set_xlabel(wrapped_prompt, fontsize=12, style='italic', labelpad=12)

    # Adjust layout to ensure the text at the bottom isn't cut off
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    
    # Save the plot
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Success! Grid saved to {output_filename}")


# 1. Define your paths
# Point to the official base model for the VAE/Text Encoder
base_model_id = "timbrooks/instruct-pix2pix" 

# Point explicitly to the "unet" folder INSIDE your checkpoint
checkpoint_unet_path = "/home/luca_piai/big_disk/mirage-anomaly-model-fine_tune_pix2pix/checkpoint-2000/unet"

#input_image_path = "/home/luca_piai/big_disk/datasets/mvtec/hazelnut/test/good/011.png"
#input_image_path = "/home/luca_piai/big_disk/datasets/my_dataset/pill_good_000.png"
input_image_path = "/home/luca_piai/big_disk/datasets/mvtec/pill/test/good/013.png"

print(f"Loading custom UNet from {checkpoint_unet_path}...")

# 2. Load JUST your trained UNet
trained_unet = UNet2DConditionModel.from_pretrained(
    checkpoint_unet_path, 
    torch_dtype=torch.bfloat16
)

print("Loading base pipeline and injecting UNet...")

# 3. Load the full pipeline, but inject your custom UNet
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    base_model_id,
    unet=trained_unet,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    requires_safety_checker=False
)
pipe.to("cuda:1")

# 4. Swap the scheduler for better InstructPix2Pix results
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 5. Load and resize the base image
init_image = Image.open(input_image_path).convert("RGB").resize((512, 512))

prompt = "scratched imprint"
print(f"Applying prompt: '{prompt}'...")

# 6. Generate images
img_paths = []
for i in range(3):
    output_image = pipe(
        prompt=prompt,
        image=init_image,
        num_inference_steps=20,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
    ).images[0]

    out_img = f"{prompt.replace(' ', '_')}_{i}.png"
    output_image.save(out_img)
    print(f"Success! Saved as {out_img}")
    img_paths.append("" + out_img)

#plot_single_row_grid([input_image_path] + img_paths, [prompt]*len(img_paths), output_filename=f"{prompt.replace(' ', '_')}_grid.jpg")
plot_single_row_grid_heat([input_image_path] + img_paths, [prompt]*len(img_paths), output_filename=f"{prompt.replace(' ', '_')}_grid.jpg")