import argparse
import json
import textwrap
import matplotlib.pyplot as plt
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a comparison grid from the evaluation JSON.")
    parser.add_argument(
        "--json_path", 
        type=str, 
        required=True, 
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_image", 
        type=str, 
        default="model_comparison_grid.jpg", 
        help="Name of the final output grid image."
    )
    parser.add_argument(
        "--start", 
        type=int,
        default=0,
        help="Starting index for the input images to include in the grid."
    )
    parser.add_argument(
        "--end", 
        type=int,
        default=None,
        help="Ending index for the input images to include in the grid."
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category of the images to include in the grid."
    )
    return parser.parse_args()

def load_image(path):
    """Safely load an image, returning a blank placeholder if it fails."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not load image {path}. Using blank placeholder. Error: {e}")
        # Return a white square as a fallback
        return Image.new('RGB', (512, 512), color='white')

def main():
    args = parse_args()

    # Load the JSON data
    print(f"Loading data from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        input_data = json.load(f)

    category = input_data.get(args.category, "unknown_category")
    inputs  = category.get("inputs",  [])[args.start:args.end]
    prompts = category.get("prompts", [])[args.start:args.end]
    outputs = category.get("outputs", {})

    if not inputs:
        raise ValueError("No 'input' data found in the JSON.")

    model_names = list(outputs.keys())
    num_cols = len(inputs)
    num_rows = 1 + len(model_names) # 1 row for inputs + 1 row per model

    print(f"Generating grid: {num_rows} rows x {num_cols} columns...")

    # 2. Setup the matplotlib figure
    # Dynamically scale the figure size (width=5 per image, height=5.5 per row to leave room for text)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5.5))
    
    # Ensure axes is always a 2D array even if there's only 1 column
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    # 3. Populate the grid
    for col_idx in range(num_cols):
        # --- ROW 0: Input Image & Prompt ---
        ax_input = axes[0, col_idx]

        img_input = load_image(inputs[col_idx])
        ax_input.imshow(img_input)
        
        # Format the prompt to wrap text so it doesn't overlap with neighbors
        raw_prompt = prompts[col_idx]
        wrapped_prompt = textwrap.fill(f'"{raw_prompt}"', width=40)
        
        # Add column title (Original Image) and xlabel (The Prompt)
        if col_idx == 0:
            ax_input.set_ylabel("Original Input", fontsize=16, fontweight='bold', labelpad=20)
            
        ax_input.set_title(f"Image {col_idx+1}", fontsize=14, pad=10)
        ax_input.set_xlabel(wrapped_prompt, fontsize=16, style='italic', labelpad=10)
        
        # Remove tick marks
        ax_input.set_xticks([])
        ax_input.set_yticks([])

        # --- ROWS 1 to N: Model Outputs ---
        for row_offset, model_name in enumerate(model_names):
            row_idx = row_offset + 1
            ax_model = axes[row_idx, col_idx]
            
            # Fetch the generated image path from the JSON
            try:
                img_path = outputs[model_name][col_idx + args.start]
                img_model = load_image(img_path)
            except IndexError:
                print(f"Warning: Missing output for model '{model_name}' at index {col_idx}.")
                img_model = Image.new('RGB', (512, 512), color='white')

            ax_model.imshow(img_model)
            
            # Label the first column with the model's name
            if col_idx == 0:
                ax_model.set_ylabel(model_name, fontsize=18, fontweight='bold', labelpad=20)
                
            # Clean up axes
            ax_model.set_xticks([])
            ax_model.set_yticks([])

    # 4. Final Formatting and Save
    plt.tight_layout()
    
    # Add a slight padding to the bottom so the prompt text isn't cut off
    plt.subplots_adjust(bottom=0.05, hspace=0.1, wspace=0.05)
    
    print(f"Saving comparison grid to {args.output_image}...")
    #plt.savefig(args.output_image, dpi=150, bbox_inches='tight')
    plt.savefig(args.output_image, bbox_inches='tight')
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    main()