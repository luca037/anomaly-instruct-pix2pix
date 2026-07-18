"""
=============================================================================
MVTec Data Augmentation Script

Description:
This script is designed to augment the 'defect_synonym' column within the 
MVTec generation tracking CSV file. Since many defect prompts are repeated,
this script increases variance by using the DeepSeek API to generate `n` new 
unique descriptions for each original defect prompt (resulting in 10 unique
prompts per defect type in total, or 100 per object category). It then 
creates a new CSV file where these generated prompts are distributed across 
the duplicated rows.

Requirements:
- Python 3.x
- 'openai' library (install via: pip install openai)
- A valid DeepSeek API key set in your environment variables.

How to run:
1. Ensure your API key is set:
   export DEEPSEEK_API_KEY="your-api-key-here"

2. To run the full pipeline (generate prompts via API AND create the CSV):
   python augment_mvtec.py --generate

3. To only create the CSV using previously generated prompts (skipping API):
   python augment_mvtec.py

=============================================================================
"""

import json
import csv
import os
import re
import argparse
from collections import defaultdict
from openai import OpenAI

###
### Settings
###

INPUT_CSV = 'mvtec_generation_tracking.csv'             # MIRAGE mvtec csv
OUTPUT_CSV = 'mvtec_generation_tracking_augmented.csv'  # Augmented output (same structure as INPUT_CSV)
JSON_FILE = 'augmented_prompts.json'                    # Augmented defect prompts

###
### Utils
###

def get_unique_prompts(csv_path):
    # Extract unique prompts.
    unique_prompts = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row['object']
            prompt = row['defect_synonym']
            if obj not in unique_prompts:
                unique_prompts[obj] = set()
            unique_prompts[obj].add(prompt)
    return unique_prompts

def generate_prompts(n):
    """
    Reads the original CSV file to extract unique defect prompts for each object.
    Uses the DeepSeek API (via the OpenAI client) to generate `n` semantically 
    equivalent variations for each prompt. The variations are formatted to keep
    the original defect type prefix (e.g., 'defect type: new description').
    The resulting dictionary of original and generated prompts is saved to a 
    local JSON file to avoid redundant API calls in the future.
    """
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("DEEPSEEK_API_KEY environment variable not set. Please set it to run the API.")
        return
        
    # Initialize the OpenAI client pointing to the DeepSeek API endpoint
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # Extract unique prompts from input csv.
    unique_prompts = get_unique_prompts(INPUT_CSV)

    augmented_prompts = {}
    
    # Load previously generated prompts to resume progress if interrupted
    if os.path.exists(JSON_FILE):
        print(f"Loading existing {JSON_FILE}")
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            augmented_prompts = json.load(f)
            
    for obj, prompts in unique_prompts.items():
        if obj not in augmented_prompts:
            augmented_prompts[obj] = {}
            
        for original_prompt in prompts:
            #if original_prompt in augmented_prompts[obj]:
            #    continue
                
            
            # Separate the defect type prefix from the actual description string
            if ": " in original_prompt:
                defect_type, desc = original_prompt.split(": ", 1)
            else:
                defect_type = ""
                desc = original_prompt

            print(f"Generating variations for: ({obj}, {defect_type})")
                
            system_prompt = "You are a helpful assistant that generates variations of sentences with the same meaning but different words."
            user_prompt = f"""Generate exactly {n} variations of the following description. Explain using other words but keep the exact same meaning. 
Do not include any prefix like numbers or bullet points, just return the {n} variations separated by a newline.

Description: {desc}"""
            
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7
                )
                
                # Split response into individual lines
                variations = response.choices[0].message.content.strip().split('\n')
                
                clean_variations = []
                for v in variations:
                    v = v.strip()
                    if v:
                        # Clean up any residual markdown formatting, dashes, or numbering
                        v = re.sub(r'^(\d+\.|-|\*)\s*', '', v)
                        v = v.strip('"\'')
                        
                        # Re-attach the original defect type prefix
                        if defect_type:
                            clean_variations.append(f"{defect_type}: {v}")
                        else:
                            clean_variations.append(v)
                
                # Ensure we only take exactly n variations as requested
                clean_variations = clean_variations[:n]
                
                if original_prompt not in augmented_prompts[obj]:
                    augmented_prompts[obj][original_prompt] = clean_variations
                else:
                    augmented_prompts[obj][original_prompt].extend(clean_variations)
                
                # Incrementally save results after each successful generation
                with open(JSON_FILE, 'w', encoding='utf-8') as f:
                    json.dump(augmented_prompts, f, indent=4)
                    
            except Exception as e:
                print(f"Error generating for {original_prompt}: {e}")
                
    print(f"Successfully generated prompts and saved to {JSON_FILE}")


def create_augmented_csv():
    """
    Reads the generated JSON file containing the augmented prompts and cycles
    through them to replace the 'defect_synonym' column in the original CSV file.
    It writes the final results to a new output CSV.
    """
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found. Please run with --generate first to create the prompts.")
        return
        
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        augmented_prompts = json.load(f)
        
    rows = []
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found.")
        return
        
    # Read all rows from the original CSV into memory
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
            
    # Counter for (object, original_prompt) pairs to distribute variations evenly
    obj_prompt_counters = defaultdict(int)
    
    # Write the modified rows to the output CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            obj = row['object']
            orig_prompt = row['defect_synonym']
            
            # The pool consists of the original prompt + the n generated variations
            variations = augmented_prompts.get(obj, {}).get(orig_prompt, [])
            pool = [orig_prompt] + variations
            
            # Distribute the prompts sequentially for this specific (obj, orig_prompt)
            idx = obj_prompt_counters[(obj, orig_prompt)] % len(pool)
            row['defect_synonym'] = pool[idx]
            
            obj_prompt_counters[(obj, orig_prompt)] += 1
                
            writer.writerow(row)
            
    print(f"Created {OUTPUT_CSV} with distributed augmented prompts.")

    # Print unique prompts stats.
    unique_prompts = get_unique_prompts(OUTPUT_CSV)
    print("\n### Unique prompts per object category ###")
    for o, p in unique_prompts.items():
        print(f"{o}, {len(p)}")


###
### Main
###

def main():
    parser = argparse.ArgumentParser(description="Augment MVTec generation tracking prompts.")
    parser.add_argument('--generate', action='store_true', help="Generate new prompts using the DeepSeek API.")
    parser.add_argument('--variations', type=int, default=0, help="How many variations per unique prompt.")
    args = parser.parse_args()

    if args.generate and args.variations > 0:
        print("Starting prompt generation...")
        generate_prompts(args.variations)
    
    print("Creating augmented CSV...")
    create_augmented_csv()


if __name__ == '__main__':
    main()
