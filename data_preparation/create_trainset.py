import json
import os
import csv
import urllib.request
import argparse

###
### Settings
###

# Input paths
MVTEC_PATH = "/home/luca_piai/big_disk/datasets/mvtec/"
VISA_PATH = "/home/luca_piai/big_disk/datasets/visa/"
MIRAGE_PATH = "/home/luca_piai/big_disk/datasets/mirage"

# Output path
TRAINSET_PATH = "/home/luca_piai/big_disk/datasets/train_set/"

# Specify the categories to include in the training set
# Set to [] to include all or [None] to skip
MVTEC_CATEGORIES = []
VISA_CATEGORIES = [None]

# Set csv urls
VISA_URL = "https://huggingface.co/datasets/visualanom/mirage_mvtec_visa/resolve/main/visa/generation_tracking.csv"
MVTEC_URL = "https://huggingface.co/datasets/visualanom/mirage_mvtec_visa/resolve/main/mvtec/generation_tracking.csv"
    
# Set csv file names
VISA_CSV = "visa_generation_tracking.csv"
MVTEC_CSV = "mvtec_generation_tracking_augmented.csv"


###
### Utils
###

def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}...")
        urllib.request.urlretrieve(url, local_path)
    print(f"File {local_path} already exists, skipping download.")


def process_csv(csv_path, dataset_path, categories, mirage_path, ds_name, perc):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = [row for row in reader]

    # Each row is associated to a pair (object, defect_type).
    # defect_type is given by the defect_synonym.
    rows_dict = {}
    for idx, row in enumerate(all_rows):
        obj = row.get('object')
        defect = row.get('defect_synonym').split(':')[0]
        if (obj, defect) not in rows_dict:
            rows_dict[(obj, defect)] = [row]
        else: 
            rows_dict[(obj, defect)].append(row)
    
    filtered_rows = []
    for _, v in rows_dict.items():
        end = int(len(v) * perc)
        filtered_rows.extend(v[:end])

    entries = []
    for row in filtered_rows:
        obj_category = row.get('object', '')
        if not categories or obj_category in categories:
            base_img = row.get('base_img_path', '')
            img = row.get('img_path', '')
            prompt = row.get('defect_synonym', '')

            # Adjust edited_image path: (needed because `img_path` is not the acutal path... idk why)
            # e.g. from `/transistor/test/general/general_0000.png`  to `transistor/images/general_0000.png`
            img = os.path.join(img.split('/')[0], 'images', os.path.basename(img))
            
            # Pre-append paths
            base_img_full = os.path.join(dataset_path, base_img.lstrip('/'))
            img_full = os.path.join(mirage_path, ds_name, img.lstrip('/'))
            
            entries.append({
                "original_image": base_img_full,
                "edited_image": img_full,
                "edit_prompt": prompt
            })
    return entries


# Needed to remove `Please generate an image...` part
def clean_cable_defect_prompt(mvtec_csv):
    with open(mvtec_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames  # Preserve column headers
        rows = list(reader)

    for row in rows:
        if row.get('object') == 'cable':
            prompt = row.get('defect_synonym', '')
            split = prompt.split('. ')
            
            if len(split) >= 2:
                row['defect_synonym'] = "".join(split[1:])

    with open(mvtec_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


###
### Main
###

def main():
    parser = argparse.ArgumentParser(description="Create train set.")
    parser.add_argument('--download', action='store_true', help="Download mvtec csv and visa csv from MIRAGE dataset.")
    parser.add_argument('--mvtec_perc', type=float, default=1.0, help="How many samples are retained (1.0 means all, 0.5 means half discarted).")
    parser.add_argument('--visa_perc', type=float, default=1.0, help="How many samples are retained (1.0 means all, 0.5 means half discarted).")
    args = parser.parse_args()

    # Download the two csv files
    if args.download:
        download_file(VISA_URL, VISA_CSV)
        download_file(MVTEC_URL, MVTEC_CSV)
        print("Cleaning MVTEC cable prompt...")
        clean_cable_defect_prompt(MVTEC_CSV)
    
    # Clip percentages
    if args.mvtec_perc > 1.0 or args.mvtec_perc < 0.0:
        args.mvtec_perc = 1.0
    if args.visa_perc > 1.0 or args.visa_perc < 0.0:
        args.visa_perc = 1.0
    
    # Process CSVs
    all_entries = []
    if not len(MVTEC_CATEGORIES) or  MVTEC_CATEGORIES[0] is not None:
        print("Processing MVTEC data...")
        all_entries.extend(process_csv(MVTEC_CSV, MVTEC_PATH, MVTEC_CATEGORIES, MIRAGE_PATH, ds_name='mvtec', perc=args.mvtec_pec))
    if not len(VISA_CATEGORIES) or VISA_CATEGORIES[0] is not None:
        print("Processing VISA data...")
        all_entries.extend(process_csv(VISA_CSV, VISA_PATH, VISA_CATEGORIES, MIRAGE_PATH, ds_name='visa', perc=args.visa_perc))
    
    # Store inside TRAINSET_PATH directory
    os.makedirs(TRAINSET_PATH, exist_ok=True)
    out_path = os.path.join(TRAINSET_PATH, "metadata.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"Created {out_path} with {len(all_entries)} entries.")


if __name__ == '__main__':
    main()