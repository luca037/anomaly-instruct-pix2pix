import json
import os
import csv
import urllib.request

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
        return 1
    print(f"File {local_path} already exists, skipping download.")
    return 0


def process_csv(csv_path, dataset_path, categories, mirage_path, ds_name):
    entries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    # Download the two csv files
    download_file(VISA_URL, VISA_CSV)
    if download_file(MVTEC_URL, MVTEC_CSV):
        print("Cleaning MVTEC cable prompt...")
        clean_cable_defect_prompt(MVTEC_CSV) # Apply only when dowloaded first time
    
    # Process CSVs
    all_entries = []
    if not len(MVTEC_CATEGORIES) or  MVTEC_CATEGORIES[0] is not None:
        print("Processing MVTEC data...")
        all_entries.extend(process_csv(MVTEC_CSV, MVTEC_PATH, MVTEC_CATEGORIES, MIRAGE_PATH, ds_name='mvtec'))
    if not len(VISA_CATEGORIES) or VISA_CATEGORIES[0] is not None:
        print("Processing VISA data...")
        all_entries.extend(process_csv(VISA_CSV, VISA_PATH, VISA_CATEGORIES, MIRAGE_PATH, ds_name='visa'))
    
    # Store inside TRAINSET_PATH directory
    os.makedirs(TRAINSET_PATH, exist_ok=True)
    out_path = os.path.join(TRAINSET_PATH, "metadata.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"Created {out_path} with {len(all_entries)} entries.")


if __name__ == '__main__':
    main()