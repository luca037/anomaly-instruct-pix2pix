import os
import json

TRAIN_PATH = "/home/luca_piai/big_disk/datasets/dpo_train_images/pill/"
MVTEC_PATH = "/home/luca_piai/big_disk/datasets/mvtec/pill/"

PREFIX = "<style1>"
PROMPTS = {
    "crack": PREFIX + "add a crack to the pill",
    "contaminant": PREFIX + "add an embedded contaminant to the pill",
    "hairline": PREFIX + "add a hairline crack to the pill",
    "hole": PREFIX + "add an insect bore hole to the pill",
    "edge": PREFIX + "a pill with a chipped edge"
}

def main():
    defects_dirs = os.listdir(TRAIN_PATH)

    entries = []
    for dir in defects_dirs:
        winners = os.path.join(TRAIN_PATH, dir, "winner")
        losers  = os.path.join(TRAIN_PATH, dir, "loser")
        for img in os.listdir(winners):
            win_path = os.path.join(winners, img)
            lose_path = os.path.join(losers, img)
            original = os.path.join(MVTEC_PATH, "train", "good", img)
            if not os.path.isfile(win_path) or not os.path.isfile(lose_path):
                print("Warning: Missing file for image {} in defect {}".format(img, dir))
            entries.append({
                "winner_image": win_path,
                "loser_image": lose_path,
                "edit_prompt": PROMPTS[dir],
                "original_image": original
            })

    with open("metadata.jsonl", 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    print(f"Created metadata.jsonl with {len(entries)} entries.")


if __name__ == "__main__":
    main()