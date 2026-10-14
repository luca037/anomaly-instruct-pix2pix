# Data preparation

Scripts that build the training sets for the generator.

- `create_finetune_trainset.py` — builds the SFT `metadata.jsonl` (MIRAGE pairs).
- `create_dpo_trainset.py` — builds the DPO preference `metadata.jsonl`.
- `augment_mvtec.py` — auxiliary; not covered here.

Run the scripts from inside this `data_preparation/` folder (they resolve their
`csv/` inputs relative to the working directory).

## Fine-tuning train set

Merges the MIRAGE MVTec/VisA generation-tracking CSVs into a single
`metadata.jsonl`.

```bash
cd data_preparation

uv run python create_finetune_trainset.py \
  --download \
  --mvtec_path /home/luca_piai/big_disk/datasets/mvtec/ \
  --visa_path /home/luca_piai/big_disk/datasets/visa/ \
  --mirage_path /home/luca_piai/big_disk/datasets/mirage \
  --trainset_path /home/luca_piai/big_disk/datasets/train_set/
```

`--download` fetches the tracking CSVs into `csv/` (skip it to reuse existing
files). `--mvtec_perc` / `--visa_perc` keep a fraction of each
`(object, defect_type)` group; `--mvtec_categories` / `--visa_categories`
select categories (`None` skips a dataset).

### Output

```
<trainset_path>/
└── metadata.jsonl
```

Each line references the source images in place (no copies are made):

```json
{"original_image": "<mvtec_path>/<obj>/train/good/<img>.png",
 "edited_image":   "<mirage_path>/{mvtec|visa}/<obj>/images/<img>.png",
 "edit_prompt":    "<instruction>"}
```

- `original_image` — clean image from MVTec/VisA.
- `edited_image` — edited image from MIRAGE (`.../images/` subtree).
- `edit_prompt` — the edit instruction.

This is the `--train_data_dir` consumed by `finetune_instruct_pix2pix.py` and
`finetune_flux2_klein_lora.py`.

## DPO train set

Pairs each preferred (`winner`) generation with its dispreferred (`loser`)
counterpart and the matching clean image.

```bash
cd data_preparation

uv run python create_dpo_trainset.py \
  --train_path /home/luca_piai/big_disk/datasets/dpo_train_images/pill/ \
  --mvtec_path /home/luca_piai/big_disk/datasets/mvtec/pill/ \
  --output /home/luca_piai/big_disk/datasets/dpo_train_set/metadata.jsonl
```

Expected input layout — winner/loser folders must share filenames, and each
filename is matched against `<mvtec_path>/train/good/`:

```
<train_path>/
├── crack/
│   ├── winner/   # e.g. 000.png
│   └── loser/    # e.g. 000.png
├── contaminant/
├── hairline/
├── hole/
└── edge/
```

The defect prompt is looked up from the built-in `PROMPTS` map (overridable via
`--prefix`).

### Output

```
<output dir>/
└── metadata.jsonl
```

```json
{"winner_image":   "<train_path>/<defect>/winner/<img>",
 "loser_image":    "<train_path>/<defect>/loser/<img>",
 "edit_prompt":    "<instruction>",
 "original_image": "<mvtec_path>/train/good/<img>"}
```

This is the `--train_data_dir` consumed by `finetune_instruct_pix2pix_dpo.py`.
