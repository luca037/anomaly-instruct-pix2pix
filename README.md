
<div align="center">
<img width="688" height="186" alt="icon" src="https://github.com/user-attachments/assets/b2bd0ce6-6ce0-420d-9593-893fd021b0ac" />
</div>

# Anomaly Instruct Pix2Pix

Pipeline for generating synthetic anomalies usable as training data for Visual
Anomaly Detection (VAD) models.

## Pipeline overview

1. **Fine-tune** InstructPix2Pix on the MIRAGE dataset, a collection of
   edit pairs distilled from Gemini Nano Banana. 
2. **Generate** an evaluation set of synthetic defects from clean MVTec images
   with the fine-tuned model.
3. **Evaluate** generation quality (KID score) and downstream utility
   by training VAD models (classification + localization) on the synthetic
   defects and testing on real MVTec.
4. **Align** the generator with Diffusion-DPO preference optimization to improve
   image quality beyond the SFT baseline.

## Training

Fine-tune instruct pix2pix:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch finetune_instruct_pix2pix.py \
  --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
  --train_data_dir="<PATH>" \
  --output_dir="<PATH>" \
  --original_image_column="original_image" \
  --edited_image_column="edited_image" \
  --edit_prompt_column="edit_prompt" \
  --resolution=512 \
  --train_batch_size=8 --gradient_accumulation_steps=2 \
  --use_ema \
  --dataloader_num_workers=8 \
  --learning_rate=5e-5 \
  --random_flip \
  --max_train_steps=12000 \
  --lr_warmup_steps=500 \
  --checkpointing_steps=1000 \
  --val_image_url="<PATH>" \
  --validation_prompt="add a crack" \
  --validation_epochs=5 \
  --num_validation_images=4 \
  --mixed_precision="bf16" \
  --allow_tf32 \
  --torch_compile \
  --report_to="wandb" \
  --seed=42 \
```

Fine-tune Flux2.Klein:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch finetune_flux2_klein_lora.py \
       --pretrained_model_name_or_path="black-forest-labs/FLUX.2-klein-4B" \
       --train_data_dir="<PATH>" \
       --output_dir="<PATH>" \
       --original_image_column="original_image" \
       --edited_image_column="edited_image" \
       --edit_prompt_column="edit_prompt" \
       --dataloader_num_workers=32 \
       --resolution=512 \
       --train_batch_size=8 --gradient_accumulation_steps=2 \
       --gradient_checkpointing \
       --learning_rate=1e-4 \
       --random_flip \
       --rank=32 \
       --lora_alpha=32 \
       --max_train_steps=10000 \
       --checkpointing_steps=500 \
       --mixed_precision="bf16" \
       --allow_tf32 \
       --val_image_url="<PATH>" \
       --validation_prompt="add a crack to the surface of the hazelnut" \
       --num_validation_images=2 \
       --validation_epochs=5 \
       --guidance_scale=1.0 \
       --torch_compile \
       --report_to="wandb" \
       --seed=42
```

Fine-tune instruct pix2pix with DPO:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch finetune_instruct_pix2pix_dpo.py \
  --pretrained_model_name_or_path="<PATH>" \
  --train_data_dir="<PATH>" \
  --output_dir="<PATH>" \
  --resolution=512 \
  --train_batch_size=8 --gradient_accumulation_steps=2 \
  --dataloader_num_workers=16 \
  --learning_rate=1e-7 \
  --center_crop \
  --random_flip \
  --max_train_steps=750 \
  --checkpointing_steps=250 \
  --val_image_url="<PATH>" \
  --validation_prompt="<retain red dots> a pill with a chipped edge" \
  --validation_epochs=8 \
  --num_validation_images=4 \
  --mixed_precision="bf16" \
  --allow_tf32 \
  --torch_compile \
  --report_to="wandb" \
  --seed=42 \
```

## Evaluation

### Generate the evaluation set

```bash
uv run eval/generate_evalset.py \
  --device cuda:1 \
  --weights_path="<PATH TO FINETUNED UNET>" \
  --input_json="eval/defect_prompts.json" \
  --output_dir="<PATH>" \
  --num_images=20
```

### Compute KID

```bash
uv run eval/compute_kid.py \
  --device cuda:1 \
  --real_path="<PATH TO MVTEC>" \
  --generated_path="<PATH>"
```

### Classification (train / test)

```bash
uv run eval/train-classification.py \
  --device cuda:1 \
  --mvtec_path="<PATH>" \
  --generated_path="<PATH>" \
  --checkpoint_path="eval/checkpoints/classification"

uv run eval/test-classification.py \
  --device cuda:1 \
  --mvtec_path="<PATH>" \
  --generated_path="<PATH>" \
  --checkpoint_path="eval/checkpoints/classification"
```

### Localization (train / test)

```bash
uv run eval/train-localization.py \
  --device cuda:1 \
  --mvtec_path="<PATH>" \
  --generated_path="<PATH>" \
  --checkpoint_path="eval/checkpoints/localization"

uv run eval/test-localization.py \
  --device cuda:1 \
  --mvtec_path="<PATH>" \
  --generated_path="<PATH>" \
  --checkpoint_path="eval/checkpoints/localization"
```
