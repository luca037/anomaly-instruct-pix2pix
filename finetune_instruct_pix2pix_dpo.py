#!/usr/bin/env python
# coding=utf-8
"""Fine-tune InstructPix2Pix with Diffusion-DPO.

Adapted from ``finetune_instruct_pix2pix.py`` (SFT) and
SalesforceAIResearch/DiffusionDPO ``train.py`` (DPO loss).

Preference dataset: ``metadata.jsonl`` with columns
``original_image, winner_image, loser_image, edit_prompt``.
Both winner and loser are sampled from the SFT ip2p checkpoint and
manually ranked (e.g. pill dots preserved vs erased). DPO is run offline:
``p_ref`` and ``p_theta`` are both initialised from the same fine-tuned
checkpoint (``--pretrained_model_name_or_path``), ``p_ref`` stays frozen.

Synchronized augmentation (center/random crop + horizontal flip) is applied
to the triplet ``(original, winner, loser)`` by concatenating along the
channel-batch axis before ``train_transforms`` – same trick as the SFT script.
"""

import argparse
import inspect
import logging
import math
import os
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import diffusers
import numpy as np
import PIL
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

check_min_version("0.15.0.dev0")
logger = get_logger(__name__, log_level="INFO")

WANDB_TABLE_COL_NAMES = ["original_image", "winner_image", "edit_prompt"]


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="DPO alignment for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to the fine-tuned ip2p pipeline (output of SFT) or HF id. Both p_ref and p_theta are initialised from here.",
    )
    parser.add_argument("--revision", type=str, default=None, help="Revision.")
    parser.add_argument("--dataset_name", type=str, default=None, help="HF dataset name (alternative to --train_data_dir).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Dataset config.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Folder with metadata.jsonl for preference pairs. Ignored if --dataset_name is set.",
    )
    parser.add_argument("--original_image_column", type=str, default="original_image", help="Column for conditioning image.")
    parser.add_argument("--winner_image_column", type=str, default="winner_image", help="Column for preferred image.")
    parser.add_argument("--loser_image_column", type=str, default="loser_image", help="Column for unpreferred image.")
    parser.add_argument("--edit_prompt_column", type=str, default="edit_prompt", help="Column for edit instruction.")
    parser.add_argument("--val_image_url", type=str, default=None, help="URL/path for validation image.")
    parser.add_argument("--validation_prompt", type=str, default=None, help="Prompt for validation inference.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Num images during validation.")
    parser.add_argument("--validation_epochs", type=int, default=1, help="Run validation every X epochs.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Truncate dataset for debugging.")
    parser.add_argument("--output_dir", type=str, default="instruct-pix2pix-dpo", help="Output dir.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir.")
    parser.add_argument("--seed", type=int, default=None, help="Seed.")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution for preprocessing.")
    parser.add_argument("--center_crop", action="store_true", help="Center crop instead of random crop.")
    parser.add_argument("--random_flip", action="store_true", help="Random horizontal flip.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Total train steps (overrides epochs).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Grad accum steps.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Gradient checkpointing.")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Initial learning rate.")
    parser.add_argument("--scale_lr", action="store_true", help="Scale LR by GPUs*bs*grad_accum.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='Scheduler: ["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max grad norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="Hub token.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub model id.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log dir.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Mixed precision.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report to.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help=" Save every X updates.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume path or 'latest'.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Use xformers.")
    parser.add_argument("--torch_compile", action="store_true", help="Compile unet with torch.compile.")
    # DPO specific
    parser.add_argument("--beta_dpo", type=float, default=5000, help="Beta DPO KL penalty.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam.")
    parser.add_argument("--non_ema_revision", type=str, default=None, help="Non-EMA revision.")
    return parser.parse_args()


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    from huggingface_hub import HfFolder

    try:
        from huggingface_hub import whoami
    except ImportError:
        def whoami(token):
            return {"name": "unknown"}

    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        image = PIL.Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        image = PIL.Image.open(url_or_path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either --dataset_name or --train_data_dir.")
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb for logging.")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.push_to_hub:
            from huggingface_hub import create_repo, Repository

            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)
            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "checkpoint-*" not in gitignore:
                    gitignore.write("checkpoint-*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision)
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ref_unet.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 has issues on some GPUs, update to >=0.0.17.")
            unet.enable_xformers_memory_efficient_attention()
            ref_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available.")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                target_model = getattr(model, "_orig_mod", model)
                target_model.register_to_config(**load_model.config)
                target_model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam.")

        optimizer_cls = bnb.optim.AdamW8bit
        fused_available = "fused" in inspect.signature(bnb.optim.AdamW8bit).parameters
    else:
        optimizer_cls = torch.optim.AdamW
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters

    if fused_available:
        print("Using fused=True.")

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        fused=fused_available,
    )

    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
    else:
        metadata_path = os.path.join(args.train_data_dir, "metadata.jsonl")
        dataset = load_dataset("json", data_files={"train": metadata_path}, cache_dir=args.cache_dir)

        def make_absolute_paths(example):
            for col in [args.original_image_column, args.winner_image_column, args.loser_image_column]:
                example[col] = os.path.join(args.train_data_dir, example[col])
            return example

        dataset["train"] = dataset["train"].map(make_absolute_paths)
        from datasets import Image as HFImage

        dataset = dataset.cast_column(args.original_image_column, HFImage())
        dataset = dataset.cast_column(args.winner_image_column, HFImage())
        dataset = dataset.cast_column(args.loser_image_column, HFImage())

    dataset["train"].column_names

    def tokenize_captions(captions):
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        # Synchronized augmentation for (original, winner, loser) via channel concat trick
        original = np.concatenate([convert_to_np(im, args.resolution) for im in examples[args.original_image_column]])
        winner = np.concatenate([convert_to_np(im, args.resolution) for im in examples[args.winner_image_column]])
        loser = np.concatenate([convert_to_np(im, args.resolution) for im in examples[args.loser_image_column]])
        images = np.concatenate([original, winner, loser])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    def preprocess_train(examples):
        preprocessed = preprocess_images(examples)
        original_images, winner_images, loser_images = preprocessed.chunk(3)
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        winner_images = winner_images.reshape(-1, 3, args.resolution, args.resolution)
        loser_images = loser_images.reshape(-1, 3, args.resolution, args.resolution)
        examples["original_pixel_values"] = original_images
        examples["winner_pixel_values"] = winner_images
        examples["loser_pixel_values"] = loser_images
        captions = [c for c in examples[args.edit_prompt_column]]
        examples["input_ids"] = tokenize_captions(captions)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([e["original_pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
        winner_pixel_values = torch.stack([e["winner_pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
        loser_pixel_values = torch.stack([e["loser_pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([e["input_ids"] for e in examples])
        return {
            "original_pixel_values": original_pixel_values,
            "winner_pixel_values": winner_pixel_values,
            "loser_pixel_values": loser_pixel_values,
            "input_ids": input_ids,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps
    )

    if args.torch_compile:
        unet = torch.compile(unet, mode="default")
        ref_unet = torch.compile(ref_unet, mode="default")

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix-dpo", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running DPO training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Beta DPO = {args.beta_dpo}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # ==========================================================
                # 1. Encode winner / loser to latents + original to cond
                # ==========================================================
                winner_latents = vae.encode(batch["winner_pixel_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                loser_latents = vae.encode(batch["loser_pixel_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                original_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

                # ==========================================================
                # 2. Sample shared timesteps + noise per preference pair
                # ==========================================================
                bsz = winner_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=winner_latents.device).long()
                noise = torch.randn_like(winner_latents)
                noise_l = noise.clone()
                timesteps_w = timesteps
                timesteps_l = timesteps

                # ==========================================================
                # 3. Add noise to latents (forward diffusion)
                # ==========================================================
                noisy_winner = noise_scheduler.add_noise(winner_latents, noise, timesteps_w)
                noisy_loser = noise_scheduler.add_noise(loser_latents, noise_l, timesteps_l)

                # ==========================================================
                # 4. Encode text prompt
                # ==========================================================
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # ==========================================================
                # 5. Concatenate conditioning image (ip2p: 8 channels)
                # ==========================================================
                concat_winner = torch.cat([noisy_winner, original_embeds], dim=1)
                concat_loser = torch.cat([noisy_loser, original_embeds], dim=1)
                concat_all = torch.cat([concat_winner, concat_loser], dim=0)
                timesteps_all = torch.cat([timesteps_w, timesteps_l], dim=0)
                encoder_hidden_states_all = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
                target_all = torch.cat([noise, noise_l], dim=0)

                # ==========================================================
                # 6. UNet forward (policy)
                # ==========================================================
                model_pred = unet(concat_all, timesteps_all, encoder_hidden_states_all).sample

                # ==========================================================
                # 7. DPO loss (Diffusion-DPO)
                # ==========================================================
                model_losses = (model_pred - target_all).pow(2).mean(dim=[1, 2, 3])
                model_losses_w, model_losses_l = model_losses.chunk(2)
                model_diff = model_losses_w - model_losses_l
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())

                with torch.no_grad():
                    ref_pred = ref_unet(concat_all, timesteps_all, encoder_hidden_states_all).sample.detach()
                    ref_losses = (ref_pred - target_all).pow(2).mean(dim=[1, 2, 3])
                    ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_losses.mean()

                scale_term = -0.5 * args.beta_dpo
                inside_term = scale_term * (model_diff - ref_diff)
                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                loss = -F.logsigmoid(inside_term).mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                avg_model_mse = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                avg_acc = accelerator.gather(implicit_acc).mean().item()
                implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "model_mse": avg_model_mse, "ref_mse": avg_ref_mse, "implicit_acc": implicit_acc_accumulated}, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "impl_acc": avg_acc}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # ==============================================================
        # End-of-epoch validation
        # ==============================================================
        if accelerator.is_main_process and args.val_image_url is not None and args.validation_prompt is not None and (epoch % args.validation_epochs == 0):
            logger.info(f"Validation: {args.validation_prompt}")
            unet_eval = accelerator.unwrap_model(unet)
            unet_eval = getattr(unet_eval, "_orig_mod", unet_eval)
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                args.pretrained_model_name_or_path, unet=unet_eval, revision=args.revision, torch_dtype=weight_dtype, safety_checker=None, requires_safety_checker=False
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
            original_image = download_image(args.val_image_url)
            edited_images = []
            with torch.autocast(str(accelerator.device), dtype=weight_dtype, enabled=accelerator.mixed_precision in ["fp16", "bf16"]):
                for _ in range(args.num_validation_images):
                    edited_images.append(pipeline(args.validation_prompt, image=original_image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, generator=generator).images[0])
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    import wandb

                    wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                    for edited_image in edited_images:
                        wandb_table.add_data(wandb.Image(original_image), wandb.Image(edited_image), args.validation_prompt)
                    tracker.log({"validation": wandb_table})
            del pipeline
            torch.cuda.empty_cache()

    # ==================================================================
    # Save final pipeline
    # ==================================================================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = getattr(unet, "_orig_mod", unet)
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.pretrained_model_name_or_path, text_encoder=text_encoder, vae=vae, unet=unet, revision=args.revision)
        pipeline.save_pretrained(args.output_dir)
        if args.push_to_hub:
            from huggingface_hub import Repository

            repo.push_to_hub(commit_message="End of DPO training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
