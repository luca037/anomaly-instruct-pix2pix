#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA fine-tuning script for FLUX.2 Klein 4B (InstructPix2Pix-style editing).

This script trains LoRA adapters on the Flux2Transformer2DModel to perform
instruction-based image editing.  It uses the native FLUX.2 sequence-stacking
mechanism: the conditioning (clean) image and the target (edited) image are
both encoded through the VAE, and the clean-image sequence is concatenated
with the noisy target sequence along the sequence dimension (dim=1).

The Transformer learns to denoise the target half of the combined sequence
while attending to the clean conditioning half, guided by a text prompt.

The dataset is expected to contain (original_image, edited_image, edit_prompt)
triplets, the same format used by finetune_instruct_pix2pix.py.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 accelerate launch finetune_flux2_klein_lora.py \\
        --pretrained_model_name_or_path="black-forest-labs/FLUX.2-klein-4B" \\
        --train_data_dir="/path/to/dataset/" \\
        --output_dir="/path/to/output/" \\
        --original_image_column="original_image" \\
        --edited_image_column="edited_image" \\
        --edit_prompt_column="edit_prompt" \\
        --resolution=512 \\
        --train_batch_size=1 \\
        --gradient_accumulation_steps=4 \\
        --gradient_checkpointing \\
        --learning_rate=1e-4 \\
        --rank=16 \\
        --max_train_steps=5000 \\
        --mixed_precision="bf16" \\
        --report_to="wandb" \\
        --seed=42
"""

import inspect
import argparse
import copy
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def convert_to_np(image, resolution):
    """Resize a PIL image and convert to CHW numpy array."""
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def load_image(url_or_path):
    """Load an image from a URL or local path, applying EXIF orientation."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        import requests

        image = Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        image = Image.open(url_or_path)
    from PIL.ImageOps import exif_transpose

    image = exif_transpose(image)
    image = image.convert("RGB")
    return image


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def log_validation(
    pipeline, args, accelerator, prompt_embeds_dict, epoch, torch_dtype,
    is_final_validation=False,
):
    """Run inference with the current model and log results."""
    num_val = args.num_validation_images or 1
    logger.info(
        f"Running validation... Generating {num_val} images "
        f"with prompt: {args.validation_prompt}."
    )
    pipeline = pipeline.to(dtype=torch_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    # Load the validation conditioning image.
    val_image = load_image(args.val_image_url)

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed is not None
        else None
    )
    autocast_ctx = (
        torch.autocast(accelerator.device.type)
        if not is_final_validation
        else nullcontext()
    )

    images = []
    for _ in range(num_val):
        with autocast_ctx:
            image = pipeline(
                image=val_image,
                prompt_embeds=prompt_embeds_dict["prompt_embeds"],
                generator=generator,
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase: [
                        wandb.Image(img, caption=f"{i}: {args.validation_prompt}")
                        for i, img in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()
    return images


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning script for FLUX.2 Klein 4B (InstructPix2Pix-style editing)."
    )

    # -- Model --
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Path to pretrained model or HuggingFace model identifier.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None,
                        help="Model variant for loading (e.g. 'fp16').")

    # -- Dataset --
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help="HuggingFace dataset name or local path.",
    )
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir", type=str, default=None,
        help="Folder containing training data with a metadata.jsonl file.",
    )
    parser.add_argument(
        "--original_image_column", type=str, default="original_image",
        help="Dataset column containing the original (clean) image.",
    )
    parser.add_argument(
        "--edited_image_column", type=str, default="cartoonized_image",
        help="Dataset column containing the edited (target) image.",
    )
    parser.add_argument(
        "--edit_prompt_column", type=str, default="edit_prompt",
        help="Dataset column containing the edit instruction.",
    )

    # -- Validation --
    parser.add_argument(
        "--val_image_url", type=str, default=None,
        help="Path or URL to the original image for validation.",
    )
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=1)

    # -- Output --
    parser.add_argument(
        "--output_dir", type=str, default="flux2-klein-lora-model",
        help="Output directory for checkpoints and final LoRA weights.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=None)

    # -- Image processing --
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)

    # -- Training --
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant",
        help='One of: "linear", "cosine", "cosine_with_restarts", '
             '"polynomial", "constant", "constant_with_warmup".',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument(
        "--conditioning_dropout_prob", type=float, default=None,
        help="Probability of dropping text and/or image conditioning for CFG training.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # -- Optimizer --
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # -- Hardware --
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--mixed_precision", type=str, default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # -- Logging / checkpointing --
    parser.add_argument(
        "--report_to", type=str, default="tensorboard",
        help='"tensorboard", "wandb", "comet_ml", or "all".',
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help='Checkpoint directory or "latest".',
    )

    # -- LoRA --
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=4, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_layers", type=str, default=None,
        help="Comma-separated list of target modules for LoRA.",
    )

    # -- FLUX.2 specific --
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5,
        help="Guidance scale for the distilled guidance embedding.",
    )
    parser.add_argument(
        "--weighting_scheme", type=str, default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Timestep sampling / loss weighting strategy.",
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument(
        "--text_encoder_out_layers", nargs="+", type=int,
        default=[10, 20, 30],
        help="Hidden layer indices of Qwen3 to extract for prompt embeddings.",
    )

    args = parser.parse_args()

    # Handle --local_rank for torch.distributed.launch compatibility
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Validate: need either dataset_name or train_data_dir
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Provide either --dataset_name or --train_data_dir.")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Accelerator
    # ------------------------------------------------------------------
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Allow TF32 on Ampere GPUs.
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------

    # 1. Tokenizer (Qwen2)
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # 2. Weight dtype from mixed-precision setting
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 3. Flow-matching scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # 4. VAE (AutoencoderKLFlux2)
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    # Extract batch-norm running statistics for latent normalization.
    # These are used to normalize patchified latents before feeding them
    # to the Transformer (and to de-normalize when decoding back).
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(accelerator.device)

    # 5. Transformer (Flux2Transformer2DModel)
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    # 6. Text encoder (Qwen3)
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )

    # Freeze all base weights — only LoRA adapters will be trained.
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Move frozen models to device in weight_dtype.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        # Default: double-block attention + single-block attention + projection
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"] + [
            "to_qkv_mlp_proj",
            *[f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)],
        ]

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # For fp16 mixed precision, upcast LoRA params to float32 during training.
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Text encoding pipeline (for prompt embedding)
    # ------------------------------------------------------------------
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

    def compute_text_embeddings(prompt):
        """Encode a prompt (or list of prompts) into Transformer-ready embeddings."""
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=args.max_sequence_length,
                text_encoder_out_layers=args.text_encoder_out_layers,
            )
        return prompt_embeds, text_ids

    # Pre-compute empty-prompt embeddings for conditioning dropout (CFG).
    empty_prompt_embeds, empty_text_ids = None, None
    if args.conditioning_dropout_prob is not None:
        empty_prompt_embeds, empty_text_ids = compute_text_embeddings("")

    # Pre-compute validation-prompt embeddings.
    validation_prompt_dict = {}
    if args.validation_prompt is not None:
        val_pe, _ = compute_text_embeddings(args.validation_prompt)
        validation_prompt_dict = {"prompt_embeds": val_pe}

    # ------------------------------------------------------------------
    # Unwrap helper
    # ------------------------------------------------------------------
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ------------------------------------------------------------------
    # Custom save / load hooks for LoRA weights
    # ------------------------------------------------------------------
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

            Flux2KleinPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
            # Pop weights so Accelerate doesn't save them a second time.
            for _ in models:
                weights.pop()

    def load_model_hook(models, input_dir):
        transformer_model = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_model = model
            else:
                raise ValueError(f"Unexpected load model: {model.__class__}")

        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(
            transformer_state_dict
        )
        incompatible = set_peft_model_state_dict(
            transformer_model, transformer_state_dict, adapter_name="default"
        )
        if incompatible is not None:
            unexpected = getattr(incompatible, "unexpected_keys", None)
            if unexpected:
                logger.warning(f"Unexpected keys in LoRA state dict: {unexpected}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes: pip install bitsandbytes")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        fused = 'fused' in inspect.signature(optimizer_cls).parameters
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        # Local dataset with metadata.jsonl
        metadata_path = os.path.join(args.train_data_dir, "metadata.jsonl")
        dataset = load_dataset(
            "json",
            data_files={"train": metadata_path},
            cache_dir=args.cache_dir,
        )

        def make_absolute_paths(example):
            example[args.original_image_column] = os.path.join(
                args.train_data_dir, example[args.original_image_column]
            )
            example[args.edited_image_column] = os.path.join(
                args.train_data_dir, example[args.edited_image_column]
            )
            return example

        dataset["train"] = dataset["train"].map(make_absolute_paths)

        from datasets import Image as DatasetsImage

        dataset = dataset.cast_column(args.original_image_column, DatasetsImage())
        dataset = dataset.cast_column(args.edited_image_column, DatasetsImage())

    column_names = dataset["train"].column_names

    # Validate column names.
    original_image_column = args.original_image_column
    if original_image_column not in column_names:
        raise ValueError(
            f"--original_image_column '{original_image_column}' not in: {column_names}"
        )
    edited_image_column = args.edited_image_column
    if edited_image_column not in column_names:
        raise ValueError(
            f"--edited_image_column '{edited_image_column}' not in: {column_names}"
        )
    edit_prompt_column = args.edit_prompt_column
    if edit_prompt_column not in column_names:
        raise ValueError(
            f"--edit_prompt_column '{edit_prompt_column}' not in: {column_names}"
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    # Spatial transforms applied to BOTH images simultaneously so they
    # receive the exact same random crop / flip.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(img, args.resolution) for img in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(img, args.resolution) for img in examples[edited_image_column]]
        )
        # Concatenate before transforms so both get the SAME crop / flip.
        images = np.concatenate([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1  # normalize to [-1, 1]
        return train_transforms(images)

    def preprocess_train(examples):
        preprocessed = preprocess_images(examples)
        original_images, edited_images = preprocessed.chunk(2)
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images
        examples["edit_prompts"] = list(examples[edit_prompt_column])
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=args.seed)
                .select(range(args.max_train_samples))
            )
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack(
            [ex["original_pixel_values"] for ex in examples]
        ).to(memory_format=torch.contiguous_format).float()
        edited_pixel_values = torch.stack(
            [ex["edited_pixel_values"] for ex in examples]
        ).to(memory_format=torch.contiguous_format).float()
        edit_prompts = [ex["edit_prompts"] for ex in examples]
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "edit_prompts": edit_prompts,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # ------------------------------------------------------------------
    # LR scheduler & Accelerator prepare
    # ------------------------------------------------------------------
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Compute total training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  LoRA rank = {args.rank}, alpha = {args.lora_alpha}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info("No checkpoint found. Starting training from scratch.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # ------------------------------------------------------------------
    # Init trackers
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        tracker_config = vars(args).copy()
        accelerator.init_trackers("flux2-klein-lora-finetune", config=tracker_config)

    # ------------------------------------------------------------------
    # get_sigmas helper (closure over noise_scheduler_copy)
    # ------------------------------------------------------------------
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(
            device=accelerator.device, dtype=dtype
        )
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [
            (schedule_timesteps == t).nonzero().item() for t in timesteps
        ]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip already-completed steps when resuming.
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(transformer):
                # ==========================================================
                # 1. Encode BOTH images through the VAE
                # ==========================================================
                # Target (edited) image — the one we learn to denoise.
                target_latents = vae.encode(
                    batch["edited_pixel_values"].to(
                        accelerator.device, dtype=weight_dtype
                    )
                ).latent_dist.mode()

                # Conditioning (original / clean) image.
                cond_latents = vae.encode(
                    batch["original_pixel_values"].to(
                        accelerator.device, dtype=weight_dtype
                    )
                ).latent_dist.mode()

                # ==========================================================
                # 2. Patchify + batch-norm normalize both
                # ==========================================================
                target_latents = Flux2KleinPipeline._patchify_latents(target_latents)
                target_latents = (target_latents - latents_bn_mean) / latents_bn_std

                cond_latents = Flux2KleinPipeline._patchify_latents(cond_latents)
                cond_latents = (cond_latents - latents_bn_mean) / latents_bn_std

                # ==========================================================
                # 3. Prepare position IDs
                # ==========================================================
                # Target uses T=0 (standard latent IDs).
                target_ids = Flux2KleinPipeline._prepare_latent_ids(
                    target_latents
                ).to(device=accelerator.device)

                # Condition uses T=10 offset via _prepare_image_ids.
                # This method expects a list of (1, C, H, W) tensors and
                # returns (1, N_cond, 4).  We create IDs for one sample
                # and expand to the full batch.
                single_cond_ids = Flux2KleinPipeline._prepare_image_ids(
                    [cond_latents[0:1]]
                )  # (1, N_cond, 4)
                cond_ids = single_cond_ids.expand(
                    cond_latents.shape[0], -1, -1
                ).to(device=accelerator.device)

                # ==========================================================
                # 4. Sample noise + flow-matching interpolation (target only)
                # ==========================================================
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]

                # Sample timesteps via density-based sampling.
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (
                    u * noise_scheduler_copy.config.num_train_timesteps
                ).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=target_latents.device
                )

                # Flow-matching interpolation:  zt = (1 − σ) · x₀ + σ · ε
                sigmas = get_sigmas(
                    timesteps,
                    n_dim=target_latents.ndim,
                    dtype=target_latents.dtype,
                )
                noisy_target = (1.0 - sigmas) * target_latents + sigmas * noise

                # ==========================================================
                # 5. Pack to sequences
                # ==========================================================
                packed_noisy_target = Flux2KleinPipeline._pack_latents(
                    noisy_target
                )  # (B, N_target, C)
                packed_cond = Flux2KleinPipeline._pack_latents(
                    cond_latents
                )  # (B, N_cond, C)

                # ==========================================================
                # 6. Encode text prompts
                # ==========================================================
                prompt_embeds, text_ids = compute_text_embeddings(
                    batch["edit_prompts"]
                )
                prompt_embeds = prompt_embeds.to(
                    device=accelerator.device, dtype=weight_dtype
                )
                text_ids = text_ids.to(device=accelerator.device)

                # ==========================================================
                # 7. Conditioning dropout (classifier-free guidance training)
                # ==========================================================
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=accelerator.device)
                    prob = args.conditioning_dropout_prob

                    # Text prompt dropout:  random_p < 2·prob
                    prompt_mask = random_p < 2 * prob
                    if prompt_mask.any():
                        for idx in prompt_mask.nonzero(as_tuple=True)[0]:
                            prompt_embeds[idx] = empty_prompt_embeds[0].to(
                                device=prompt_embeds.device,
                                dtype=prompt_embeds.dtype,
                            )
                            text_ids[idx] = empty_text_ids[0].to(
                                device=text_ids.device, dtype=text_ids.dtype
                            )

                    # Image conditioning dropout:  prob ≤ random_p < 3·prob
                    image_drop = (random_p >= prob) & (random_p < 3 * prob)
                    if image_drop.any():
                        packed_cond[image_drop] = 0.0

                # ==========================================================
                # 8. Concatenate along sequence dimension
                # ==========================================================
                hidden_states = torch.cat(
                    [packed_noisy_target, packed_cond], dim=1
                )
                img_ids = torch.cat([target_ids, cond_ids], dim=1)

                # ==========================================================
                # 9. Handle guidance embedding
                # ==========================================================
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full(
                        [1], args.guidance_scale, device=accelerator.device
                    )
                    guidance = guidance.expand(bsz)
                else:
                    guidance = None

                # ==========================================================
                # 10. Transformer forward pass
                # ==========================================================
                model_pred = transformer(
                    hidden_states=hidden_states,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                # ==========================================================
                # 11. Keep ONLY the target portion of the prediction
                # ==========================================================
                model_pred = model_pred[:, : packed_noisy_target.size(1)]

                # Unpack back to spatial layout for loss computation.
                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(
                    model_pred, target_ids
                )

                # ==========================================================
                # 12. Flow-matching loss
                # ==========================================================
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # Velocity target:  v = noise − x₀
                target = noise - target_latents

                loss = torch.mean(
                    (
                        weighting.float()
                        * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                # Gather losses for logging.
                avg_loss = accelerator.gather(
                    loss.repeat(args.train_batch_size)
                ).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagation.
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # -- Post-step bookkeeping --
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Checkpoint saving.
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                for removing in checkpoints[:num_to_remove]:
                                    path_to_remove = os.path.join(
                                        args.output_dir, removing
                                    )
                                    logger.info(f"Removing checkpoint: {path_to_remove}")
                                    shutil.rmtree(path_to_remove)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # ==============================================================
        # End-of-epoch validation
        # ==============================================================
        if accelerator.is_main_process:
            if (
                args.validation_prompt is not None
                and args.val_image_url is not None
                and (epoch + 1) % args.validation_epochs == 0
            ):
                pipeline = Flux2KleinPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                log_validation(
                    pipeline,
                    args,
                    accelerator,
                    validation_prompt_dict,
                    epoch,
                    weight_dtype,
                )

    # ==================================================================
    # Save final LoRA weights
    # ==================================================================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_model = unwrap_model(transformer)
        transformer_lora_layers = get_peft_model_state_dict(transformer_model)

        Flux2KleinPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        # Final validation with freshly-loaded LoRA.
        if args.validation_prompt is not None and args.val_image_url is not None:
            pipeline = Flux2KleinPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            pipeline.load_lora_weights(args.output_dir)
            log_validation(
                pipeline,
                args,
                accelerator,
                validation_prompt_dict,
                epoch,
                weight_dtype,
                is_final_validation=True,
            )

    accelerator.end_training()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
