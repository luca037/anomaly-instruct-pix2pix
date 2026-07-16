import os
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

print("Loading trained UNet...")

# Load JUST the UNet from your intermediate checkpoint
trained_unet = UNet2DConditionModel.from_pretrained("/home/luca_piai/big_disk/mirage-anomaly-model-fine_tune_pix2pix/checkpoint-1000/unet")

print("Merging into base pipeline...")
# Load the base model, but swap in your newly trained UNet
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", unet=trained_unet)

print("Saving new full model...")
# Save this entire combo as a brand new, fully-formed model
pipe.save_pretrained("/home/luca_piai/big_disk/mirage_resume_base")

print("Done! You can now use 'mirage_resume_base' in your training script.")