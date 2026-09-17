#!/usr/bin/env python3
"""
scripts/load_models.py
-----------------------
Loads SD1.5 (VAE, UNet, tokenizer, text encoder) + the AnimateDiff motion
module, wires them into a single motion-aware UNet, and freezes everything
except the motion module — that's the only part AnimateDiff fine-tuning
actually trains.

Run directly for a sanity check: loads models, pulls one batch from
scripts/dataset.py's DataLoader, encodes frames to VAE latents, and runs one
forward pass through the motion UNet to confirm every shape lines up before
we write the real training loop.

USAGE
  python scripts/load_models.py
"""

import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, MotionAdapter, UNet2DConditionModel, UNetMotionModel
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from dataset import TOKENIZER_NAME, get_dataloader

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models(device=None, dtype=torch.float32):
    """Returns (tokenizer, text_encoder, vae, unet, noise_scheduler), all on `device`."""
    device = device or get_device()

    print(f"Loading tokenizer + text encoder from {SD_MODEL_ID} ...")
    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD_MODEL_ID, subfolder="text_encoder")

    print(f"Loading VAE from {SD_MODEL_ID} ...")
    vae = AutoencoderKL.from_pretrained(SD_MODEL_ID, subfolder="vae")

    print(f"Loading base UNet from {SD_MODEL_ID} ...")
    base_unet = UNet2DConditionModel.from_pretrained(SD_MODEL_ID, subfolder="unet")

    print(f"Loading motion adapter from {MOTION_ADAPTER_ID} ...")
    motion_adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID)

    print("Wiring motion adapter into UNet ...")
    unet = UNetMotionModel.from_unet2d(base_unet, motion_adapter)

    noise_scheduler = DDPMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")

    # Freeze everything except the motion module. That's the whole point of
    # AnimateDiff fine-tuning: the spatial UNet/VAE/text-encoder stay as
    # pretrained SD1.5, only the temporal motion layers get updated.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    for name, param in unet.named_parameters():
        param.requires_grad_("motion_modules" in name)

    vae.to(device, dtype=dtype).eval()
    text_encoder.to(device, dtype=dtype).eval()
    unet.to(device, dtype=dtype)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"UNet trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    return tokenizer, text_encoder, vae, unet, noise_scheduler


@torch.no_grad()
def encode_latents(vae, pixel_values, device, dtype):
    """[B, T, C, H, W] pixel values in [-1, 1] -> [B, T, 4, H/8, W/8] VAE latents."""
    b, t, c, h, w = pixel_values.shape
    flat = pixel_values.reshape(b * t, c, h, w).to(device, dtype=dtype)
    latents = vae.encode(flat).latent_dist.sample() * vae.config.scaling_factor
    return latents.reshape(b, t, *latents.shape[1:])


def main():
    device = get_device()
    dtype = torch.float32  # keep fp32 for the sanity check; switch to fp16/bf16 + accelerate for real training
    print(f"Device: {device}\n")

    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(device, dtype)

    print("\nPulling one batch from the DataLoader ...")
    loader = get_dataloader(batch_size=1, shuffle=True, tokenizer=tokenizer)
    batch = next(iter(loader))
    print(f"  pixel_values: {tuple(batch['pixel_values'].shape)}")
    print(f"  caption:      {batch['caption']}")

    print("\nEncoding frames to VAE latents ...")
    latents = encode_latents(vae, batch["pixel_values"], device, dtype)
    print(f"  latents: {tuple(latents.shape)}")  # [B, T, 4, 64, 64]

    print("\nRunning one forward pass through the motion UNet ...")
    latents_btcwh = latents.permute(0, 2, 1, 3, 4)  # UNetMotionModel expects [B, C, T, H, W]
    noise = torch.randn_like(latents_btcwh)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents_btcwh.shape[0],), device=device)
    noisy_latents = noise_scheduler.add_noise(latents_btcwh, noise, timesteps)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
        # UNetMotionModel flattens [B, T, ...] into batch B*T for its spatial attention
        # layers, so encoder_hidden_states (batch B) must be expanded to match (batch B*T)
        # before every frame's cross-attention can see the caption embedding.
        num_frames = latents_btcwh.shape[2]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

    print(f"  noise_pred: {tuple(noise_pred.shape)}")
    assert noise_pred.shape == latents_btcwh.shape, "shape mismatch between prediction and target!"
    print("\nEnd-to-end shapes check out. Ready to write the training loop.")


if __name__ == "__main__":
    main()
