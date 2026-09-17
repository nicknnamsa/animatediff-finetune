#!/usr/bin/env python3
"""
scripts/generate.py
--------------------
Loads SD1.5 + the AnimateDiff motion adapter, swaps in a trained motion-module
checkpoint (from scripts/train.py), and generates a sample clip as a GIF — a
quick visual check of what N epochs of fine-tuning actually did.

USAGE
  python scripts/generate.py --checkpoint /workspace/outputs/motion_module_step000655.safetensors \
      --prompt "a stick figure walking, hand-drawn flipbook animation style" --output sample.gif
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from load_models import MOTION_ADAPTER_ID, SD_MODEL_ID, get_device, load_models


def build_pipeline(device):
    """Loads SD1.5 + the pretrained motion adapter once; swap in checkpoints afterward with load_checkpoint()."""
    tokenizer, text_encoder, vae, unet, _ = load_models(device, torch.float32)
    unet.eval()

    # required by the pipeline's constructor signature even though unet is already wired; unused since
    # `isinstance(unet, UNet2DConditionModel)` is False (see diffusers pipeline_animatediff.py __init__)
    motion_adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID)
    scheduler = DDIMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")

    pipe = AnimateDiffPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, motion_adapter=motion_adapter, scheduler=scheduler,
    )
    pipe.to(device)
    return pipe


def load_checkpoint(pipe, checkpoint_path):
    """Overwrites pipe.unet's motion-module weights in place with a scripts/train.py checkpoint."""
    state_dict = load_file(checkpoint_path)
    _, unexpected = pipe.unet.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    return state_dict


def generate_sample(pipe, prompt, negative_prompt, num_frames, steps, guidance_scale, seed, output_path, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    export_to_gif(output.frames[0], str(output_path))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a sample clip from a trained motion module checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a stick figure walking, hand-drawn flipbook animation style")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, static")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="sample.gif")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}", flush=True)

    pipe = build_pipeline(device)

    print(f"Loading motion module checkpoint from {args.checkpoint} ...", flush=True)
    state_dict = load_checkpoint(pipe, args.checkpoint)
    print(f"  loaded {len(state_dict)} trained tensors", flush=True)

    print(f"Generating {args.num_frames} frames for: {args.prompt!r}", flush=True)
    generate_sample(pipe, args.prompt, args.negative_prompt, args.num_frames,
                     args.steps, args.guidance_scale, args.seed, args.output, device)
    print(f"Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
