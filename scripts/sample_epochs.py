#!/usr/bin/env python3
"""
scripts/sample_epochs.py
-------------------------
Generates one comparison GIF per epoch checkpoint from scripts/train.py, using
a FIXED prompt + seed so all of them are directly comparable — a visual record
of how the motion module's style/motion changes over the training run.

USAGE
  python scripts/sample_epochs.py --checkpoint-dir /workspace/outputs --output-dir /workspace/outputs/samples
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate import build_pipeline, generate_sample, load_checkpoint
from load_models import get_device

EPOCH_RE = re.compile(r"epoch(\d+)")


def main():
    parser = argparse.ArgumentParser(description="Generate one comparison GIF per epoch checkpoint.")
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/outputs")
    parser.add_argument("--output-dir", type=str, default="/workspace/outputs/samples")
    parser.add_argument("--prompt", type=str, default="a stick figure walking, hand-drawn flipbook animation style")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, static")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoints = sorted(
        Path(args.checkpoint_dir).glob("motion_module_epoch*.safetensors"),
        key=lambda p: int(EPOCH_RE.search(p.name).group(1)),
    )
    if not checkpoints:
        sys.exit(f"no motion_module_epoch*.safetensors found under {args.checkpoint_dir}")

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building pipeline on {device} ...", flush=True)
    pipe = build_pipeline(device)

    for ckpt in checkpoints:
        epoch = int(EPOCH_RE.search(ckpt.name).group(1))
        load_checkpoint(pipe, ckpt)
        out_path = output_dir / f"epoch{epoch:02d}.gif"
        print(f"epoch {epoch}: generating -> {out_path}", flush=True)
        generate_sample(pipe, args.prompt, args.negative_prompt, args.num_frames,
                         args.steps, args.guidance_scale, args.seed, out_path, device)

    print("SAMPLING_DONE", flush=True)


if __name__ == "__main__":
    main()
