#!/usr/bin/env python3
"""
scripts/dataset.py
-------------------
PyTorch Dataset/DataLoader for AnimateDiff motion-module fine-tuning.

Reads data/windows/<window_id>/frames/frame_000.png..frame_015.png + caption.txt,
resizes frames to the target training resolution, and tokenizes captions with
the SD1.5 CLIP tokenizer.

USAGE
  python scripts/dataset.py                     # sanity-check: load a batch, print shapes
  python scripts/dataset.py --resize pad --n 4   # preview a few batches with padding instead of crop
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

WINDOWS_DIR = Path("data/windows")
TOKENIZER_NAME = "openai/clip-vit-large-patch14"  # SD1.5's text encoder tokenizer
TOKENIZER_MAX_LENGTH = 77


class WindowDataset(Dataset):
    """
    One sample = one 16-frame window: caption + [T, C, H, W] tensor in [-1, 1].

    resize_mode:
      "crop" (default) - resize shorter edge to target, then center/random-crop
                          to target x target. Standard SD training recipe, but
                          on a 640x360 source hitting 512x512 crops ~44% of the
                          width off (22% each side).
      "pad"             - resize longer edge to target, letterbox-pad the rest
                           with black. Keeps every pixel of the original frame,
                           at the cost of bars the model has to learn to ignore.

    The crop/pad placement is computed once per window and reused across all
    16 frames — cropping each frame independently would shift the frame
    differently every step and destroy the motion signal being trained on.
    """

    def __init__(self, windows_dir=WINDOWS_DIR, target_size=(512, 512),
                 resize_mode="crop", augment=False, tokenizer=None):
        self.windows_dir = Path(windows_dir)
        self.target_h, self.target_w = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        self.resize_mode = resize_mode
        self.augment = augment  # random crop placement instead of center crop (train split only)

        self.window_dirs = sorted(
            d for d in self.windows_dir.iterdir()
            if d.is_dir() and (d / "caption.txt").exists() and (d / "frames").is_dir()
        )
        if not self.window_dirs:
            raise RuntimeError(f"no windows with caption.txt + frames/ found under {self.windows_dir}")

        if tokenizer is None:
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_NAME)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.window_dirs)

    def _load_frames(self, frames_dir: Path) -> torch.Tensor:
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        w, h = images[0].size

        if self.resize_mode == "crop":
            scale = max(self.target_w / w, self.target_h / h)
            new_w, new_h = round(w * scale), round(h * scale)
            resize = transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)

            max_left, max_top = new_w - self.target_w, new_h - self.target_h
            if self.augment:
                left = torch.randint(0, max_left + 1, (1,)).item()
                top = torch.randint(0, max_top + 1, (1,)).item()
            else:
                left, top = max_left // 2, max_top // 2

            frames = [F.to_tensor(F.crop(resize(img), top, left, self.target_h, self.target_w)) for img in images]

        elif self.resize_mode == "pad":
            scale = min(self.target_w / w, self.target_h / h)
            new_w, new_h = round(w * scale), round(h * scale)
            resize = transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)

            pad_left, pad_top = (self.target_w - new_w) // 2, (self.target_h - new_h) // 2
            pad_right = self.target_w - new_w - pad_left
            pad_bottom = self.target_h - new_h - pad_top

            frames = [F.to_tensor(F.pad(resize(img), [pad_left, pad_top, pad_right, pad_bottom], fill=0)) for img in images]

        else:
            raise ValueError(f"unknown resize_mode: {self.resize_mode!r}")

        video = torch.stack(frames)  # [T, C, H, W], float in [0, 1]
        return video * 2 - 1  # -> [-1, 1] to match SD's VAE input range

    def __getitem__(self, idx):
        window_dir = self.window_dirs[idx]
        caption = (window_dir / "caption.txt").read_text(encoding="utf-8").strip()
        pixel_values = self._load_frames(window_dir / "frames")

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=TOKENIZER_MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,               # [T, C, H, W]
            "input_ids": tokens.input_ids[0],            # [77]
            "attention_mask": tokens.attention_mask[0],  # [77]
            "caption": caption,
            "window_id": window_dir.name,
        }


def get_dataloader(windows_dir=WINDOWS_DIR, target_size=(512, 512), resize_mode="crop",
                    augment=False, batch_size=2, shuffle=True, num_workers=0, tokenizer=None):
    dataset = WindowDataset(windows_dir, target_size, resize_mode, augment, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(description="Sanity-check the window Dataset/DataLoader.")
    parser.add_argument("--resize", choices=["crop", "pad"], default="crop")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n", type=int, default=1, help="number of batches to preview")
    args = parser.parse_args()

    loader = get_dataloader(target_size=(args.size, args.size), resize_mode=args.resize, batch_size=args.batch_size)
    print(f"{len(loader.dataset)} windows found.\n")

    for i, batch in enumerate(loader):
        if i >= args.n:
            break
        pv = batch["pixel_values"]
        print(f"batch {i}:")
        print(f"  pixel_values: {tuple(pv.shape)} range [{pv.min():.2f}, {pv.max():.2f}]")
        print(f"  input_ids:    {tuple(batch['input_ids'].shape)}")
        print(f"  window_ids:   {batch['window_id']}")
        print(f"  captions:     {batch['caption']}")


if __name__ == "__main__":
    main()
