#!/usr/bin/env python3
"""
check_processed.py
-------------------
Audits every clip_XXX/ folder in processed/ against the spec:
  - exactly 16 frames, named frame_000.png .. frame_015.png
  - each frame 512x512, RGB (no alpha)
  - caption.txt exists and is non-empty

USAGE
  python check_processed.py [processed_dir]   # defaults to ./processed

Requires Pillow (pip install pillow).
"""

import re
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("error: Pillow is required — pip install pillow")

TARGET_FRAMES = 16
TARGET_RES = 512
CLIP_RE = re.compile(r"^clip_(\d{3})$")
FRAME_RE = re.compile(r"^frame_(\d{3})\.png$")


def check_clip(clip_dir: Path) -> list:
    problems = []
    frames_dir = clip_dir / "frames"
    caption_path = clip_dir / "caption.txt"

    if not frames_dir.is_dir():
        return [f"missing frames/ folder"]

    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) != TARGET_FRAMES:
        problems.append(f"{len(frames)} frames, expected {TARGET_FRAMES}")

    expected_indices = set(range(TARGET_FRAMES))
    seen_indices = set()
    for f in frames:
        m = FRAME_RE.match(f.name)
        if not m:
            problems.append(f"unexpected filename in frames/: {f.name}")
            continue
        seen_indices.add(int(m.group(1)))

        try:
            with Image.open(f) as im:
                if im.mode != "RGB":
                    problems.append(f"{f.name}: mode {im.mode}, expected RGB (has alpha?)")
                if im.size != (TARGET_RES, TARGET_RES):
                    problems.append(f"{f.name}: size {im.size}, expected {TARGET_RES}x{TARGET_RES}")
        except Exception as e:
            problems.append(f"{f.name}: couldn't open ({e})")

    missing = expected_indices - seen_indices
    if missing and len(frames) == TARGET_FRAMES:
        # only worth reporting separately if count matched but indices didn't
        problems.append(f"frame index gaps: {sorted(missing)}")

    if not caption_path.exists():
        problems.append("missing caption.txt")
    elif not caption_path.read_text().strip():
        problems.append("caption.txt is empty")

    return problems


def main():
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed")
    if not processed_dir.exists():
        sys.exit(f"error: {processed_dir} does not exist")

    clips = sorted(d for d in processed_dir.iterdir() if d.is_dir() and CLIP_RE.match(d.name))
    if not clips:
        sys.exit(f"no clip_XXX folders found in {processed_dir}")

    total_problems = 0
    for clip_dir in clips:
        problems = check_clip(clip_dir)
        if problems:
            print(f"✗ {clip_dir.name}:")
            for p in problems:
                print(f"    - {p}")
            total_problems += len(problems)
        else:
            print(f"✓ {clip_dir.name}: OK")

    print(f"\n{len(clips)} clip(s) checked, {total_problems} problem(s) found.")
    sys.exit(1 if total_problems else 0)


if __name__ == "__main__":
    main()