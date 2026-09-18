#!/usr/bin/env python3
"""
scripts/relabel_windows.py
----------------------------
Re-captions every window in data/windows/*/ using a vision-capable Claude
model, replacing the BLIP-generated captions with more accurate ones — and
flags windows that are just static/noise (not real drawn content, e.g. the
title-card/transition static that shows up at the start of some clips) so
they can be excluded from training instead of teaching the motion module
that "blank noise" is a valid thing to draw.

Sends 3 frames per window (first, middle, last) for temporal context, asks
for a structured JSON response: {"is_valid": bool, "caption": str}.

Nothing is deleted. Results are written to:
  - data/windows/<window>/caption.txt   (overwritten, only when is_valid)
  - data/excluded_windows.txt           (one window_id per line)
scripts/dataset.py's WindowDataset skips anything listed there.

USAGE
  python scripts/relabel_windows.py                  # all windows
  python scripts/relabel_windows.py --limit 10        # sample first
  python scripts/relabel_windows.py --dry-run         # print, don't write
"""

import argparse
import base64
import json
from pathlib import Path

import anthropic

WINDOWS_DIR = Path("data/windows")
EXCLUDED_PATH = Path("data/excluded_windows.txt")
MODEL = "claude-haiku-4-5"
STYLE_SUFFIX = ", hand-drawn flipbook animation style"

SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {
            "type": "boolean",
            "description": "true if these frames show real hand-drawn flipbook content; false if "
                            "they're blank, static/noise, a title card, or otherwise not actual drawn animation.",
        },
        "caption": {
            "type": "string",
            "description": "One concise sentence describing the subject and action, present tense. "
                            "Empty string if is_valid is false.",
        },
    },
    "required": ["is_valid", "caption"],
    "additionalProperties": False,
}

PROMPT = """These are 3 frames (first, middle, last) sampled from a 16-frame window of a hand-drawn flipbook-style animation clip.

Look at them and decide:
1. is_valid: Are these actually hand-drawn animation frames of a real subject/character/scene? Set this to false if the frames are blank, look like TV static/visual noise, a solid color, a title card, or otherwise aren't real drawn content (this happens sometimes at the start of a clip).
2. caption: If is_valid, write ONE concise sentence describing the subject and the action/pose, present tense, similar in style to: "a stick figure walking", "a cartoon drawing of a man with a cigarette", "a black and white drawing of an ufo ship". Don't include any style/medium description (no "hand-drawn", "flipbook", "black and white", etc.) — just the subject and action. Leave empty ("") if is_valid is false."""


def encode_image(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def caption_window(client, window_dir: Path) -> dict:
    frame_paths = sorted((window_dir / "frames").glob("frame_*.png"))
    sample = [frame_paths[0], frame_paths[len(frame_paths) // 2], frame_paths[-1]]

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{
            "role": "user",
            "content": [encode_image(p) for p in sample] + [{"type": "text", "text": PROMPT}],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Re-caption windows with Claude vision + flag static/noise windows.")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N windows")
    parser.add_argument("--dry-run", action="store_true", help="print results without writing files")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    window_dirs = sorted(d for d in WINDOWS_DIR.iterdir() if d.is_dir() and (d / "frames").is_dir())
    if args.limit:
        window_dirs = window_dirs[:args.limit]

    excluded = []
    n_valid = n_invalid = n_errors = 0

    for window_dir in window_dirs:
        try:
            result = caption_window(client, window_dir)
        except Exception as e:
            print(f"  ERROR on {window_dir.name}: {e}")
            n_errors += 1
            continue

        if result["is_valid"]:
            caption = result["caption"].strip().rstrip(".") + STYLE_SUFFIX
            print(f"OK   {window_dir.name}: {caption}")
            n_valid += 1
            if not args.dry_run:
                (window_dir / "caption.txt").write_text(caption, encoding="utf-8")
        else:
            print(f"SKIP {window_dir.name}: flagged as static/noise")
            excluded.append(window_dir.name)
            n_invalid += 1

    if not args.dry_run and excluded:
        with EXCLUDED_PATH.open("a", encoding="utf-8") as f:
            for name in excluded:
                f.write(name + "\n")

    print(f"\n{n_valid} captioned, {n_invalid} flagged as static/noise, {n_errors} errors.")
    if not args.dry_run:
        print(f"Excluded window IDs appended to {EXCLUDED_PATH}")


if __name__ == "__main__":
    main()
