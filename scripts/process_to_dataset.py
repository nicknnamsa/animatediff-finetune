#!/usr/bin/env python3
"""
Post-processes video files (.mp4, .mkv, .webm, ...) into a training-ready
directory structure:

    processed/
    ├── clip_001/
    │   ├── frames/
    │   │   ├── frame_000.png
    │   │   ├── frame_001.png
    │   │   └── ...
    │   └── caption.txt
    ├── clip_002/
    │   ├── frames/
    │   └── caption.txt
    └── ...

Frames are extracted as lossless PNG, RGB (no alpha channel).
caption.txt is created empty (or with a placeholder) for you to fill in
by hand, or you can supply captions up front via a mapping file.

Requires ffmpeg to be installed and on your PATH.

Usage:
    python process_clips.py /path/to/videos
    python process_clips.py /path/to/videos --fps 8
    python process_clips.py /path/to/videos --output processed --start 1
    python process_clips.py /path/to/videos --captions captions.txt
    python process_clips.py /path/to/videos --dry-run

Captions file format (optional, one line per clip, tab or "::" separated):
    raw_1.mp4::A cat walking across a kitchen counter
    raw_2.mkv::Time-lapse of clouds moving over a mountain
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v"}


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found on PATH. Install it first (e.g. `apt install ffmpeg`, "
              "`brew install ffmpeg`, or from https://ffmpeg.org/download.html).")
        sys.exit(1)


def load_captions(captions_path: Path) -> dict:
    """Parses an optional captions file into {source_filename: caption_text}."""
    mapping = {}
    if not captions_path:
        return mapping
    if not captions_path.is_file():
        print(f"Warning: captions file '{captions_path}' not found, skipping.")
        return mapping

    for line in captions_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "::" in line:
            fname, caption = line.split("::", 1)
        elif "\t" in line:
            fname, caption = line.split("\t", 1)
        else:
            continue
        mapping[fname.strip()] = caption.strip()
    return mapping


def extract_frames(video_path: Path, frames_dir: Path, fps: float, dry_run: bool):
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(frames_dir / "frame_%03d.png")

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if fps:
        cmd += ["-vf", f"fps={fps},format=rgb24"]
    else:
        cmd += ["-vf", "format=rgb24"]
    cmd += ["-pix_fmt", "rgb24", out_pattern]

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"  ffmpeg failed on {video_path.name}:\n{result.stderr.decode(errors='ignore')}")
        return

    # Re-number frames starting at frame_000 instead of ffmpeg's default frame_001
    frames = sorted(frames_dir.glob("frame_*.png"))
    for f in frames:
        f.rename(frames_dir / f".__tmp__{f.name}")
    temp_frames = sorted(frames_dir.glob(".__tmp__frame_*.png"))
    for i, f in enumerate(temp_frames):
        f.rename(frames_dir / f"frame_{i:03d}.png")


def process_videos(input_dir: Path, output_dir: Path, fps: float, start: int,
                    captions_path: Path, dry_run: bool):
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a valid directory.")
        sys.exit(1)

    videos = sorted(
        [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda f: f.name.lower(),
    )
    if not videos:
        print("No matching video files found.")
        return

    captions = load_captions(captions_path)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, video in enumerate(videos, start=start):
        clip_name = f"clip_{i:03d}"
        clip_dir = output_dir / clip_name
        frames_dir = clip_dir / "frames"
        caption_file = clip_dir / "caption.txt"
        caption_text = captions.get(video.name, "")

        print(f"[{clip_name}] {video.name}")

        if dry_run:
            extract_frames(video, frames_dir, fps, dry_run=True)
            print(f"  Would write: {caption_file} (caption: {caption_text or '<empty, fill in manually>'})")
            continue

        clip_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(video, frames_dir, fps, dry_run=False)
        caption_file.write_text(caption_text, encoding="utf-8")

    if dry_run:
        print(f"\n{len(videos)} clip(s) would be created under '{output_dir}'.")
    else:
        print(f"\nDone. {len(videos)} clip(s) written to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos into clip_XXX/frames + caption.txt structure.")
    parser.add_argument("input_dir", type=str, help="Directory containing source video files")
    parser.add_argument("--output", type=str, default="processed", help="Output directory (default: processed)")
    parser.add_argument("--fps", type=float, default=0,
                         help="Frames per second to extract (default: extract every frame, i.e. native fps)")
    parser.add_argument("--start", type=int, default=1, help="Starting clip number (default: 1)")
    parser.add_argument("--captions", type=str, default=None,
                         help="Optional path to a captions mapping file (filename::caption per line)")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing anything")
    args = parser.parse_args()

    if not args.dry_run:
        check_ffmpeg()

    process_videos(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output),
        fps=args.fps,
        start=args.start,
        captions_path=Path(args.captions) if args.captions else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()