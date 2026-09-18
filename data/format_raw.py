#!/usr/bin/env python3
"""
Renames all video files in a directory to raw_1, raw_2, raw_3, ...
while preserving their original file extension (.mp4, .mkv, .webm, etc).

Usage:
    python rename_raw.py /path/to/directory
    python rename_raw.py /path/to/directory --start 1
    python rename_raw.py /path/to/directory --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Extensions considered "video" files. Add/remove as needed.
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v"}


def rename_videos(directory: Path, start: int = 1, dry_run: bool = False, sort_by: str = "name"):
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    # Collect only video files (skip subdirectories and non-video files)
    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not files:
        print("No matching video files found.")
        return

    # Sort for consistent, predictable ordering
    if sort_by == "mtime":
        files.sort(key=lambda f: f.stat().st_mtime)
    else:
        files.sort(key=lambda f: f.name.lower())

    # Step 1: rename to temporary names first to avoid collisions
    # (e.g. if raw_2.mp4 already exists and we're renaming into that slot)
    temp_names = []
    for f in files:
        temp_path = f.with_name(f".__tmp__{f.name}")
        if not dry_run:
            f.rename(temp_path)
        temp_names.append((temp_path, f.suffix))

    # Step 2: rename from temp names to final raw_N names
    for i, (temp_path, ext) in enumerate(temp_names, start=start):
        final_name = f"raw_{i}{ext.lower()}"
        final_path = directory / final_name
        original_display = temp_path.name.replace(".__tmp__", "", 1) if dry_run else "(temp)"

        if dry_run:
            print(f"Would rename: {original_display}  ->  {final_name}")
        else:
            temp_path.rename(final_path)
            print(f"Renamed: {final_name}")

    if dry_run:
        print(f"\n{len(files)} file(s) would be renamed. Run without --dry-run to apply.")
    else:
        print(f"\nDone. {len(files)} file(s) renamed.")


def main():
    parser = argparse.ArgumentParser(description="Rename video files to raw_1, raw_2, ... preserving extensions.")
    parser.add_argument("directory", type=str, help="Path to the directory containing video files")
    parser.add_argument("--start", type=int, default=1, help="Starting number (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without renaming anything")
    parser.add_argument(
        "--sort-by",
        choices=["name", "mtime"],
        default="name",
        help="Order to assign numbers: alphabetical name (default) or file modified time",
    )
    args = parser.parse_args()

    rename_videos(Path(args.directory), start=args.start, dry_run=args.dry_run, sort_by=args.sort_by)


if __name__ == "__main__":
    main()