# scripts/make_windows.py
import os
import shutil
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
WINDOWS_DIR = Path("data/windows")
WINDOW_SIZE = 16
STRIDE = 16  # non-overlapping; set to 8 for overlapping windows

def make_windows():
    WINDOWS_DIR.mkdir(exist_ok=True)
    
    clip_dirs = sorted(PROCESSED_DIR.glob("clip_*"))
    total_windows = 0

    for clip_dir in clip_dirs:
        frames_dir = clip_dir / "frames"
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        num_frames = len(frame_files)

        if num_frames < WINDOW_SIZE:
            print(f"Skipping {clip_dir.name}: only {num_frames} frames")
            continue

        window_idx = 0
        start = 0
        while start + WINDOW_SIZE <= num_frames:
            window_frames = frame_files[start:start + WINDOW_SIZE]
            
            window_name = f"{clip_dir.name}_w{window_idx:03d}"
            window_out_dir = WINDOWS_DIR / window_name / "frames"
            window_out_dir.mkdir(parents=True, exist_ok=True)

            for i, src_frame in enumerate(window_frames):
                dst_frame = window_out_dir / f"frame_{i:03d}.png"
                shutil.copy(src_frame, dst_frame)

            total_windows += 1
            window_idx += 1
            start += STRIDE

        print(f"{clip_dir.name}: {num_frames} frames -> {window_idx} windows")

    print(f"\nTotal windows created: {total_windows}")

if __name__ == "__main__":
    make_windows()