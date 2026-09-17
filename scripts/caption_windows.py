# scripts/caption_windows.py
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
# For BLIP-2 instead, swap the two lines above for:
# from transformers import Blip2Processor, Blip2ForConditionalGeneration

WINDOWS_DIR = Path("data/windows")
METADATA_PATH = Path("data/metadata.jsonl")
STYLE_SUFFIX = ", hand-drawn flipbook animation style"

MODEL_NAME = "Salesforce/blip-image-captioning-base"
# For BLIP-2 instead: "Salesforce/blip2-opt-2.7b"

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model():
    device = get_device()
    print(f"Loading {MODEL_NAME} on {device}...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model, device

def caption_window(window_dir, processor, model, device):
    frames_dir = window_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        return None

    # Use the middle frame as representative of the window's action
    middle_frame = frame_files[len(frame_files) // 2]
    image = Image.open(middle_frame).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption.strip() + STYLE_SUFFIX

def main():
    processor, model, device = load_model()
    window_dirs = sorted(WINDOWS_DIR.glob("clip_*_w*"))
    print(f"Found {len(window_dirs)} windows to caption")

    results = []
    for i, window_dir in enumerate(window_dirs):
        caption = caption_window(window_dir, processor, model, device)
        if caption is None:
            print(f"  Skipping {window_dir.name}: no frames found")
            continue

        # Save caption alongside the window's frames
        caption_path = window_dir / "caption.txt"
        caption_path.write_text(caption)

        frame_count = len(list((window_dir / "frames").glob("frame_*.png")))
        results.append({
            "window_id": window_dir.name,
            "caption": caption,
            "num_frames": frame_count,
            "resolution": Image.open(next((window_dir / "frames").glob("frame_*.png"))).size
        })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(window_dirs)} captioned")

    with open(METADATA_PATH, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone. Wrote {len(results)} entries to {METADATA_PATH}")

if __name__ == "__main__":
    main()