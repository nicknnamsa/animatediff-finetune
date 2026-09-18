# AnimateDiff Flipbook Fine-Tune

Fine-tuning AnimateDiff's motion module (on top of frozen Stable Diffusion 1.5)
to generate short clips in a hand-drawn flipbook animation style, from a
personal dataset of flipbook-style video clips.

Only the motion module is trained (~453M of the combined model's 1.31B
params, ~34.5%) — the VAE, text encoder, and SD1.5's spatial UNet layers stay
frozen. The base model already knows how to draw; this only teaches it how
*your* clips move.

## Status

🚧 In progress. A first full training run (15 epochs) completed but produced
inconsistent results — some checkpoints generated clean, on-style clips,
others collapsed into blank/static output. Root-caused two data issues (see
below) and retrained on a cleaned dataset with better captions, which
noticeably improved consistency, but quality still isn't where it needs to
be. Currently investigating: dataset size (669 training windows is small for
motion fine-tuning), a proper learning-rate schedule (the current runs use a
flat LR with no warmup/decay), and EMA (exponential moving average) of the
trained weights, none of which have been tried yet.

## Pipeline

```
raw videos                                (data/raw/)
  -> extract frames per clip              scripts/process_to_dataset.py
  -> chop into 16-frame windows            scripts/make_windows.py
  -> caption + flag static/noise windows   scripts/relabel_windows.py
  -> Dataset/DataLoader (resize, tokenize) scripts/dataset.py
  -> load SD1.5 + motion adapter, freeze   scripts/load_models.py
     everything but the motion module
  -> train the motion module               scripts/train.py
  -> generate a sample clip from a         scripts/generate.py
     checkpoint + prompt
  -> generate one comparison clip per      scripts/sample_epochs.py
     epoch checkpoint (fixed prompt/seed)
```

## Scripts

| Script | Role |
| --- | --- |
| `format_raw.py` | Renames source videos to `raw_N.ext` |
| `process_to_dataset.py` | Extracts frames from raw videos into `processed/<clip>/frames/` + an empty `caption.txt` |
| `make_windows.py` | Chops each clip's frames into non-overlapping 16-frame windows under `data/windows/` |
| `caption_windows.py` | Original BLIP-based auto-captioner (superseded by `relabel_windows.py`) |
| `relabel_windows.py` | Re-captions every window with a vision-capable Claude model and flags windows that are just static/noise (common at the start of some clips) for exclusion — writes `data/excluded_windows.txt` |
| `check_processed.py` | Sanity-checks `processed/` clips (frame count, resolution, non-empty captions) |
| `check_setup.py` | Confirms PyTorch + MPS/CUDA are working |
| `dataset.py` | `WindowDataset`/`DataLoader` — reads windows, resizes/crops to 512x512, tokenizes captions, skips anything in `excluded_windows.txt` |
| `load_models.py` | Loads SD1.5 + the AnimateDiff motion adapter, wires them into one `UNetMotionModel`, freezes everything except the motion module |
| `train.py` | The training loop — noise prediction loss, AdamW, gradient checkpointing, saves a checkpoint per epoch to `--output-dir` |
| `generate.py` | Loads a trained checkpoint into the pipeline and generates one clip from a prompt |
| `sample_epochs.py` | Runs `generate.py`'s logic once per epoch checkpoint with a fixed prompt/seed, for an apples-to-apples comparison across a training run |

## Setup

Requires Python 3.11+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

Training runs on a Runpod GPU pod (a Mac's MPS backend is far too slow, and
runs into memory issues with AnimateDiff's attention layers). The general
loop:

1. Provision a pod + network volume, upload `data/windows/`, `scripts/`, and set up a Python venv on the volume (so it persists across pods).
2. `python scripts/train.py --epochs 15 --batch-size 4` (tune `--batch-size` to the GPU's VRAM; gradient checkpointing is on by default).
3. `python scripts/sample_epochs.py` to generate one comparison clip per epoch.
4. Download results, terminate the pod to stop billing. The network volume (dataset + checkpoints + Python env) can be kept for the next run.

## Generating a clip

```bash
python scripts/generate.py \
  --checkpoint path/to/motion_module_epochNN.safetensors \
  --prompt "a stick figure walking, hand-drawn flipbook animation style" \
  --output out.gif
```

## Structure

- `data/` — `raw/` (source videos), `processed/` (extracted frames per clip), `windows/` (16-frame training windows + captions), `metadata.jsonl`, `excluded_windows.txt` — all gitignored except small text files (this is 2GB+ of binary media, kept out of git)
- `scripts/` — the pipeline above
- `outputs/` — generated samples, checkpoints (gitignored)
- `notebooks/` — exploration and testing
