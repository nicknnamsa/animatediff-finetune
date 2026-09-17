#!/usr/bin/env python3
"""
scripts/train.py
-----------------
Fine-tunes the AnimateDiff motion module on data/windows/*/ (SD1.5 UNet + VAE +
text encoder stay frozen, per scripts/load_models.py — only the motion module's
temporal attention layers get gradients). Checkpoints + logs to --output-dir so
a run on a remote pod can be launched detached and monitored by tailing the log.

USAGE
  python scripts/train.py --epochs 3 --batch-size 2 --lr 1e-5
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from dataset import WindowDataset
from load_models import encode_latents, get_device, load_models

DEFAULT_OUTPUT_DIR = "/workspace/outputs" if Path("/workspace").is_dir() else "outputs"


def motion_module_state_dict(unet):
    """Only the trained params are worth checkpointing — the rest is unmodified pretrained SD1.5."""
    return {name: p.detach().to("cpu", torch.float32) for name, p in unet.named_parameters() if "motion_modules" in name}


def save_checkpoint(unet, output_dir: Path, tag: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"motion_module_{tag}.safetensors"
    save_file(motion_module_state_dict(unet), str(path))
    return path


def split_train_val(n: int, val_fraction: float, seed: int = 0):
    n_val = max(1, int(n * val_fraction))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[n_val:], perm[:n_val]


def compute_loss(unet, vae, text_encoder, noise_scheduler, batch, device, use_amp):
    latents = encode_latents(vae, batch["pixel_values"], device, torch.float32)  # [B, T, 4, 64, 64]
    latents = latents.permute(0, 2, 1, 3, 4)  # [B, 4, T, 64, 64] — what UNetMotionModel expects

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
    num_frames = latents.shape[2]
    # spatial attention flattens [B, T, ...] to batch B*T, so captions need matching expansion (see load_models.py)
    encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

    return torch.nn.functional.mse_loss(noise_pred.float(), noise.float())


@torch.no_grad()
def evaluate(unet, vae, text_encoder, noise_scheduler, val_loader, device, use_amp):
    unet.eval()
    total, count = 0.0, 0
    for batch in val_loader:
        total += compute_loss(unet, vae, text_encoder, noise_scheduler, batch, device, use_amp).item()
        count += 1
    unet.train()
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the AnimateDiff motion module.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=200, help="steps between checkpoints")
    parser.add_argument("--log-every", type=int, default=10, help="steps between loss prints")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    device = get_device()
    use_amp = device == "cuda"
    print(f"Device: {device}", flush=True)

    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(device, torch.float32)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet.train()

    dataset = WindowDataset(tokenizer=tokenizer)  # augment=False: deterministic center crop, see scripts/dataset.py
    train_idx, val_idx = split_train_val(len(dataset), args.val_fraction)
    print(f"{len(dataset)} windows -> {len(train_idx)} train / {len(val_idx)} val", flush=True)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    output_dir = Path(args.output_dir)
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            loss = compute_loss(unet, vae, text_encoder, noise_scheduler, batch, device, use_amp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s elapsed)", flush=True)

            if step % args.save_every == 0:
                path = save_checkpoint(unet, output_dir, f"step{step:06d}")
                print(f"  saved checkpoint -> {path}", flush=True)

        val_loss = evaluate(unet, vae, text_encoder, noise_scheduler, val_loader, device, use_amp)
        print(f"epoch {epoch} done, val_loss {val_loss:.4f}", flush=True)
        final_path = save_checkpoint(unet, output_dir, f"epoch{epoch + 1:02d}")
        print(f"  saved checkpoint -> {final_path}", flush=True)

    print(f"TRAIN_DONE final checkpoint -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
