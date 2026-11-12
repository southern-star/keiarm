"""Training loop for KeiGrasp (L1 regression on normalised action chunks)."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from keigrasp.data import load_dataset
from keigrasp.model import KeiGrasp, ModelConfig, save_checkpoint


def augment(images: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    """Per-sample, per-camera random translation (pixels), brightness and contrast. images in [0, 1]."""
    b, c, _, h, w = images.shape
    x = images.flatten(0, 1)
    n = x.shape[0]
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = (torch.randint(-max_shift, max_shift + 1, (n,), device=x.device)) * (2.0 / w)
    theta[:, 1, 2] = (torch.randint(-max_shift, max_shift + 1, (n,), device=x.device)) * (2.0 / h)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="nearest", padding_mode="border", align_corners=False)
    contrast = 1.0 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.3
    brightness = (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.12
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean + brightness).clamp(0.0, 1.0)
    return x.view(b, c, 3, h, w)


def to_batch(data: dict, idx: np.ndarray, device: str):
    images = torch.from_numpy(data["images"][idx]).to(device, non_blocking=True)
    images = images.permute(0, 1, 4, 2, 3).float() / 255.0
    joints = torch.from_numpy(data["joints"][idx]).to(device)
    chunks = torch.from_numpy(data["chunks"][idx]).to(device)
    return images, joints, chunks


def train(data_dirs: list[Path], out: Path, joint_range: np.ndarray, epochs: int = 30, batch_size: int = 128,
          lr: float = 3e-4, weight_decay: float = 1e-4, val_fraction: float = 0.05, seed: int = 0,
          init: KeiGrasp | None = None, device: str = "cuda", log=print,
          absolute: tuple[int, ...] = (5,)) -> Path:
    """`absolute`: action columns predicted as absolute targets (the grippers)."""
    out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(data_dirs)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Hold out whole scenes for validation.
    scenes = np.unique(data["scene"])
    val_scenes = rng.choice(scenes, max(1, int(len(scenes) * val_fraction)), replace=False)
    is_val = np.isin(data["scene"], val_scenes)
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    log(f"data: {len(train_idx)} train / {len(val_idx)} val samples from {len(scenes)} scenes, "
        f"{data['images'].nbytes / 1e9:.1f} GB of images "
        f"({'memory-mapped' if isinstance(data['images'], np.memmap) else 'in RAM'})")

    model = init or KeiGrasp(ModelConfig(chunk=data["chunks"].shape[1], cameras=data["images"].shape[1],
                                         image_height=data["images"].shape[2], image_width=data["images"].shape[3],
                                         action_dim=data["chunks"].shape[2], absolute=tuple(absolute)))
    model.to(device)
    if init is None:
        model.set_normalization(torch.from_numpy(joint_range).float().to(device),
                                torch.from_numpy(data["joints"][train_idx]).to(device),
                                torch.from_numpy(data["chunks"][train_idx]).to(device))
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: {n_params / 1e6:.2f} M parameters")

    steps_per_epoch = len(train_idx) // batch_size
    total = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    history = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        running = 0.0
        for i in range(steps_per_epoch):
            idx = np.sort(perm[i * batch_size : (i + 1) * batch_size])
            images, joints, chunks = to_batch(data, idx, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = model.predict_normalized(augment(images), joints)
            loss = F.l1_loss(pred.float(), model.normalize_target(joints, chunks))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        val = evaluate(model, data, val_idx, batch_size, device)
        row = {"epoch": epoch + 1, "train_l1_norm": running / steps_per_epoch, **val,
               "minutes": (time.perf_counter() - t0) / 60}
        history.append(row)
        log(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
        with (out / "log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    path = out / "model.pt"
    save_checkpoint(path, model, history=history, data_dirs=[str(d) for d in data_dirs],
                    train_samples=len(train_idx), parameters=n_params)
    return path


@torch.no_grad()
def evaluate(model: KeiGrasp, data: dict, idx: np.ndarray, batch_size: int, device: str) -> dict:
    """Validation error: normalised L1 and, in radians, arm joints and grippers separately."""
    model.eval()
    is_arm = model.relative.bool()
    norm, arm, grip, n = 0.0, 0.0, 0.0, 0
    for i in range(0, len(idx), batch_size):
        images, joints, chunks = to_batch(data, idx[i : i + batch_size], device)
        pred = model.predict_normalized(images, joints).float()
        norm += F.l1_loss(pred, model.normalize_target(joints, chunks), reduction="sum").item() / chunks[0].numel()
        rad = (pred * model.target_std + model.target_mean + model.relative * joints[:, None, :] - chunks).abs()
        arm += rad[..., is_arm].mean(dim=(1, 2)).sum().item()
        grip += rad[..., ~is_arm].mean(dim=(1, 2)).sum().item()
        n += len(images)
    return {"val_l1_norm": norm / n, "val_arm_rad": arm / n, "val_gripper_rad": grip / n}
