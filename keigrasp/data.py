"""Expert demonstrations: collection (DART noise, look-ahead chunk labels) and loading.

One .npz per episode. Only every `label_every`-th step is stored, because a chunk label costs
a 30-step look-ahead rollout. Stored per sample: both camera images, joint angles, and the
expert's next K joint-target actions from that state (noise-free).
"""

from pathlib import Path

import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.expert import Expert
from keiarm.rollout import (
    CHUNK,
    DART_STREAM,
    SUCCESS_HOLD_STEPS,
    dart_noise,
    expert_chunk,
    planned_teleport,
)
from keiarm.scene import episode_rng
from keiarm.task import is_success

OFFSET_STREAM = 3
DART_SIGMAS = (0.0, 0.005, 0.01, 0.015)  # cycled over scene seeds


def require_train_seed(scene_seed: int) -> None:
    if config.split_of(scene_seed) != "train":
        raise ValueError(f"scene_seed {scene_seed} is not in the training split; refusing to use it as data")


def collect_episode(scene_seed: int, episode_seed: int, sigma: float = 0.0, label_every: int = 3,
                    tail_steps: int = 20, max_steps: int = 300, disturb: int | str | None = None,
                    policy=None) -> dict:
    """Roll out and label a subset of the visited states with the expert's next K actions.

    policy=None: the expert drives, with DART noise `sigma` added to the executed actions
    (behaviour cloning data). Otherwise `policy` (a keiarm Policy) drives and the expert only
    labels (DAgger). `disturb` moves the cube once mid-episode (see rollout.planned_teleport).
    """
    require_train_seed(scene_seed)
    env = KeiArmEnv(scene_seed)
    env.reset(episode_seed)
    expert = Expert(env.model, env.scene)
    if policy is not None:
        policy.reset(env)
    teleport = planned_teleport(env, disturb)
    noise = dart_noise(episode_rng(scene_seed, episode_seed, DART_STREAM), max_steps, sigma)
    offset = int(episode_rng(scene_seed, episode_seed, OFFSET_STREAM).integers(label_every))
    images, joints, chunks, steps = [], [], [], []
    held, success_at, remaining = 0, -1, None
    for t in range(max_steps):
        if teleport is not None and t == teleport.step:
            teleport.apply(env)
        label = None
        if (t + offset) % label_every == 0:
            obs = env.observe()
            chunk = expert_chunk(env, expert)
            images.append(np.stack([obs.images[c] for c in config.CAMERAS]))
            joints.append(obs.joints)
            chunks.append(chunk)
            steps.append(t)
            label = chunk[0]
        if policy is not None:
            action = policy.act(env)
        else:
            action = (label if label is not None else expert.act(env.data)[0]) + noise[t]
        env.step(np.clip(action, *env.ctrl_range.T))
        held = held + 1 if is_success(env.privileged(), env.scene) else 0
        if remaining is None and held >= SUCCESS_HOLD_STEPS:
            success_at, remaining = t + 1, tail_steps  # keep recording the retreat for a while
        if remaining is not None:
            remaining -= 1
            if remaining <= 0:
                break
    env.close()
    return {
        "images": np.array(images, dtype=np.uint8),  # (N, 2, H, W, 3)
        "joints": np.array(joints, dtype=np.float32),  # (N, 6)
        "chunks": np.array(chunks, dtype=np.float32),  # (N, K, 6)
        "steps": np.array(steps, dtype=np.int32),
        "scene_seed": scene_seed, "episode_seed": episode_seed, "sigma": sigma,
        "success_at": success_at, "source": "expert" if policy is None else "dagger",
        "teleport_step": -1 if teleport is None else teleport.step,
    }


def save_episode(directory: Path, ep: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"s{int(ep['scene_seed']):06d}_e{int(ep['episode_seed'])}_{ep['source']}.npz"
    np.savez_compressed(path, **ep)
    return path


IMAGE_CACHE_BYTES = 8e9  # larger image sets are unpacked to a memory-mapped file instead of RAM


def _image_cache_path(files: list[tuple[Path, int]]) -> Path:
    import hashlib

    from keiarm.paths import REPO_ROOT

    key = hashlib.sha1("\n".join(f"{f.resolve()}:{f.stat().st_size}:{f.stat().st_mtime_ns}"
                                  for f, _ in files).encode()).hexdigest()[:16]
    return REPO_ROOT / "runs" / "cache" / f"images_{key}.npy"


def load_dataset(directories: list[Path], chunk: int = CHUNK,
                 cache_above: float = IMAGE_CACHE_BYTES) -> dict[str, np.ndarray]:
    """Load every episode under the given directories into memory (and re-check the split)."""
    files = sorted(f for d in directories for f in Path(d).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no episodes found in {directories}")
    # Two passes: count samples first, then fill preallocated arrays (concatenating a list
    # would hold every image twice; OpenArm datasets are ~10 GB).
    keep, total, shapes = [], 0, None
    for f in files:
        with np.load(f) as z:
            require_train_seed(int(z["scene_seed"]))
            n = len(z["steps"])
            if n == 0:
                continue
            c = z["chunks"]
            if c.shape[1] != chunk:
                continue
            if shapes is None:
                shapes = {"images": (z["images"].shape[1:], np.uint8), "joints": (z["joints"].shape[1:], np.float32),
                          "chunks": (c.shape[1:], np.float32)}
            keep.append((f, n))
            total += n
    if shapes is None:
        raise FileNotFoundError(f"no usable episodes in {directories}")
    out = {k: np.empty((total, *shape), dtype=dt) for k, (shape, dt) in shapes.items() if k != "images"}
    out["scene"] = np.empty(total, dtype=np.int64)
    image_shape = (total, *shapes["images"][0])
    cached = None
    if total * np.prod(shapes["images"][0]) > cache_above:
        # Unpack the images once to runs/cache/ (keyed by the episode files) and memory-map
        # them: the OS page cache keeps what fits, the rest is read from disk per batch.
        cached = _image_cache_path(keep)
        if cached.exists():
            out["images"] = np.load(cached, mmap_mode="r")
        else:
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp.npy")
            out["images"] = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=image_shape)
    else:
        out["images"] = np.empty(image_shape, dtype=np.uint8)
    fill_images = cached is None or not cached.exists()
    i = 0
    for f, n in keep:
        with np.load(f) as z:
            for k in ("joints", "chunks", *(("images",) if fill_images else ())):
                out[k][i : i + n] = z[k]
            out["scene"][i : i + n] = int(z["scene_seed"])
        i += n
    if cached is not None and fill_images:
        out["images"].flush()
        del out["images"]
        tmp.rename(cached)
        out["images"] = np.load(cached, mmap_mode="r")
    return out
