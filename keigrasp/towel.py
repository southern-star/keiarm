"""Towel demonstrations (milestone 3): collection with DART noise and look-ahead chunk labels.

The towel counterpart of keigrasp/openarm.py; the stored format is the same, so
keigrasp.data.load_dataset and keigrasp.train read it unchanged. Per sample: the three
camera images (env.CAMERAS order), the 16 joint angles, and the expert's next K actions
from that state (noise-free).

The look-ahead runs on the environment itself and restores it afterwards (env.get_state /
set_state), because the grasp rule writes the model's constraint slots: a copy of MjData
alone would not be independent. Restoring is bit-exact (tests/test_towel.py).
"""

import numpy as np

from keiarm import config
from keiarm.openarm.rollout import dart_noise
from keiarm.towel.env import CAMERAS, MAX_EPISODE_STEPS, TowelEnv
from keiarm.towel.eval import planned_disturbance
from keiarm.towel.expert import Expert
from keiarm.towel.rollout import DART_STREAM
from keiarm.towel.scene import episode_rng
from keiarm.towel.task import STAGES, SUCCESS_HOLD_STEPS, check_fold, read_state, stage_of
from keigrasp.data import require_train_seed
from keigrasp.openarm import CHUNK, GRIPPER_COLUMNS  # noqa: F401  (same action layout as milestone 2)

OFFSET_STREAM = 3
# Expert success with DART noise (100 eval scenes, N = 16, docs/milestone3.md Phase 4):
# 0.00325 -> 100, 0.0065 -> 91, 0.00975 -> 10. The same set as milestone 2.
DART_SIGMAS = (0.0, 0.00325, 0.0065)  # cycled over scene seeds
COLLECT_MAX_STEPS = MAX_EPISODE_STEPS  # 600; the clean expert needs ~160, the 0.0065-noisy one ~315


def expert_chunk(env: TowelEnv, expert: Expert, commanded: np.ndarray | None,
                 length: int = CHUNK) -> np.ndarray:
    """The next `length` clean expert actions from the current state, (length, 16).

    `commanded`: the previous clean label when the executed actions carry DART noise (the
    carry continues from it, Expert.act), None otherwise. Inside the look-ahead the actions
    are clean, so each step continues from the one before. The environment is restored."""
    saved = env.get_state()
    lo, hi = env.ctrl_range.T
    out = np.empty((length, env.act.size))
    prev = commanded
    try:
        for k in range(length):
            out[k] = np.clip(expert.act(env.data, prev)[0], lo, hi)
            env.step(out[k])
            prev = out[k]
    finally:
        env.set_state(saved)
    return out


def collect_episode(scene_seed: int, episode_seed: int, sigma: float = 0.0, label_every: int = 3,
                    tail_steps: int = 20, max_steps: int = COLLECT_MAX_STEPS, disturb: str = "none",
                    policy=None, n: int | None = None) -> dict:
    """Roll out and label every `label_every`-th visited state with the expert's next K actions.

    policy=None: the expert drives with DART noise `sigma` (behaviour cloning data); between
    labelled steps it continues from its previous clean action. Otherwise `policy` (a
    keiarm.towel.policies Policy) drives and the expert only labels (DAgger), continuing from
    the policy's actual command. `disturb`: one of keiarm.towel.eval.DISTURBANCES.
    """
    require_train_seed(scene_seed)
    env = TowelEnv(scene_seed) if n is None else TowelEnv(scene_seed, n=n)
    try:
        env.reset(episode_seed)
        expert = Expert(env.model, env.scene)
        ix = expert.ix
        if policy is not None:
            policy.reset(env)
        dist = planned_disturbance(env, disturb)
        noise = dart_noise(episode_rng(scene_seed, episode_seed, DART_STREAM), max_steps, sigma)
        offset = int(episode_rng(scene_seed, episode_seed, OFFSET_STREAM).integers(label_every))
        lo, hi = env.ctrl_range.T
        noisy = policy is None and sigma > 0
        images, joints, chunks, steps = [], [], [], []
        stage_step: dict[int, int] = {}
        stage, held, success_at, remaining = 0, 0, -1, None
        clean = None  # the expert's previous clean action (DART only)
        for t in range(max_steps):
            if dist is not None and dist.step in (None, t) and dist.due(t, stage_step):
                dist.step = t
                dist.apply(env)
            commanded = clean if noisy else None
            label = None
            if (t + offset) % label_every == 0:
                obs = env.observe()
                chunk = expert_chunk(env, expert, commanded)
                images.append(np.stack([obs.images[c] for c in CAMERAS]))
                joints.append(obs.joints)
                chunks.append(chunk)
                steps.append(t)
                label = chunk[0]
            if policy is not None:
                action = policy.act(env)
            else:
                if label is None:
                    label = np.clip(expert.act(env.data, commanded)[0], lo, hi)
                clean = label
                action = label + noise[t]
            env.step(np.clip(action, lo, hi))
            st = read_state(env.model, ix, env.data)
            s = stage_of(st, ix, env.scene)
            for k in range(1, s + 1):
                stage_step.setdefault(k, t + 1)
            stage = max(stage, s)
            held = held + 1 if check_fold(st, ix, env.scene).success else 0
            if remaining is None and held >= SUCCESS_HOLD_STEPS and (dist is None or dist.step is not None):
                success_at, remaining = t + 1, tail_steps  # keep recording the retreat for a while
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break
    finally:
        env.close()
    return {
        "images": np.array(images, dtype=np.uint8),  # (N, 3, H, W, 3)
        "joints": np.array(joints, dtype=np.float32),  # (N, 16)
        "chunks": np.array(chunks, dtype=np.float32),  # (N, K, 16)
        "steps": np.array(steps, dtype=np.int32),
        "scene_seed": scene_seed, "episode_seed": episode_seed, "sigma": sigma,
        "success_at": success_at, "stage": STAGES[stage], "source": "expert" if policy is None else "dagger",
        "disturb": disturb, "disturb_step": -1 if dist is None or dist.step is None else dist.step,
        "robot": "openarm_v2_towel", "mesh": int(ix.n), "image_size": [config.IMAGE_HEIGHT, config.IMAGE_WIDTH],
    }
