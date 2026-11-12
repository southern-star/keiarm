"""OpenArm demonstrations (milestone 2): collection with DART noise and look-ahead chunk labels.

The OpenArm counterpart of keigrasp/data.py; the stored format is the same, so
keigrasp.data.load_dataset and keigrasp.train read it unchanged. Per sample: the three
camera images (env.CAMERAS order), the 16 joint angles, and the expert's next K actions
from that state (noise-free).
"""

import numpy as np

from keiarm import config
from keiarm.openarm.env import CAMERAS, OpenArmEnv
from keiarm.openarm.eval import planned_disturbance
from keiarm.openarm.expert import Expert
from keiarm.openarm.robot import ACTUATOR_NAMES
from keiarm.openarm.rollout import DART_STREAM, SUCCESS_HOLD_STEPS, dart_noise
from keiarm.openarm.scene import episode_rng
from keiarm.openarm.task import STAGES, is_success, stage_of
from keigrasp.data import require_train_seed

CHUNK = 30
OFFSET_STREAM = 3
# Expert success with DART noise (100 eval scenes): 0 -> 100, 0.00325 -> 98, 0.0065 -> 90,
# 0.00975 -> 48. The last one mostly produces unrecoverable states, so it is not used.
DART_SIGMAS = (0.0, 0.00325, 0.0065)  # cycled over scene seeds
COLLECT_MAX_STEPS = 600  # the clean expert needs ~225 (Phase 3), the 0.0065-noisy one ~330
GRIPPER_COLUMNS = tuple(i for i, n in enumerate(ACTUATOR_NAMES) if "finger" in n)  # predicted absolute


def expert_chunk(env: OpenArmEnv, expert: Expert, length: int = CHUNK) -> np.ndarray:
    """The next `length` expert actions from the current state, (length, 16), rolled out
    on a copy of the simulator state (env.data is left untouched)."""
    data = env.scratch_copy()
    lo, hi = env.ctrl_range.T
    out = np.empty((length, len(ACTUATOR_NAMES)))
    for k in range(length):
        out[k] = np.clip(expert.act(data)[0], lo, hi)
        env.step_data(data, out[k])
    return out


def collect_episode(scene_seed: int, episode_seed: int, sigma: float = 0.0, label_every: int = 3,
                    tail_steps: int = 20, max_steps: int = COLLECT_MAX_STEPS, disturb: str = "none",
                    policy=None) -> dict:
    """Roll out and label every `label_every`-th visited state with the expert's next K actions.

    policy=None: the expert drives with DART noise `sigma` (behaviour cloning data). Otherwise
    `policy` (a keiarm.openarm.policies Policy) drives and the expert only labels (DAgger).
    `disturb`: one of keiarm.openarm.eval.DISTURBANCES, decided by the seeds as in evaluation.
    """
    require_train_seed(scene_seed)
    env = OpenArmEnv(scene_seed)
    env.reset(episode_seed)
    expert = Expert(env.model, env.scene)
    if policy is not None:
        policy.reset(env)
    dist = planned_disturbance(env, disturb)
    noise = dart_noise(episode_rng(scene_seed, episode_seed, DART_STREAM), max_steps, sigma)
    offset = int(episode_rng(scene_seed, episode_seed, OFFSET_STREAM).integers(label_every))
    lo, hi = env.ctrl_range.T
    images, joints, chunks, steps = [], [], [], []
    stage_step: dict[int, int] = {}
    stage, held, success_at, remaining = 0, 0, -1, None
    for t in range(max_steps):
        if dist is not None and dist.step is None and dist.due(t, stage_step):
            dist.step = t
            dist.apply(env)
        label = None
        if (t + offset) % label_every == 0:
            obs = env.observe()
            chunk = expert_chunk(env, expert)
            images.append(np.stack([obs.images[c] for c in CAMERAS]))
            joints.append(obs.joints)
            chunks.append(chunk)
            steps.append(t)
            label = chunk[0]
        if policy is not None:
            action = policy.act(env)
        else:
            action = (label if label is not None else expert.act(env.data)[0]) + noise[t]
        env.step(np.clip(action, lo, hi))
        st = env.privileged()
        s = stage_of(st, env.scene)
        for k in range(1, s + 1):
            stage_step.setdefault(k, t + 1)
        stage = max(stage, s)
        held = held + 1 if is_success(st, env.scene) else 0
        if remaining is None and held >= SUCCESS_HOLD_STEPS:
            success_at, remaining = t + 1, tail_steps  # keep recording the retreat for a while
        if remaining is not None:
            remaining -= 1
            if remaining <= 0:
                break
    env.close()
    return {
        "images": np.array(images, dtype=np.uint8),  # (N, 3, H, W, 3)
        "joints": np.array(joints, dtype=np.float32),  # (N, 16)
        "chunks": np.array(chunks, dtype=np.float32),  # (N, K, 16)
        "steps": np.array(steps, dtype=np.int32),
        "scene_seed": scene_seed, "episode_seed": episode_seed, "sigma": sigma,
        "success_at": success_at, "stage": STAGES[stage], "source": "expert" if policy is None else "dagger",
        "disturb": disturb, "disturb_step": -1 if dist is None or dist.step is None else dist.step,
        "robot": "openarm_v2", "image_size": [config.IMAGE_HEIGHT, config.IMAGE_WIDTH],
    }
