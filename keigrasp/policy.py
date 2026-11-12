"""Closed-loop execution of a trained KeiGrasp model inside the evaluation harness."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.paths import REPO_ROOT
from keigrasp.model import load_checkpoint


@lru_cache(maxsize=2)
def _load(path: str, device: str):
    return load_checkpoint(path, device)[0]


def resolve_checkpoint(name_or_path: str) -> Path:
    p = Path(name_or_path)
    return p if p.suffix == ".pt" else REPO_ROOT / "runs" / "train" / name_or_path / "model.pt"


class KeiGraspPolicy:
    """Predict a K-step chunk, execute its first `execute` steps, then look again.

    Only uses env.observe(): two camera images and the joint angles.
    """

    needs_images = True

    def __init__(self, checkpoint: str, execute: int = 10, device: str | None = None,
                 cameras: tuple[str, ...] = config.CAMERAS, blind: tuple[str, ...] = ()):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load(str(resolve_checkpoint(checkpoint)), self.device)
        self.execute = execute
        self.cameras = tuple(cameras)  # image order the model was trained with
        self.blind = set(blind)  # cameras replaced by black images (control experiments)
        if self.model.cfg.cameras != len(self.cameras):
            raise ValueError(f"model expects {self.model.cfg.cameras} cameras, got {self.cameras}")

    def reset(self, env: KeiArmEnv) -> None:
        self.plan = np.zeros((0, self.model.cfg.action_dim))
        self.cursor = 0

    @torch.no_grad()
    def infer(self, env: KeiArmEnv) -> np.ndarray:
        obs = env.observe()
        images = np.stack([obs.images[c] * (c not in self.blind) for c in self.cameras]).astype(np.uint8)
        x = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float()[None] / 255.0
        q = torch.from_numpy(obs.joints).float().to(self.device)[None]
        return self.model(x, q)[0].double().cpu().numpy()

    def act(self, env: KeiArmEnv) -> np.ndarray:
        if self.cursor >= min(self.execute, len(self.plan)):
            self.plan = self.infer(env)
            self.cursor = 0
        action = self.plan[self.cursor]
        self.cursor += 1
        return action
