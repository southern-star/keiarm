"""KeiGrasp: a small action-chunking policy (two camera images + joint angles -> K joint targets).

Deterministic regression in the ACT family: per-camera CNN encoders produce feature tokens, a
transformer encoder mixes them with a joint-state token, and a transformer decoder turns K
learned queries into the next K actions. All input/output normalisation lives inside the
module, so the exported graph takes raw images in [0, 1] and joint angles in radians and
returns joint targets in radians.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    chunk: int = 30
    cameras: int = 2
    image_height: int = 120
    image_width: int = 160
    dim: int = 256
    heads: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 2
    widths: tuple[int, ...] = (32, 64, 128, 256)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class ImageEncoder(nn.Module):
    """Narrow ResNet: (3, 120, 160) -> (dim, 8, 10)."""

    def __init__(self, widths: tuple[int, ...], dim: int):
        super().__init__()
        layers = [nn.Conv2d(3, widths[0], 5, 2, 2, bias=False), nn.GroupNorm(8, widths[0]), nn.ReLU()]
        cin = widths[0]
        for w in widths[1:]:
            layers += [Block(cin, w, 2), Block(w, w, 1)]
            cin = w
        layers.append(nn.Conv2d(cin, dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class KeiGrasp(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d = cfg.dim
        self.encoders = nn.ModuleList(ImageEncoder(cfg.widths, d) for _ in range(cfg.cameras))
        with torch.no_grad():
            fh, fw = self.encoders[0](torch.zeros(1, 3, cfg.image_height, cfg.image_width)).shape[-2:]
        self.tokens_per_camera = fh * fw
        self.image_pos = nn.Parameter(torch.randn(cfg.cameras, self.tokens_per_camera, d) * 0.02)
        self.joint_embed = nn.Sequential(nn.Linear(6, d), nn.ReLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, cfg.heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, cfg.encoder_layers, enable_nested_tensor=False)
        dec = nn.TransformerDecoderLayer(d, cfg.heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, cfg.decoder_layers)
        self.queries = nn.Parameter(torch.randn(cfg.chunk, d) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 6))
        # Normalisation constants (set from the model's joint limits and the dataset).
        self.register_buffer("joint_mid", torch.zeros(6))
        self.register_buffer("joint_half", torch.ones(6))
        self.register_buffer("target_mean", torch.zeros(6))
        self.register_buffer("target_std", torch.ones(6))
        # 1 for joints predicted relative to the current angle (the arm), 0 for absolute (gripper).
        self.register_buffer("relative", torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))

    def set_normalization(self, joint_range: torch.Tensor, joints: torch.Tensor, chunks: torch.Tensor) -> None:
        """joint_range (6, 2); joints (N, 6) and chunks (N, K, 6) from the training data [rad]."""
        self.joint_mid.copy_(joint_range.mean(dim=1))
        self.joint_half.copy_((joint_range[:, 1] - joint_range[:, 0]) / 2)
        raw = chunks - self.relative * joints[:, None, :]
        self.target_mean.copy_(raw.mean(dim=(0, 1)))
        self.target_std.copy_(raw.std(dim=(0, 1)).clamp_min(1e-3))

    def normalize_target(self, joints: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        return (chunks - self.relative * joints[:, None, :] - self.target_mean) / self.target_std

    def predict_normalized(self, images: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """images (B, cameras, 3, H, W) in [0, 1]; joints (B, 6) [rad] -> (B, K, 6) normalised."""
        b = images.shape[0]
        tokens = []
        for i, enc in enumerate(self.encoders):
            f = enc(images[:, i] * 2.0 - 1.0)  # (B, d, h, w)
            tokens.append(f.flatten(2).transpose(1, 2) + self.image_pos[i])
        joint = self.joint_embed((joints - self.joint_mid) / self.joint_half)[:, None, :]
        memory = self.encoder(torch.cat([joint, *tokens], dim=1))
        out = self.decoder(self.queries.expand(b, -1, -1), memory)
        return self.head(out)

    def forward(self, images: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """Joint position targets [rad] for the next K control steps, (B, K, 6)."""
        pred = self.predict_normalized(images, joints)
        return pred * self.target_std + self.target_mean + self.relative * joints[:, None, :]


def save_checkpoint(path, model: KeiGrasp, **extra) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[KeiGrasp, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**{**ckpt["config"], "widths": tuple(ckpt["config"]["widths"])})
    model = KeiGrasp(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
