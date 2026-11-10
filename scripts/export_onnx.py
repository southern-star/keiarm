"""Export a trained KeiGrasp checkpoint to ONNX for onnxruntime-web and check it against PyTorch.

Usage: uv run python scripts/export_onnx.py --run r1 [--out web/public/models/keigrasp.onnx]

Graph inputs:  images  float32 [1, 2, 3, 120, 160] in [0, 1] (camera order: cam_front, wrist_cam)
               joints  float32 [1, 6] joint angles [rad]
Graph output:  actions float32 [1, 30, 6] joint position targets [rad]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from keiarm import config
from keiarm.paths import REPO_ROOT
from keigrasp.data import load_dataset
from keigrasp.model import load_checkpoint
from keigrasp.policy import resolve_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under runs/train/ or a .pt path")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "web" / "public" / "models" / "keigrasp.onnx")
    ap.add_argument("--check-data", default="bc1", help="dataset under runs/data/ to compare outputs on")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(resolve_checkpoint(args.run))
    cfg = model.cfg
    images = torch.rand(1, cfg.cameras, 3, cfg.image_height, cfg.image_width)
    joints = torch.zeros(1, 6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (images, joints), str(args.out), input_names=["images", "joints"],
                      output_names=["actions"], opset_version=18, dynamo=True, external_data=False)

    strip_debug_info(args.out)

    # Parity check on real observations.
    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    worst = 0.0
    data_dir = REPO_ROOT / "runs" / "data" / args.check_data
    files = sorted(data_dir.glob("*.npz"))[:8]
    if files:
        tmp = load_dataset_subset(files)
        for i in np.linspace(0, len(tmp["images"]) - 1, 64).astype(int):
            x = torch.from_numpy(tmp["images"][i]).permute(0, 3, 1, 2).float()[None] / 255.0
            q = torch.from_numpy(tmp["joints"][i])[None]
            with torch.no_grad():
                ref = model(x, q).numpy()
            out = session.run(None, {"images": x.numpy(), "joints": q.numpy()})[0]
            worst = max(worst, float(np.abs(out - ref).max()))
    meta = {
        "run": args.run, "parameters": int(sum(p.numel() for p in model.parameters())),
        "chunk": cfg.chunk, "cameras": list(config.CAMERAS),
        "image_height": cfg.image_height, "image_width": cfg.image_width,
        "max_abs_diff_vs_pytorch_rad": worst, "bytes": args.out.stat().st_size,
        "train_data": [Path(d).name for d in ckpt.get("data_dirs", [])],
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    if worst > 1e-3:
        raise SystemExit("ONNX output differs from PyTorch by more than 1e-3 rad")


def strip_debug_info(path: Path) -> None:
    """Remove exporter debug metadata (it embeds local source paths and stack traces)."""
    proto = onnx.load(str(path))
    graphs = [proto.graph, *proto.functions]
    for g in graphs:
        g.doc_string = ""
        for node in g.node:
            node.doc_string = ""
            del node.metadata_props[:]
    for g in (proto.graph,):
        del g.metadata_props[:]
        for v in [*g.value_info, *g.input, *g.output]:
            v.doc_string = ""
            del v.metadata_props[:]
    del proto.metadata_props[:]
    proto.doc_string = ""
    onnx.save(proto, str(path))


def load_dataset_subset(files: list[Path]) -> dict:
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for f in files:
            shutil.copy(f, d)
        return load_dataset([Path(d)])


if __name__ == "__main__":
    main()
