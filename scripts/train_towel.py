"""Train KeiGrasp on towel data by behaviour cloning (milestone 3; same trainer as scripts/train.py).

Usage: uv run python scripts/train_towel.py --name towel_bc1 --data towel_bc1 [--epochs 30]
Reads runs/data/<data>/, writes runs/train/<name>/{model.pt,log.jsonl}.
"""

import argparse

from keiarm.paths import REPO_ROOT
from keiarm.towel.env import TowelEnv
from keigrasp.model import load_checkpoint
from keigrasp.policy import resolve_checkpoint
from keigrasp.towel import GRIPPER_COLUMNS
from keigrasp.train import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="dataset names under runs/data/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", help="fine-tune from this run (keeps its normalisation constants)")
    args = ap.parse_args()
    init = load_checkpoint(resolve_checkpoint(args.init))[0] if args.init else None
    env = TowelEnv(0, render=False)
    joint_range = env.ctrl_range.copy()
    env.close()
    path = train([REPO_ROOT / "runs" / "data" / d for d in args.data], REPO_ROOT / "runs" / "train" / args.name,
                 joint_range, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, init=init,
                 absolute=GRIPPER_COLUMNS)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
