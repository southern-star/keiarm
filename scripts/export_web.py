"""Write scene bundles and meshes for the browser demo into web/public/.

Usage: uv run python scripts/export_web.py [--scenes 100000 100001 ...] [--episodes 0 1]
"""

import argparse

from keiarm.export import export_web


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+",
                    default=[*range(100000, 100010), 0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    paths = export_web(args.scenes, args.episodes)
    total = sum(p.stat().st_size for p in paths)
    print(f"wrote {len(paths)} scene bundles ({total / 1e3:.0f} kB) to {paths[0].parent}")


if __name__ == "__main__":
    main()
