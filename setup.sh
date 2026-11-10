#!/usr/bin/env bash
# One-shot environment setup. Requires uv (https://docs.astral.sh/uv/).
set -euo pipefail
cd "$(dirname "$0")"
uv sync
MUJOCO_GL=egl uv run python -c "import mujoco; print('mujoco', mujoco.__version__)"
