import functools
import os
import subprocess
import sys

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

_PROBE = """
import mujoco
model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
mujoco.Renderer(model, 8, 8).render()
"""


@functools.cache
def gl_available() -> bool:
    """Can this machine create an off-screen GL context?

    Probed in a subprocess: on a machine without one, MuJoCo's renderer can abort the whole
    interpreter instead of raising, which would take the test run down with it.
    """
    try:
        return subprocess.run([sys.executable, "-c", _PROBE], check=False, capture_output=True,
                              timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def gl():
    """Request this fixture in tests that render."""
    if not gl_available():
        pytest.skip(f"no off-screen GL context available (MUJOCO_GL={os.environ['MUJOCO_GL']})")
