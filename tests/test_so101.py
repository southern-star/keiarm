import mujoco
import numpy as np
import pytest

from keiarm.robot import EE_SITE, JOINT_NAMES, WRIST_CAMERA, so101_spec

PINNED_MUJOCO = "3.17.0"  # must equal the @mujoco/mujoco version used in web/ (see CLAUDE.md)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return so101_spec().compile()


def test_mujoco_version_is_pinned():
    assert mujoco.__version__ == PINNED_MUJOCO


def test_joint_and_actuator_order(model):
    assert [model.joint(i).name for i in range(model.njnt)] == list(JOINT_NAMES)
    assert [model.actuator(i).name for i in range(model.nu)] == list(JOINT_NAMES)
    # Position actuators whose ctrl range equals the joint range (to 1e-4 rad).
    np.testing.assert_allclose(model.actuator_ctrlrange, model.jnt_range, atol=1e-4)


def test_expected_site_and_camera(model):
    assert model.site(EE_SITE).id >= 0
    assert model.camera(WRIST_CAMERA).id >= 0


def test_stepping_is_deterministic(model):
    finals = []
    for _ in range(2):
        data = mujoco.MjData(model)
        rng = np.random.default_rng(0)
        lo, hi = model.actuator_ctrlrange.T
        for _ in range(60):
            data.ctrl[:] = rng.uniform(lo, hi)
            mujoco.mj_step(model, data, nstep=7)
        finals.append(data.qpos.copy())
    np.testing.assert_array_equal(finals[0], finals[1])
