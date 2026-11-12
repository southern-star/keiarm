"""OpenArm v2 constants and model loading.

The vendored MJCF (third_party/openarm_v2) is used unmodified; everything we change
(timestep, gravity compensation, mounting height) is applied to the spec here.
"""

import mujoco

from keiarm import config
from keiarm.paths import OPENARM_XML

SIDES = ("left", "right")
ARM_DOF = 7

# Action order = actuator order in the MJCF: left arm, left gripper, right arm, right gripper.
ACTUATOR_NAMES = tuple(
    name
    for side in SIDES
    for name in [*(f"{side}_joint{i}_ctrl" for i in range(1, ARM_DOF + 1)), f"{side}_finger1_ctrl"]
)
ACTION_DIM = len(ACTUATOR_NAMES)  # 16


def arm_joints(side: str) -> tuple[str, ...]:
    return tuple(f"openarm_{side}_joint{i}" for i in range(1, ARM_DOF + 1))


def finger_joints(side: str) -> tuple[str, str]:
    return (f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2")


def ee_site(side: str) -> str:
    """Wrist frame (not the fingertips: those are ~0.15-0.17 m along the site's -z)."""
    return f"{side}_ee_control_point"


def wrist_camera(side: str) -> str:
    return f"camera_wrist_{side}"


# Finger joint targets. The two arms are mirrored: the left fingers open towards +q,
# the right ones towards -q. 0 = jaws touching; the ctrl range lets the closed command
# push past contact (0.4 rad) so the actuator squeezes.
GRIPPER_OPEN = {"left": 0.785, "right": -0.785}
GRIPPER_CLOSED = {"left": -0.4, "right": 0.4}

# Pinch point between the finger pads, in the ee_site frame (fingertips end at z = -0.17).
PINCH_OFFSET = (0.0, 0.0, -0.150)

# Arms mount height above the table top. Enactic's pedestal scenes put the arms at
# z = 0.697 over a table top at 0.400; we keep that difference with the table top at z = 0.
MOUNT_HEIGHT = 0.297
FLOOR_Z = -0.400

# Start pose: pinch point at (0.25, +-0.20, 0.20), fingers pointing straight down, jaws
# closing along world x. Found by random-restart IK, keeping the largest margin to the
# joint limits (>= 18% of each range). The arms are mirror images: the left arm is the
# right one with joints 1, 2, 3, 5, 6, 7 negated. (Enactic's own "home" holds the
# forearms horizontal 7.7 cm above the table, where they hit objects on it.)
HOME_ARM_QPOS = {
    "right": (-0.465, 2.519, 0.266, 1.866, 1.007, 0.383, 1.008),
    "left": (0.465, -2.519, -0.266, 1.866, -1.007, -0.383, -1.008),
}
HOME_PINCH = {"left": (0.25, 0.20, 0.20), "right": (0.25, -0.20, 0.20)}
# Jaw yaw at the start pose (yaw of the jaw closing direction, the ee site's y axis:
# (-sin, cos)(yaw)). Both close along world x, but the mirrored left hand's site y points
# along +x and the right's along -x. A hand holds the bar "naturally" with
# yaw = HOME_YAW + phi (phi = the bar's angle from the y axis); the 180-deg flipped grasp
# cannot turn the bar to lie along y without passing a wrist limit.
HOME_YAW = {"left": -1.5707963, "right": 1.5707963}


def openarm_spec() -> mujoco.MjSpec:
    """Bare bimanual robot, mounted at MOUNT_HEIGHT, with our physics settings.

    - timestep 1/240 (control 30 Hz x 8); the vendored 1 ms gives the same grasp
      results (docs/openarm_spike.md) at 4x the cost.
    - implicitfast + elliptic cone + impratio 10: Enactic's settings, restated because
      <option> must be set by the top-level model.
    - gravity compensation on every arm body. Without it the low-gain position
      actuators sag ~0.02 rad and the hand drops ~14 mm. It must be in the model
      before compiling: writing body_gravcomp afterwards has no effect in WASM.
    """
    spec = mujoco.MjSpec.from_file(str(OPENARM_XML))
    spec.option.timestep = config.TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    for body in spec.bodies:
        if body.name.startswith("openarm_"):
            body.gravcomp = 1.0
    for side in SIDES:
        base = spec.body(f"openarm_{side}_base_link")
        base.pos = [base.pos[0], base.pos[1], base.pos[2] + MOUNT_HEIGHT]
    origin = spec.site("arm_origin")
    origin.pos = [origin.pos[0], origin.pos[1], origin.pos[2] + MOUNT_HEIGHT]
    return spec
