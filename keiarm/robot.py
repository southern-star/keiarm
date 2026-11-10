"""SO-101 constants and model loading."""

import mujoco

from keiarm.paths import SO101_XML

# Action / joint order. Identical in the Menagerie MJCF (joint and actuator order)
# and in LeRobot's SO follower (motor ids 1..6). See docs/so101_model.md.
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINT_NAMES = JOINT_NAMES[:5]
EE_SITE = "gripperframe"
WRIST_CAMERA = "wrist_cam"


def so101_spec() -> mujoco.MjSpec:
    """Return an editable spec of the bare robot (no floor, no lights)."""
    return mujoco.MjSpec.from_file(str(SO101_XML))
