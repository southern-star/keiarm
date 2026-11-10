"""Project-wide constants. Values here are mirrored in CLAUDE.md."""

CONTROL_HZ = 30
# The Menagerie model ships timestep=0.005, which does not divide 1/30 s. We use the
# nearest smaller step that does: 8 physics steps of 1/240 s per control step.
SUBSTEPS = 8
TIMESTEP = 1.0 / (CONTROL_HZ * SUBSTEPS)

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
CAMERAS = ("cam_front", "wrist_cam")  # order of the policy's image inputs

TRAIN_SCENE_SEEDS = range(4000)
EVAL_SCENE_SEEDS = range(100000, 100100)

EPISODE_SECONDS = 20
MAX_EPISODE_STEPS = EPISODE_SECONDS * CONTROL_HZ


def split_of(scene_seed: int) -> str | None:
    """Return 'train', 'eval', or None for a scene seed that belongs to neither split."""
    if scene_seed in TRAIN_SCENE_SEEDS:
        return "train"
    if scene_seed in EVAL_SCENE_SEEDS:
        return "eval"
    return None
