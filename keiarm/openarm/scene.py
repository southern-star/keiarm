"""OpenArm scenes: everything is a pure function of (scene_seed, episode_seed).

Task (docs/milestone2.md): a bar lies on the right-hand side of the table. The right hand
grasps it near its -y end, carries it to the middle, the left hand grasps the +y end, the
right hand lets go, and the left hand puts the bar on the plate on the left-hand side.

scene_seed   -> SceneParams   (appearance, bar and plate geometry, physics, camera jitter)
episode_seed -> EpisodeParams (bar and plate placement, arm start pose)

Like keiarm/scene.py: plain dataclasses (JSON-able for the browser), no global RNG. The
plate is a mocap body and the bar has a free joint, so placement lives in MjData and the
model depends on scene_seed only.
"""

import colorsys
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from keiarm.openarm.robot import (
    ACTUATOR_NAMES,
    FLOOR_Z,
    GRIPPER_OPEN,
    HOME_ARM_QPOS,
    SIDES,
    arm_joints,
    finger_joints,
    openarm_spec,
    wrist_camera,
)
from keiarm.paths import OPENARM_DIR

# --- Workspace (docs/openarm/reach_bar.png) ---------------------------------------------
# Table top at z = 0, footprint as Enactic's pedestal scenes (0.82 x 1.10 m, starting
# 6 cm in front of the arm mount).
TABLE_CENTER = (0.47, 0.0)
TABLE_HALF_SIZE = (0.41, 0.55, 0.02)

# Each hand holds the bar BAR_GRIP_OFFSET from its centre along the long axis, so the two
# pinch points are 12 cm apart during the hand-over (wrist housings 4 cm apart; at 9 cm they
# touch, and DART noise closes a 2 cm gap).
BAR_GRIP_OFFSET = 0.06
# The bar's long axis is within [-20, 45] deg of the world y axis (left-right), so the right
# hand always takes the -y end and the left hand the +y end. Below -20 deg the right hand's
# natural grasp (the one that can turn the bar to lie along y) runs into its wrist limits
# near the inner edge of the sector (Phase 2).
BAR_YAW_RANGE = (np.radians(-20.0), np.radians(45.0))

# Spawn sector of the right hand's grasp point (and, mirrored, of the left hand's place
# point): an annular sector around SECTOR_CENTER, cut at |y| >= SECTOR_MIN_ABS_Y. It is the
# largest such sector inside the right arm's "grasp and carry" reach for every bar yaw in
# BAR_YAW_RANGE with the natural grasp, shrunk by one 2.5 cm grid cell
# (tools/openarm_reach.py --object bar).
SECTOR_CENTER = (0.0, 0.152)  # y is mirrored per side
SECTOR_RADIUS = (0.21, 0.27)
SECTOR_AZIMUTH = (np.radians(-65.0), np.radians(-10.0))  # right side; from +x, positive towards +y
SECTOR_MIN_ABS_Y = 0.16
BAR_MIN_X = 0.10  # every bar corner stays 4 cm onto the table (clear of the pedestal)

PLATE_HEIGHT = 0.004

ARM_START_NOISE = 0.05  # [rad], uniform, per arm joint
GRIPPER_START_RANGE = (0.6, 1.0)  # fraction of fully open

CAM_FRONT_POS = (0.78, 0.0, 0.58)
CAM_FRONT_LOOKAT = (0.20, 0.0, 0.08)
CAM_FRONT_FOVY = 58.0

FLAT = {"specular": 0.0, "shininess": 0.0, "reflectance": 0.0}

_SCENE_TAG = 0x0A2C5CE
_EPISODE_TAG = 0x0A2E915


@dataclass(frozen=True)
class SceneParams:
    scene_seed: int
    # Appearance
    table_rgb: list[float]
    bar_rgb: list[float]
    plate_rgb: list[float]
    background_rgb: list[float]
    floor_rgb: list[float]
    light_dir: list[float]
    light_diffuse: float
    ambient: float
    # Geometry
    bar_size: list[float]  # full extents [m]: across the jaws (x), long axis (y), height (z)
    plate_radius: float
    cam_front_pos: list[float]
    cam_front_quat: list[float]  # wxyz, MuJoCo camera convention
    cam_front_fovy: float
    wrist_cam_pos: dict[str, list[float]]  # in each ee_base_link frame
    wrist_cam_quat: dict[str, list[float]]
    # Physics
    bar_mass: float
    bar_friction: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeParams:
    scene_seed: int
    episode_seed: int
    bar_xy: list[float]
    bar_yaw: float  # long axis = (-sin, cos)(yaw); 0 = along world y
    plate_xy: list[float]
    start_ctrl: list[float]  # 16 actuator targets in ACTUATOR_NAMES order (= start qpos)

    def to_dict(self) -> dict:
        return asdict(self)


def scene_rng(scene_seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([_SCENE_TAG, scene_seed]))


def episode_rng(scene_seed: int, episode_seed: int, stream: int = 0) -> np.random.Generator:
    """RNG for one episode. stream 0 = placement; other streams are independent."""
    return np.random.default_rng(np.random.SeedSequence([_EPISODE_TAG, scene_seed, episode_seed, stream]))


def _rounded(x, nd: int = 6) -> list[float]:
    return [round(float(v), nd) for v in np.asarray(x).ravel()]


def look_at_quat(pos, target) -> np.ndarray:
    """wxyz quaternion of a MuJoCo camera at `pos` looking at `target` (+z world up)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    back = pos - target
    back /= np.linalg.norm(back)  # camera +z points away from the target
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([right, up, back]).ravel())
    return quat


def _jitter_quat(quat, rotvec: np.ndarray) -> np.ndarray:
    dq, out = np.empty(4), np.empty(4)
    angle = np.linalg.norm(rotvec)
    mujoco.mju_axisAngle2Quat(dq, rotvec / max(angle, 1e-12), angle)
    mujoco.mju_mulQuat(out, np.asarray(quat, dtype=float), dq)
    return out / np.linalg.norm(out)


def bar_axis(yaw: float) -> np.ndarray:
    return np.array([-np.sin(yaw), np.cos(yaw)])


def grip_points(bar_xy, bar_yaw: float) -> dict[str, np.ndarray]:
    """Where each hand holds the bar (xy): right at the -y end, left at the +y end."""
    c, a = np.asarray(bar_xy, float), bar_axis(bar_yaw)
    return {"right": c - BAR_GRIP_OFFSET * a, "left": c + BAR_GRIP_OFFSET * a}


def in_sector(xy, side: str) -> bool:
    """Is xy inside `side`'s spawn sector (grasp point for right, place point for left)?"""
    x, y = float(xy[0]), float(xy[1]) * (1.0 if side == "right" else -1.0)  # mirror left onto right
    dx, dy = x - SECTOR_CENTER[0], y + SECTOR_CENTER[1]
    r, az = np.hypot(dx, dy), np.arctan2(dy, dx)
    return bool(SECTOR_RADIUS[0] <= r <= SECTOR_RADIUS[1] and SECTOR_AZIMUTH[0] <= az <= SECTOR_AZIMUTH[1]
                and y <= -SECTOR_MIN_ABS_Y)


def _sample_sector(rng: np.random.Generator, side: str) -> np.ndarray:
    """Uniform-by-area point in the side's sector (rejection on the |y| cut)."""
    mirror = 1.0 if side == "right" else -1.0
    while True:
        r = np.sqrt(rng.uniform(SECTOR_RADIUS[0] ** 2, SECTOR_RADIUS[1] ** 2))
        az = rng.uniform(*SECTOR_AZIMUTH)
        y = -SECTOR_CENTER[1] + r * np.sin(az)
        if y <= -SECTOR_MIN_ABS_Y:
            return np.array([SECTOR_CENTER[0] + r * np.cos(az), mirror * y])


def bar_corners(bar_xy, bar_yaw: float, bar_size) -> np.ndarray:
    a = bar_axis(bar_yaw)
    n = np.array([a[1], -a[0]])
    hx, hy = bar_size[0] / 2, bar_size[1] / 2
    return np.array([np.asarray(bar_xy) + sx * hx * n + sy * hy * a for sx in (-1, 1) for sy in (-1, 1)])


def _bar_fits(bar_xy, bar_yaw: float, bar_size) -> bool:
    corners = bar_corners(bar_xy, bar_yaw, bar_size)
    x1, y1 = TABLE_CENTER[0] + TABLE_HALF_SIZE[0], TABLE_HALF_SIZE[1]
    return bool((corners[:, 0] >= BAR_MIN_X).all() and (corners[:, 0] <= x1 - 0.02).all()
                and (np.abs(corners[:, 1]) <= y1 - 0.02).all())


def _wrist_cam_nominal() -> tuple[dict, dict]:
    """Wrist camera poses in their body frames (the MJCF gives euler angles; read the
    compiled quaternion)."""
    model = openarm_spec().compile()
    ids = {side: model.camera(wrist_camera(side)).id for side in SIDES}
    return ({s: model.cam_pos[i].copy() for s, i in ids.items()},
            {s: model.cam_quat[i].copy() for s, i in ids.items()})


_WRIST_NOMINAL = None


def sample_scene(scene_seed: int) -> SceneParams:
    global _WRIST_NOMINAL
    if _WRIST_NOMINAL is None:
        _WRIST_NOMINAL = _wrist_cam_nominal()
    rng = scene_rng(scene_seed)

    # Colours: muted table, floor and background; saturated bar and plate with distinct hues.
    table_rgb = colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(0.0, 0.25), rng.uniform(0.40, 0.85))
    background_rgb = colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(0.0, 0.2), rng.uniform(0.15, 0.9))
    floor_rgb = colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(0.0, 0.15), rng.uniform(0.15, 0.5))
    bar_hue = rng.uniform()
    plate_hue = (bar_hue + rng.uniform(0.25, 0.75)) % 1.0
    bar_rgb = colorsys.hsv_to_rgb(bar_hue, rng.uniform(0.7, 1.0), rng.uniform(0.7, 1.0))
    plate_rgb = colorsys.hsv_to_rgb(plate_hue, rng.uniform(0.5, 1.0), rng.uniform(0.6, 1.0))

    tilt = rng.uniform(0.0, np.radians(35.0))
    az = rng.uniform(-np.pi, np.pi)
    light_dir = [np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)]
    light_diffuse = rng.uniform(0.45, 0.8)
    ambient = rng.uniform(0.3, 0.5)

    bar_size = [rng.uniform(0.025, 0.035), rng.uniform(0.15, 0.18), rng.uniform(0.025, 0.04)]
    plate_radius = rng.uniform(0.07, 0.09)

    cam_pos = np.array(CAM_FRONT_POS) + rng.uniform(-0.015, 0.015, 3)
    cam_target = np.array(CAM_FRONT_LOOKAT) + rng.uniform(-0.010, 0.010, 3)
    cam_fovy = CAM_FRONT_FOVY + rng.uniform(-1.0, 1.0)
    wpos, wquat = _WRIST_NOMINAL
    wrist_pos = {s: _rounded(wpos[s] + rng.uniform(-0.002, 0.002, 3)) for s in SIDES}
    wrist_quat = {s: _rounded(_jitter_quat(wquat[s], rng.uniform(-1.0, 1.0, 3) * np.radians(1.0))) for s in SIDES}

    bar_mass = rng.uniform(0.05, 0.12)
    bar_friction = rng.uniform(0.8, 1.2)

    return SceneParams(
        scene_seed=int(scene_seed),
        table_rgb=_rounded(table_rgb, 4), bar_rgb=_rounded(bar_rgb, 4), plate_rgb=_rounded(plate_rgb, 4),
        background_rgb=_rounded(background_rgb, 4), floor_rgb=_rounded(floor_rgb, 4),
        light_dir=_rounded(light_dir), light_diffuse=round(float(light_diffuse), 4), ambient=round(float(ambient), 4),
        bar_size=_rounded(bar_size, 5), plate_radius=round(float(plate_radius), 5),
        cam_front_pos=_rounded(cam_pos), cam_front_quat=_rounded(look_at_quat(cam_pos, cam_target)),
        cam_front_fovy=round(float(cam_fovy), 3),
        wrist_cam_pos=wrist_pos, wrist_cam_quat=wrist_quat,
        bar_mass=round(float(bar_mass), 5), bar_friction=round(float(bar_friction), 4),
    )


def sample_bar_pose(rng: np.random.Generator, scene: SceneParams) -> tuple[np.ndarray, float]:
    """A bar pose whose right grasp point is in the right sector and that lies on the table."""
    while True:
        grip = _sample_sector(rng, "right")
        yaw = rng.uniform(*BAR_YAW_RANGE)
        center = grip + BAR_GRIP_OFFSET * bar_axis(yaw)
        if _bar_fits(center, yaw, scene.bar_size):
            return center, float(yaw)


def sample_plate_xy(rng: np.random.Generator, scene: SceneParams) -> np.ndarray:
    """Plate centre such that the left hand, holding the bar at +offset with the bar along
    y, places it at the centre from a point inside the left sector."""
    while True:
        place = _sample_sector(rng, "left")
        center = place - np.array([0.0, BAR_GRIP_OFFSET])
        x0, x1 = TABLE_CENTER[0] - TABLE_HALF_SIZE[0], TABLE_CENTER[0] + TABLE_HALF_SIZE[0]
        if center[0] - scene.plate_radius >= x0 + 0.02 and center[0] + scene.plate_radius <= x1 - 0.02:
            return center


def sample_episode(scene: SceneParams, episode_seed: int) -> EpisodeParams:
    rng = episode_rng(scene.scene_seed, episode_seed)
    bar_xy, bar_yaw = sample_bar_pose(rng, scene)
    plate_xy = sample_plate_xy(rng, scene)
    ctrl = []
    for side in SIDES:
        ctrl += list(np.array(HOME_ARM_QPOS[side]) + rng.uniform(-ARM_START_NOISE, ARM_START_NOISE, 7))
        ctrl.append(GRIPPER_OPEN[side] * rng.uniform(*GRIPPER_START_RANGE))
    return EpisodeParams(
        scene_seed=scene.scene_seed, episode_seed=int(episode_seed),
        bar_xy=_rounded(bar_xy), bar_yaw=round(bar_yaw, 6), plate_xy=_rounded(plate_xy),
        start_ctrl=_rounded(ctrl),
    )


def build_spec(scene: SceneParams) -> mujoco.MjSpec:
    """Robot + pedestal + table + floor + bar + plate + cameras + light.

    Flat shading as in keiarm/scene.py: no headlight diffuse, no specular, no shadows.
    """
    spec = openarm_spec()
    spec.modelname = "keiarm_openarm_scene"
    spec.visual.headlight.ambient = [scene.ambient] * 3
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    for mat in spec.materials:
        mat.specular = FLAT["specular"]
        mat.shininess = FLAT["shininess"]
        mat.reflectance = FLAT["reflectance"]
    spec.add_texture(name="background", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=scene.background_rgb,
                     rgb2=scene.background_rgb, width=8, height=48)
    for name, rgb in (("table", scene.table_rgb), ("bar", scene.bar_rgb), ("plate", scene.plate_rgb),
                      ("pedestal", (0.25, 0.25, 0.25)), ("floor", scene.floor_rgb)):
        spec.add_material(name=name, rgba=[*rgb, 1.0], **FLAT)

    world = spec.worldbody
    world.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=scene.light_dir,
                    pos=[0.3, 0.0, 1.5], diffuse=[scene.light_diffuse] * 3, specular=[0.0, 0.0, 0.0],
                    castshadow=False)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05],
                   pos=[0.0, 0.0, FLOOR_Z], material="floor")
    # Pedestal: Enactic's simplified collision mesh, also used as the visual (the original
    # visual mesh has 160k faces). Its origin is on the floor.
    spec.add_mesh(name="pedestal", file=str(OPENARM_DIR / "assets" / "collision" / "body_link0_symp.stl"),
                  scale=[0.001, 0.001, 0.001])
    world.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="pedestal",
                   pos=[0.0, 0.0, FLOOR_Z], material="pedestal")
    hx, hy, hz = TABLE_HALF_SIZE
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                   pos=[TABLE_CENTER[0], TABLE_CENTER[1], -hz], material="table")

    world.add_camera(name="cam_front", pos=scene.cam_front_pos, quat=scene.cam_front_quat,
                     fovy=scene.cam_front_fovy)
    for side in SIDES:
        cam = spec.camera(wrist_camera(side))
        cam.pos = scene.wrist_cam_pos[side]
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        cam.quat = scene.wrist_cam_quat[side]

    plate = world.add_body(name="plate", mocap=True, pos=[0.25, 0.25, PLATE_HEIGHT / 2])
    plate.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[scene.plate_radius, PLATE_HEIGHT / 2, 0], material="plate")

    bx, by, bz = (s / 2 for s in scene.bar_size)
    bar = world.add_body(name="bar", pos=[0.25, -0.25, bz])
    bar.add_freejoint(name="bar")
    bar.add_geom(name="bar", type=mujoco.mjtGeom.mjGEOM_BOX, size=[bx, by, bz], mass=scene.bar_mass,
                 friction=[scene.bar_friction, 0.01, 0.001], material="bar")
    return spec


def build_model(scene: SceneParams) -> mujoco.MjModel:
    return build_spec(scene).compile()


def set_bar_pose(model: mujoco.MjModel, data: mujoco.MjData, xy, yaw: float, z: float) -> None:
    adr = model.joint("bar").qposadr[0]
    data.qpos[adr : adr + 3] = [*xy, z]
    data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    data.qvel[model.joint("bar").dofadr[0] : model.joint("bar").dofadr[0] + 6] = 0.0


def reset_data(model: mujoco.MjModel, data: mujoco.MjData, scene: SceneParams, episode: EpisodeParams) -> None:
    """Put `data` into the episode's initial state (arms held at their start pose)."""
    mujoco.mj_resetData(model, data)
    ctrl = dict(zip(ACTUATOR_NAMES, episode.start_ctrl))
    for side in SIDES:
        for name in arm_joints(side):
            act = name.removeprefix("openarm_") + "_ctrl"
            data.qpos[model.joint(name).qposadr[0]] = ctrl[act]
            data.ctrl[model.actuator(act).id] = ctrl[act]
        g = ctrl[f"{side}_finger1_ctrl"]
        for name in finger_joints(side):
            data.qpos[model.joint(name).qposadr[0]] = g
        data.ctrl[model.actuator(f"{side}_finger1_ctrl").id] = g
    set_bar_pose(model, data, episode.bar_xy, episode.bar_yaw, scene.bar_size[2] / 2 + 1e-3)
    data.mocap_pos[model.body("plate").mocapid[0]] = [*episode.plate_xy, PLATE_HEIGHT / 2]
    mujoco.mj_forward(model, data)
