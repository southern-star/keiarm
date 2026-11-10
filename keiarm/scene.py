"""Procedural scenes: everything is a pure function of (scene_seed, episode_seed).

scene_seed   -> SceneParams   (appearance, geometry, physics; fixes the MjModel)
episode_seed -> EpisodeParams (object placement, arm start pose; fixes the initial MjData)

Parameters are plain dataclasses of floats/lists so they can be dumped to JSON for the
browser demo. No global RNG is used anywhere.
"""

import colorsys
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from keiarm import config
from keiarm.robot import JOINT_NAMES, WRIST_CAMERA, so101_spec

# --- Workspace -----------------------------------------------------------------------
# Top-down grasps are reachable (at 1.5 cm and 6 cm site height) in an annulus of radius
# 0.10-0.24 m around the shoulder pan axis; see docs/reachability.png. Objects spawn
# inside it with a margin.
PAN_AXIS_XY = (0.0388353, 0.0)
CUBE_RADIUS_RANGE = (0.13, 0.22)
PLATE_RADIUS_RANGE = (0.15, 0.22)  # of the plate centre; further out so it clears the base
AZIMUTH_LIMIT = np.radians(60.0)
PLATE_HEIGHT = 0.004
TABLE_CENTER = (0.20, 0.0)
TABLE_HALF_SIZE = (0.40, 0.50, 0.02)

# Arm start pose: gripper pointing straight down 8 cm above the table at x = 0.20 m.
HOME_QPOS = (0.0, -0.382, 0.365, 1.587, 0.0, 0.8)
HOME_ARM_NOISE = 0.08  # [rad], uniform, per arm joint
HOME_GRIPPER_RANGE = (0.5, 1.1)

CAM_FRONT_POS = (0.70, -0.28, 0.46)
CAM_FRONT_LOOKAT = (0.13, 0.0, 0.09)
CAM_FRONT_FOVY = 46.0

_SCENE_TAG = 0x5CE7E
_EPISODE_TAG = 0xE9150DE


@dataclass(frozen=True)
class SceneParams:
    scene_seed: int
    # Appearance
    table_rgb: list[float]
    cube_rgb: list[float]
    plate_rgb: list[float]
    background_rgb: list[float]
    light_dir: list[float]  # direction the light travels (unit vector, pointing down)
    light_diffuse: float
    ambient: float
    # Geometry
    cube_size: float  # edge length [m]
    plate_radius: float
    cam_front_pos: list[float]
    cam_front_quat: list[float]  # wxyz, MuJoCo camera convention (looks along -z, +y up)
    cam_front_fovy: float
    wrist_cam_pos: list[float]  # in the camera_mount body frame
    wrist_cam_quat: list[float]
    # Physics
    cube_mass: float
    cube_friction: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeParams:
    scene_seed: int
    episode_seed: int
    cube_xy: list[float]
    cube_yaw: float
    plate_xy: list[float]
    arm_qpos: list[float]  # 6 joint angles in JOINT_NAMES order

    def to_dict(self) -> dict:
        return asdict(self)


def scene_rng(scene_seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([_SCENE_TAG, scene_seed]))


def episode_rng(scene_seed: int, episode_seed: int, stream: int = 0) -> np.random.Generator:
    """RNG for one episode. stream 0 = placement; other streams (e.g. DART noise) are independent."""
    return np.random.default_rng(np.random.SeedSequence([_EPISODE_TAG, scene_seed, episode_seed, stream]))


def _rounded(x, nd: int = 6) -> list[float]:
    return [round(float(v), nd) for v in np.asarray(x).ravel()]


def _look_at_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).reshape(9))
    return quat


def _jitter_quat(quat: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    dq, out = np.empty(4), np.empty(4)
    angle = np.linalg.norm(rotvec)
    mujoco.mju_axisAngle2Quat(dq, rotvec / max(angle, 1e-12), angle)
    mujoco.mju_mulQuat(out, np.asarray(quat, dtype=float), dq)
    return out / np.linalg.norm(out)


def _wrist_cam_nominal() -> tuple[np.ndarray, np.ndarray]:
    cam = so101_spec().camera(WRIST_CAMERA)
    return np.array(cam.pos), np.array(cam.quat)


def sample_scene(scene_seed: int) -> SceneParams:
    rng = scene_rng(scene_seed)

    # Colours: muted table and background, saturated cube and plate with distinct hues.
    table_v = rng.uniform(0.35, 0.85)
    table_rgb = colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(0.0, 0.25), table_v)
    background_rgb = colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(0.0, 0.2), rng.uniform(0.15, 0.9))
    cube_hue = rng.uniform()
    plate_hue = (cube_hue + rng.uniform(0.25, 0.75)) % 1.0
    cube_rgb = colorsys.hsv_to_rgb(cube_hue, rng.uniform(0.7, 1.0), rng.uniform(0.7, 1.0))
    plate_rgb = colorsys.hsv_to_rgb(plate_hue, rng.uniform(0.5, 1.0), rng.uniform(0.6, 1.0))

    tilt = rng.uniform(0.0, np.radians(35.0))
    az = rng.uniform(-np.pi, np.pi)
    light_dir = [np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)]
    light_diffuse = rng.uniform(0.45, 0.8)
    ambient = rng.uniform(0.3, 0.5)

    cube_size = rng.uniform(0.025, 0.035)
    plate_radius = rng.uniform(0.045, 0.060)

    cam_pos = np.array(CAM_FRONT_POS) + rng.uniform(-0.015, 0.015, 3)
    cam_target = np.array(CAM_FRONT_LOOKAT) + rng.uniform(-0.010, 0.010, 3)
    cam_fovy = CAM_FRONT_FOVY + rng.uniform(-1.0, 1.0)
    wrist_pos, wrist_quat = _wrist_cam_nominal()
    wrist_pos = wrist_pos + rng.uniform(-0.002, 0.002, 3)
    wrist_quat = _jitter_quat(wrist_quat, rng.uniform(-1.0, 1.0, 3) * np.radians(1.0))

    cube_mass = rng.uniform(0.015, 0.040)
    cube_friction = rng.uniform(0.8, 1.2)

    return SceneParams(
        scene_seed=int(scene_seed),
        table_rgb=_rounded(table_rgb, 4), cube_rgb=_rounded(cube_rgb, 4),
        plate_rgb=_rounded(plate_rgb, 4), background_rgb=_rounded(background_rgb, 4),
        light_dir=_rounded(light_dir), light_diffuse=round(float(light_diffuse), 4),
        ambient=round(float(ambient), 4),
        cube_size=round(float(cube_size), 5), plate_radius=round(float(plate_radius), 5),
        cam_front_pos=_rounded(cam_pos), cam_front_quat=_rounded(_look_at_quat(cam_pos, cam_target)),
        cam_front_fovy=round(float(cam_fovy), 3),
        wrist_cam_pos=_rounded(wrist_pos), wrist_cam_quat=_rounded(wrist_quat),
        cube_mass=round(float(cube_mass), 5), cube_friction=round(float(cube_friction), 4),
    )


def _sample_polar(rng: np.random.Generator, radius_range: tuple[float, float]) -> np.ndarray:
    """Uniform-by-area point in the annular sector in front of the robot."""
    r = np.sqrt(rng.uniform(radius_range[0] ** 2, radius_range[1] ** 2))
    a = rng.uniform(-AZIMUTH_LIMIT, AZIMUTH_LIMIT)
    return np.array([PAN_AXIS_XY[0] + r * np.cos(a), PAN_AXIS_XY[1] + r * np.sin(a)])


def min_cube_plate_distance(scene: SceneParams) -> float:
    """Centre distance that keeps the whole cube (any yaw) at least 3 cm outside the plate rim."""
    return scene.plate_radius + scene.cube_size * np.sqrt(0.5) + 0.03


def sample_cube_xy(rng: np.random.Generator, scene: SceneParams, plate_xy) -> np.ndarray:
    """A cube position in the spawn region, clear of the plate (used for disturbances)."""
    while True:
        xy = _sample_polar(rng, CUBE_RADIUS_RANGE)
        if np.linalg.norm(xy - np.asarray(plate_xy)) >= min_cube_plate_distance(scene):
            return xy


def sample_episode(scene: SceneParams, episode_seed: int) -> EpisodeParams:
    rng = episode_rng(scene.scene_seed, episode_seed)
    cube_xy = _sample_polar(rng, CUBE_RADIUS_RANGE)
    cube_yaw = rng.uniform(-np.pi, np.pi)
    while True:
        plate_xy = _sample_polar(rng, PLATE_RADIUS_RANGE)
        if np.linalg.norm(plate_xy - cube_xy) >= min_cube_plate_distance(scene):
            break
    arm = np.array(HOME_QPOS)
    arm[:5] += rng.uniform(-HOME_ARM_NOISE, HOME_ARM_NOISE, 5)
    arm[5] = rng.uniform(*HOME_GRIPPER_RANGE)
    return EpisodeParams(
        scene_seed=scene.scene_seed, episode_seed=int(episode_seed),
        cube_xy=_rounded(cube_xy), cube_yaw=round(float(cube_yaw), 6),
        plate_xy=_rounded(plate_xy), arm_qpos=_rounded(arm),
    )


def build_spec(scene: SceneParams) -> mujoco.MjSpec:
    """Robot + table + cube + plate + cameras + light. Flat shading: no shadows, no specular."""
    spec = so101_spec()
    spec.modelname = "keiarm_scene"
    spec.option.timestep = config.TIMESTEP

    spec.visual.headlight.ambient = [scene.ambient] * 3
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]  # no camera-attached light: easy to mirror in three.js
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    for mat in spec.materials:
        mat.specular = 0.0
        mat.shininess = 0.0
        mat.reflectance = 0.0
    spec.add_texture(name="background", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=scene.background_rgb,
                     rgb2=scene.background_rgb, width=8, height=48)
    for name, rgb in (("table", scene.table_rgb), ("cube", scene.cube_rgb), ("plate", scene.plate_rgb)):
        spec.add_material(name=name, rgba=[*rgb, 1.0], specular=0.0, shininess=0.0, reflectance=0.0)

    world = spec.worldbody
    world.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=scene.light_dir,
                    pos=[0.2, 0.0, 1.5], diffuse=[scene.light_diffuse] * 3, specular=[0.0, 0.0, 0.0],
                    castshadow=False)
    hx, hy, hz = TABLE_HALF_SIZE
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                   pos=[TABLE_CENTER[0], TABLE_CENTER[1], -hz], material="table")
    world.add_camera(name="cam_front", pos=scene.cam_front_pos, quat=scene.cam_front_quat,
                     fovy=scene.cam_front_fovy)
    wrist = spec.camera(WRIST_CAMERA)
    wrist.pos = scene.wrist_cam_pos
    wrist.quat = scene.wrist_cam_quat

    # Plate: a mocap body so its position is per-episode state, not baked into the model.
    plate = world.add_body(name="plate", mocap=True, pos=[0.2, 0.15, PLATE_HEIGHT / 2])
    plate.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[scene.plate_radius, PLATE_HEIGHT / 2, 0], material="plate")

    # Cube: contact settings follow the box in Menagerie's scene_box.xml.
    h = scene.cube_size / 2
    cube = world.add_body(name="cube", pos=[0.2, 0.0, h])
    cube.add_freejoint(name="cube")
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, h], mass=scene.cube_mass,
                  friction=[scene.cube_friction, 0.03, 0.003], solref=[0.01, 1.0], material="cube")
    return spec


def build_model(scene: SceneParams) -> mujoco.MjModel:
    return build_spec(scene).compile()


def reset_data(model: mujoco.MjModel, data: mujoco.MjData, scene: SceneParams, episode: EpisodeParams) -> None:
    """Put `data` into the episode's initial state (arm held at its start pose)."""
    mujoco.mj_resetData(model, data)
    for name, q in zip(JOINT_NAMES, episode.arm_qpos):
        data.qpos[model.joint(name).qposadr[0]] = q
        data.ctrl[model.actuator(name).id] = q
    adr = model.joint("cube").qposadr[0]
    data.qpos[adr : adr + 3] = [*episode.cube_xy, scene.cube_size / 2 + 1e-3]
    data.qpos[adr + 3 : adr + 7] = [np.cos(episode.cube_yaw / 2), 0.0, 0.0, np.sin(episode.cube_yaw / 2)]
    data.mocap_pos[model.body("plate").mocapid[0]] = [*episode.plate_xy, PLATE_HEIGHT / 2]
    mujoco.mj_forward(model, data)
