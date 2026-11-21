"""Towel scenes: everything is a pure function of (scene_seed, episode_seed).

Milestone 3 (docs/milestone3.md). The robot, table, cameras and lighting come from the
milestone 2 scene builder (keiarm/openarm/scene.py, used unchanged: its bar and plate are
deleted). The towel is a 2D flexcomp grid (spikes/cloth/REPORT.md, candidate A):

- edge length constraints + bending elasticity. Bending needs integrator="discrete" in
  MuJoCo 3.17 (implicitfast does not compile), so the whole towel scene runs discrete + CG
- elliptic cone and impratio 10 (as milestone 2): pyramidal lets the grabbed towel sink
  through the table
- the towel collides with a plane at the table top (contype 2), not with the table box
  (which gives ~3 contacts per triangle); the table box only meets the arms
- one vertex body per grid point (`towel_<k>`, k = i * n + j, i along the towel's local x).
  The model is built with the towel flat at the origin; the episode puts it in place by
  setting the vertices' slide joints, so the model depends on scene_seed only

scene_seed   -> SceneParams   (appearance and cameras from milestone 2, towel physics)
episode_seed -> EpisodeParams (towel pose, arm start pose)

Phase 1: the towel's size, mass, friction, bending stiffness and colours (checker front,
plain back) are drawn from the scene seed, and `cam_front` is moved closer and higher than in
milestone 2 so the towel fills a useful part of the image. Ranges below; the reach check is
tools/towel_reach.py.
"""

import colorsys
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from keiarm import config
from keiarm.openarm import scene as oa_scene
from keiarm.openarm.robot import (
    ACTUATOR_NAMES,
    GRIPPER_OPEN,
    HOME_ARM_QPOS,
    SIDES,
    arm_joints,
    finger_joints,
)
from keiarm.paths import OPENARM_XML

TOWEL = "towel"
TOWEL_N = 16  # vertices per side (training mesh). The browser uses a coarser one (Phase 4)
TOWEL_RADIUS = 0.002  # collision radius; layers rest 2 * radius apart
TOWEL_SOLVER_ITERATIONS = 50  # do not lower: at 10 the layers pass through (spike E2)
GRAB_SLOTS = 4  # connect constraints per hand (keiarm/towel/grasp.py)

# Spawn: towel centre and yaw (docs/milestone3.md). With the fold's jaw yaw the hands reach
# top-down from x = 0.13 (the hand meets the pedestal) to 0.33 along y = +-0.08..0.09, so the
# near corners (where the far corners are put down) and the far corners must both fit in
# that band: tools/towel_reach.py checks the whole range.
TOWEL_CENTER = (0.239, 0.0)
TOWEL_CENTER_JITTER = (0.008, 0.01)  # x, y
TOWEL_YAW_RANGE = np.radians(5.0)
TOWEL_DROP_GAP = 0.001  # vertices start this far above resting on the table

ARM_START_NOISE = 0.05  # [rad], as milestone 2

# Towel parameters per scene (docs/milestone3.md). A rectangle: depth along the fold (x, far
# edge onto near edge) and width across it (y). Wide, so the hands work at |y| ~ 0.09 where
# they reach closest to the robot; shallow, so the far and near corners both fit in the
# hands' x band (tools/towel_reach.py). Bending stiffness is drawn log-uniformly.
TOWEL_DEPTH_RANGE = (0.15, 0.165)  # x [m]
TOWEL_WIDTH_RANGE = (0.185, 0.20)  # y [m]
TOWEL_MASS_RANGE = (0.010, 0.030)  # [kg]
TOWEL_FRICTION_RANGE = (0.5, 0.8)
TOWEL_YOUNG_RANGE = (2e3, 1e4)  # [Pa] with thickness TOWEL_THICKNESS
TOWEL_THICKNESS = 1e-3
TOWEL_CHECKS_RANGE = (6.0, 10.0)  # checker squares across the width (square checks)

# cam_front for the towel (milestone 2's camera at (0.78, 0, 0.58) sees the towel as 2.5% of
# the image): above the table in front of the towel, looking down at the near half so the
# folded towel stays in view. The flat towel covers about 22% of the image.
CAM_FRONT_POS = (0.30, 0.0, 0.40)
CAM_FRONT_LOOKAT = (0.17, 0.0, 0.0)
CAM_FRONT_FOVY = 48.0

# Collision bits: arms (MJCF default) contype 1 / conaffinity 1.
TABLE_BITS = (4, 1)  # table box meets the arms only
TOWEL_BITS = (2, 3)  # towel meets the arms, the towel floor and itself
FLOOR_BITS = (2, 2)  # towel floor meets the towel only

_SCENE_TAG = 0x70E1C5
_EPISODE_TAG = 0x70E1E9


@dataclass(frozen=True)
class SceneParams:
    scene_seed: int
    base: dict  # milestone 2 appearance and cameras (keiarm.openarm.scene.SceneParams.to_dict())
    towel_size: list[float]  # (depth x, width y) [m], in the towel's local frame
    towel_mass: float  # [kg]
    towel_friction: float
    towel_young: float  # bending stiffness via elasticity young [Pa], thickness below
    towel_thickness: float
    towel_rgb: list[float]  # checker colours (front)
    towel_rgb2: list[float]
    towel_back_rgb: list[float]  # plain back (keiarm/towel/render.py)
    towel_checks: float  # checker squares across the width
    cam_front_pos: list[float]  # replaces milestone 2's cam_front (base["cam_front_*"] unused)
    cam_front_quat: list[float]
    cam_front_fovy: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeParams:
    scene_seed: int
    episode_seed: int
    towel_xy: list[float]
    towel_yaw: float
    start_ctrl: list[float]  # 16 actuator targets in ACTUATOR_NAMES order (= start qpos)

    def to_dict(self) -> dict:
        return asdict(self)


def scene_rng(scene_seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([_SCENE_TAG, scene_seed]))


def episode_rng(scene_seed: int, episode_seed: int, stream: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([_EPISODE_TAG, scene_seed, episode_seed, stream]))


def _r(x, nd: int = 6):
    return [round(float(v), nd) for v in x]


def sample_scene(scene_seed: int) -> SceneParams:
    base = oa_scene.sample_scene(scene_seed)
    rng = scene_rng(scene_seed)
    size = (rng.uniform(*TOWEL_DEPTH_RANGE), rng.uniform(*TOWEL_WIDTH_RANGE))
    mass = rng.uniform(*TOWEL_MASS_RANGE)
    friction = rng.uniform(*TOWEL_FRICTION_RANGE)
    young = np.exp(rng.uniform(*np.log(TOWEL_YOUNG_RANGE)))
    # Front: two checker colours of one hue family. Back: plain, a clearly different hue and
    # less saturated, so a fold shows in the images.
    hue = rng.uniform()
    rgb = colorsys.hsv_to_rgb(hue, rng.uniform(0.6, 0.9), rng.uniform(0.75, 0.95))
    rgb2 = colorsys.hsv_to_rgb((hue + 0.12) % 1.0, rng.uniform(0.3, 0.6), rng.uniform(0.85, 1.0))
    back = colorsys.hsv_to_rgb((hue + rng.uniform(0.35, 0.65)) % 1.0, rng.uniform(0.35, 0.7), rng.uniform(0.55, 0.85))
    checks = rng.uniform(*TOWEL_CHECKS_RANGE)
    cam_pos = np.array(CAM_FRONT_POS) + rng.uniform(-0.01, 0.01, 3)
    cam_target = np.array(CAM_FRONT_LOOKAT) + rng.uniform(-0.008, 0.008, 3)
    fovy = CAM_FRONT_FOVY + rng.uniform(-1.0, 1.0)
    return SceneParams(
        scene_seed=int(scene_seed), base=base.to_dict(),
        towel_size=_r(size, 5), towel_mass=round(float(mass), 5),
        towel_friction=round(float(friction), 4), towel_young=round(float(young), 1),
        towel_thickness=TOWEL_THICKNESS,
        towel_rgb=_r(rgb, 4), towel_rgb2=_r(rgb2, 4), towel_back_rgb=_r(back, 4),
        towel_checks=round(float(checks), 3),
        cam_front_pos=_r(cam_pos), cam_front_quat=_r(oa_scene.look_at_quat(cam_pos, cam_target)),
        cam_front_fovy=round(float(fovy), 3),
    )


def sample_episode(scene: SceneParams, episode_seed: int) -> EpisodeParams:
    rng = episode_rng(scene.scene_seed, episode_seed)
    jitter = np.array(TOWEL_CENTER_JITTER)
    xy = np.array(TOWEL_CENTER) + rng.uniform(-jitter, jitter)
    yaw = rng.uniform(-TOWEL_YAW_RANGE, TOWEL_YAW_RANGE)
    ctrl = []
    for side in SIDES:
        ctrl += list(np.array(HOME_ARM_QPOS[side]) + rng.uniform(-ARM_START_NOISE, ARM_START_NOISE, 7))
        ctrl.append(GRIPPER_OPEN[side])
    return EpisodeParams(scene_seed=scene.scene_seed, episode_seed=int(episode_seed),
                         towel_xy=[round(float(v), 6) for v in xy], towel_yaw=round(float(yaw), 6),
                         start_ctrl=[round(float(v), 6) for v in ctrl])


def towel_xml(scene: SceneParams, n: int) -> str:
    sx, sy = (v / (n - 1) for v in scene.towel_size)
    return f"""
    <flexcomp name="{TOWEL}" type="grid" count="{n} {n} 1" spacing="{sx:.8g} {sy:.8g} {sy:.8g}" dim="2"
              radius="{TOWEL_RADIUS:g}" mass="{scene.towel_mass:g}" material="towel"
              pos="0 0 {TOWEL_RADIUS + TOWEL_DROP_GAP:g}">
      <edge equality="true" damping="0"/>
      <elasticity young="{scene.towel_young:g}" poisson="0" thickness="{scene.towel_thickness:g}" elastic2d="bend"/>
      <contact selfcollide="sap" contype="{TOWEL_BITS[0]}" conaffinity="{TOWEL_BITS[1]}" condim="3"
               friction="{scene.towel_friction:g} 0.005 0.0001" solref="0.01 1"/>
    </flexcomp>"""


def grab_equalities_xml() -> str:
    return "\n".join(
        f'<connect name="{side}_grab{k}" body1="openarm_{side}_ee_base_link" body2="{TOWEL}_0" anchor="0 0 0" '
        f'active="false" solref="0.01 1"/>' for side in SIDES for k in range(GRAB_SLOTS))


def build_xml(scene: SceneParams, n: int = TOWEL_N, web: bool = False) -> str:
    """The full MJCF text (flexcomp is a macro, so the towel goes in as XML). With `web`, the
    mesh files are relative to one directory "assets" (the browser's copy, as milestone 2)."""
    spec = oa_scene.build_spec(oa_scene.SceneParams(**scene.base))
    spec.modelname = "keiarm_towel_scene"
    spec.delete(spec.body("bar"))
    spec.delete(spec.body("plate"))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = TOWEL_SOLVER_ITERATIONS
    # elliptic cone + impratio 10 as milestone 2 (openarm_spec). Pyramidal with impratio 10
    # lets the towel sink through the table once the hands grab it (Phase 0, 3/3 episodes).
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    table = spec.geom("table")
    table.contype, table.conaffinity = TABLE_BITS
    spec.worldbody.add_geom(name="towel_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5, 0.6, 0.01],
                            rgba=[0, 0, 0, 0], contype=FLOOR_BITS[0], conaffinity=FLOOR_BITS[1])
    spec.add_texture(name="towel", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=scene.towel_rgb, rgb2=scene.towel_rgb2, width=256, height=256)
    # the builtin checker texture holds 2 x 2 squares; it repeats texrepeat times across the towel
    spec.add_material(name="towel", textures=["", "towel"], texrepeat=[scene.towel_checks / 2 * scene.towel_size[0] / scene.towel_size[1], scene.towel_checks / 2], **oa_scene.FLAT)
    cam = spec.camera("cam_front")
    cam.pos, cam.quat, cam.fovy = scene.cam_front_pos, scene.cam_front_quat, scene.cam_front_fovy
    # Mesh paths stay relative to the vendored MJCF; make them absolute for the XML text.
    if web:
        from keiarm.openarm.export import relative_mesh_file

        spec.meshdir = "assets"
        for mesh in spec.meshes:
            mesh.file = relative_mesh_file(mesh.file)
    else:
        spec.meshdir = str(OPENARM_XML.parent / spec.meshdir)
    xml = spec.to_xml()  # <compiler angle="radian">
    xml = xml.replace("</worldbody>", towel_xml(scene, n) + "\n</worldbody>", 1)
    eqs = grab_equalities_xml()
    if "<equality>" in xml:
        xml = xml.replace("<equality>", "<equality>\n" + eqs, 1)
    else:
        xml = xml.replace("</mujoco>", f"<equality>\n{eqs}\n</equality>\n</mujoco>", 1)
    return xml


def build_model(scene: SceneParams, n: int = TOWEL_N) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(build_xml(scene, n))
    model.opt.timestep = config.TIMESTEP  # the XML text carries a rounded value
    return model


class TowelIndex:
    """Ids and addresses of the towel in a compiled model."""

    def __init__(self, model: mujoco.MjModel):
        self.flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, TOWEL)
        self.vadr = int(model.flex_vertadr[self.flex])
        self.nvert = int(model.flex_vertnum[self.flex])
        self.n = round(np.sqrt(self.nvert))
        self.vert_body = np.array([model.body(f"{TOWEL}_{k}").id for k in range(self.nvert)])
        jnt = np.array([[model.body_jntadr[b] + a for a in range(3)] for b in self.vert_body])
        self.vert_qadr = model.jnt_qposadr[jnt]  # (nvert, 3) slide joints x, y, z
        self.vert_dadr = model.jnt_dofadr[jnt]
        self.rest = model.body_pos[self.vert_body].copy()  # towel flat, centred at the origin

    def vid(self, i: int, j: int) -> int:
        return i * self.n + j

    def verts(self, data: mujoco.MjData) -> np.ndarray:
        return data.flexvert_xpos[self.vadr : self.vadr + self.nvert].reshape(-1, 3)

    def place(self, data: mujoco.MjData, xy, yaw: float) -> None:
        """Lay the towel flat at (xy, yaw), at rest (slide joints hold offsets from `rest`)."""
        c, s = np.cos(yaw), np.sin(yaw)
        local = self.rest[:, :2]
        target = np.column_stack([c * local[:, 0] - s * local[:, 1] + xy[0],
                                  s * local[:, 0] + c * local[:, 1] + xy[1], self.rest[:, 2]])
        data.qpos[self.vert_qadr] = target - self.rest
        data.qvel[self.vert_dadr] = 0.0


def reset_data(model: mujoco.MjModel, data: mujoco.MjData, ix: TowelIndex, episode: EpisodeParams) -> None:
    """Put `data` into the episode's initial state (towel flat, arms at the start pose).
    The grasp rule's state is reset separately (keiarm.towel.grasp.GraspRule.reset)."""
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
    ix.place(data, episode.towel_xy, episode.towel_yaw)
    mujoco.mj_forward(model, data)
