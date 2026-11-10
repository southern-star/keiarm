import json
from dataclasses import asdict

import mujoco
import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.export import REPLAY_TAIL_STEPS, export_web, scene_bundle
from keiarm.paths import SO101_DIR
from keiarm.rollout import planned_teleport
from keiarm.scene import PLATE_HEIGHT
from keiarm.task import TaskIndex, is_success, read_state


def test_bundle_replays_from_its_own_xml():
    """The MJCF text in the bundle (what the browser loads) reproduces the expert's success."""
    bundle = json.loads(json.dumps(scene_bundle(100002, [0])))
    assert bundle["mujoco"] == mujoco.__version__
    assets = {f: (SO101_DIR / "assets" / f).read_bytes() for f in bundle["mesh_files"]}
    model = mujoco.MjModel.from_xml_string(bundle["xml"], assets)
    model.opt.timestep = bundle["constants"]["timestep"]
    data = mujoco.MjData(model)
    ix = TaskIndex(model)
    ep = bundle["episodes"][0]
    assert ep["expert"]["success"] and len(ep["expert"]["actions"]) == ep["expert"]["steps"] + REPLAY_TAIL_STEPS
    # The eval suite's disturbance travels with the episode so the browser can apply the same one.
    env = KeiArmEnv(100002, render=False)
    env.reset(0)
    assert ep["teleport"] == asdict(planned_teleport(env, "random"))

    data.qpos[ix.qadr] = ep["arm_qpos"]
    data.ctrl[ix.act] = ep["arm_qpos"]
    size = bundle["scene"]["cube_size"]
    a = ix.cube_qadr
    data.qpos[a : a + 7] = [*ep["cube_xy"], size / 2 + 1e-3, np.cos(ep["cube_yaw"] / 2), 0, 0,
                            np.sin(ep["cube_yaw"] / 2)]
    data.mocap_pos[ix.plate_mocap] = [*ep["plate_xy"], PLATE_HEIGHT / 2]
    mujoco.mj_forward(model, data)
    for action in ep["expert"]["actions"]:
        data.ctrl[ix.act] = action
        mujoco.mj_step(model, data, nstep=config.SUBSTEPS)

    class _Scene:
        cube_size = size
        plate_radius = bundle["scene"]["plate_radius"]

    st = read_state(model, ix, data)
    assert is_success(st, _Scene)
    np.testing.assert_allclose(st.cube_pos, ep["expert"]["final_cube_pos"], atol=5e-3)


def test_export_writes_bundles_meshes_and_index(tmp_path):
    paths = export_web([3], [0], out=tmp_path)
    assert [p.name for p in paths] == ["3.json"]
    index = json.loads((tmp_path / "scenes" / "index.json").read_text())
    assert index["scenes"] == [3] and index["episodes"] == [0]
    bundle = json.loads(paths[0].read_text())
    for f in bundle["mesh_files"]:
        assert (tmp_path / "assets" / "so101" / f).stat().st_size > 0
    assert (tmp_path / "assets" / "so101" / "LICENSE").exists()
    assert config.split_of(bundle["scene"]["scene_seed"]) == "train"
