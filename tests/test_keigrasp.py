import os

import numpy as np
import pytest

from keiarm import config

os.environ.setdefault("MUJOCO_GL", "egl")
torch = pytest.importorskip("torch")

from keiarm.rollout import CHUNK
from keigrasp.data import collect_episode, load_dataset, require_train_seed, save_episode
from keigrasp.model import KeiGrasp, ModelConfig, load_checkpoint, save_checkpoint
from keigrasp.train import augment


def test_eval_seeds_are_refused_as_training_data(tmp_path):
    for seed in (config.EVAL_SCENE_SEEDS[0], config.EVAL_SCENE_SEEDS[-1], 4000, 99999):
        with pytest.raises(ValueError):
            require_train_seed(seed)
        with pytest.raises(ValueError):
            collect_episode(seed, 0, 0.0)
    # A stray file with an eval seed makes the loader fail rather than silently train on it.
    np.savez(tmp_path / "bad.npz", scene_seed=100000, images=np.zeros((1, 2, 120, 160, 3), np.uint8),
             joints=np.zeros((1, 6), np.float32), chunks=np.zeros((1, CHUNK, 6), np.float32))
    with pytest.raises(ValueError):
        load_dataset([tmp_path])


def test_collect_episode_and_reload(tmp_path):
    try:
        ep = collect_episode(7, 0, sigma=0.01, label_every=5)
    except Exception as exc:  # noqa: BLE001 - no GL context available
        pytest.skip(f"renderer unavailable: {exc}")
    n = len(ep["steps"])
    assert n > 5 and ep["success_at"] > 0
    assert ep["images"].shape == (n, 2, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3)
    assert ep["joints"].shape == (n, 6) and ep["chunks"].shape == (n, CHUNK, 6)
    assert np.all(np.diff(ep["steps"]) == 5)
    save_episode(tmp_path, ep)
    data = load_dataset([tmp_path])
    np.testing.assert_array_equal(data["images"], ep["images"])
    assert set(data["scene"]) == {7}
    again = collect_episode(7, 0, sigma=0.01, label_every=5)
    np.testing.assert_array_equal(again["chunks"], ep["chunks"])  # collection is deterministic


def test_model_shapes_size_and_normalization(tmp_path):
    model = KeiGrasp(ModelConfig())
    params = sum(p.numel() for p in model.parameters())
    assert params < 20e6
    joints = torch.randn(64, 6) * 0.5
    chunks = joints[:, None, :] + torch.randn(64, CHUNK, 6) * 0.1
    rng = torch.tensor([[-1.9, 1.9]] * 5 + [[-0.17, 1.75]])
    model.set_normalization(rng, joints, chunks)
    target = model.normalize_target(joints, chunks)
    assert abs(float(target.mean())) < 0.05 and 0.8 < float(target.std()) < 1.2
    images = torch.rand(2, 2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    out = model(images, joints[:2])
    assert out.shape == (2, CHUNK, 6)
    # forward() undoes normalize_target()
    back = model.predict_normalized(images, joints[:2]) * model.target_std + model.target_mean
    torch.testing.assert_close(out, back + model.relative * joints[:2, None, :])
    save_checkpoint(tmp_path / "m.pt", model)
    loaded, _ = load_checkpoint(tmp_path / "m.pt")
    torch.testing.assert_close(loaded(images, joints[:2]), model.eval()(images, joints[:2]))


def test_augmentation_keeps_shape_and_range():
    x = torch.rand(4, 2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    y = augment(x)
    assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1
    assert not torch.equal(x, y)


def test_dagger_collection_lets_the_policy_drive_and_the_expert_label(tmp_path):
    from keiarm.env import KeiArmEnv
    from keiarm.expert import Expert
    from keiarm.policies import make_policy
    from keiarm.rollout import expert_chunk

    try:
        ep = collect_episode(11, 0, label_every=10, max_steps=40, disturb=15, policy=make_policy("hold"))
    except Exception as exc:  # noqa: BLE001 - no GL context available
        pytest.skip(f"renderer unavailable: {exc}")
    assert ep["source"] == "dagger" and ep["teleport_step"] == 15 and ep["success_at"] == -1
    # The arm never moved (the idle policy drove) ...
    np.testing.assert_allclose(ep["joints"], ep["joints"][:1].repeat(len(ep["joints"]), 0), atol=0.02)
    # ... but the labels are the expert's plan from those states: it wants to move.
    assert np.abs(ep["chunks"][:, -1, :5] - ep["joints"][:, :5]).max() > 0.1
    env = KeiArmEnv(11, render=False)
    env.reset(0)
    first = expert_chunk(env, Expert(env.model, env.scene))
    if ep["steps"][0] == 0:
        np.testing.assert_allclose(ep["chunks"][0], first, atol=1e-6)
    assert save_episode(tmp_path, ep).name.endswith("_dagger.npz")
