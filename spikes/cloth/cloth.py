"""布たたみスパイクの共通部品（spikes/cloth/ 専用。本体のコードからは使わない）。

- `ClothCfg`: 布と物理の設定。`cloth_xml()` が flexcomp の MJCF 断片を返す
- `table_scene_xml()`: 机 + 布 + 掴み用 mocap 2 個 + 拘束スロット（E1, E2, E4, E5）
- `Grabber`: mocap と布の頂点をつなぐ connect 拘束の実行時切り替え
- 計測: 破綻判定、層の順序、形の保持

頂点の番号は flexcomp grid の順: k = i * N + j（i が x 方向、j が y 方向）。
頂点ごとにボディ `cloth_k`（slide 関節 3 本）ができる。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace

import mujoco
import numpy as np


@dataclass(frozen=True)
class ClothCfg:
    n: int = 16  # 一辺の頂点数
    side: float = 0.24  # 一辺 [m]
    mass: float = 0.02  # [kg]
    radius: float = 0.002  # 衝突の半径（層の厚み = 2 * radius）
    equality: str = "true"  # edge equality: "true" / "vert" / "false"（E1: vert は箱の縁で大きく伸びる）
    edge_damping: float = 0.0
    elastic2d: str = "bend"  # none / bend / stretch / both
    young: float = 5e3
    poisson: float = 0.0
    thickness: float = 1e-3
    damping: float = 0.0  # elasticity damping（Rayleigh、時間の単位）
    selfcollide: str = "auto"  # none / narrow / bvh / sap / auto
    friction: float = 0.6
    condim: int = 3
    solref: tuple[float, float] = (0.01, 1.0)
    # 物理
    timestep: float = 1 / 240
    solver: str = "CG"  # Newton / CG（E1: 接触が多いと CG が数倍速い）
    iterations: int = 50
    integrator: str = "discrete"  # 3.17.0 では flex の elasticity は discrete 必須
    cone: str = "pyramidal"
    rgba_front: tuple[float, ...] = (0.85, 0.25, 0.20, 1.0)
    textured: bool = True

    def spacing(self) -> float:
        return self.side / (self.n - 1)

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> ClothCfg:
        return replace(self, **kw)


def vid(cfg: ClothCfg, i: int, j: int) -> int:
    return i * cfg.n + j


def option_xml(cfg: ClothCfg) -> str:
    return (f'<option timestep="{cfg.timestep:.8g}" solver="{cfg.solver}" iterations="{cfg.iterations}" '
            f'integrator="{cfg.integrator}" cone="{cfg.cone}"/>')


def assets_xml(cfg: ClothCfg) -> str:
    # 表: 格子模様（2 色）。裏の色は MuJoCo の flex では別に指定できない（E4 で確認）
    return """
    <texture name="cloth_tex" type="2d" builtin="checker" rgb1="0.90 0.30 0.20" rgb2="0.95 0.85 0.30"
             width="256" height="256"/>
    <material name="cloth_mat" texture="cloth_tex" texrepeat="4 4" specular="0" shininess="0"/>
    <material name="table_mat" rgba="0.55 0.55 0.6 1" specular="0" shininess="0"/>
    """


def cloth_xml(cfg: ClothCfg, pos=(0.0, 0.0, 0.05), euler=(0.0, 0.0, 0.0), name: str = "cloth") -> str:
    s = cfg.spacing()
    mat = 'material="cloth_mat"' if cfg.textured else f'rgba="{" ".join(map(str, cfg.rgba_front))}"'
    edge = f'<edge equality="{cfg.equality}" damping="{cfg.edge_damping}"/>'
    elas = ""
    if cfg.elastic2d != "none":  # none のとき elasticity 要素を書かない（edge 拘束と young は併用不可）
        elas = (f'<elasticity young="{cfg.young:g}" poisson="{cfg.poisson:g}" thickness="{cfg.thickness:g}" '
                f'damping="{cfg.damping:g}" elastic2d="{cfg.elastic2d}"/>')
    contact = (f'<contact selfcollide="{cfg.selfcollide}" friction="{cfg.friction} 0.005 0.0001" '
               f'condim="{cfg.condim}" solref="{cfg.solref[0]} {cfg.solref[1]}"/>')
    return f"""
    <flexcomp name="{name}" type="grid" count="{cfg.n} {cfg.n} 1" spacing="{s:.8g} {s:.8g} {s:.8g}" dim="2"
              radius="{cfg.radius:g}" mass="{cfg.mass:g}" pos="{pos[0]:g} {pos[1]:g} {pos[2]:g}"
              euler="{euler[0]:g} {euler[1]:g} {euler[2]:g}" {mat}>
      {edge}
      {elas}
      {contact}
    </flexcomp>"""


N_SLOTS = 4  # mocap 1 個あたりの拘束スロット数


def grab_equalities_xml(mocaps: list[str], cloth: str = "cloth") -> str:
    """mocap ごとに N_SLOTS 本の connect（非アクティブ）。相手は仮に cloth_0。実行時に差し替える。"""
    out = []
    for m in mocaps:
        for k in range(N_SLOTS):
            out.append(f'<connect name="{m}_slot{k}" body1="{m}" body2="{cloth}_0" anchor="0 0 0" active="false"'
                       ' solref="0.01 1"/>')
    return "\n".join(out)


def table_scene_xml(cfg: ClothCfg, cloth_pos=(0.0, 0.0, 0.01), cloth_euler=(0, 0, 0), extra_world: str = "",
                    mocaps=("grip_a", "grip_b")) -> str:
    """机（上面 z = 0）+ 布 + 掴み用 mocap。"""
    mocap_xml = "\n".join(
        f'<body name="{m}" mocap="true" pos="0 0 0.3"><geom type="sphere" size="0.006" rgba="0.1 0.6 0.9 0.6" '
        f'contype="0" conaffinity="0"/></body>' for m in mocaps)
    return f"""<mujoco model="cloth_table">
  {option_xml(cfg)}
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="0"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0 0 0"/></visual>
  <asset>{assets_xml(cfg)}</asset>
  <worldbody>
    <!-- 机: 見た目は箱、布との衝突は平面（箱だと三角形ごとに約 3 点の接触ができ、平面だと頂点ごとに 1 点） -->
    <geom name="table_vis" type="box" size="0.4 0.4 0.02" pos="0 0 -0.0201" material="table_mat" contype="0" conaffinity="0"/>
    <geom name="table" type="plane" size="0.4 0.4 0.02" rgba="0 0 0 0" friction="0.6 0.005 0.0001"/>
    <camera name="top" pos="0 0 0.55" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <camera name="side" pos="0 -0.55 0.18" xyaxes="1 0 0 0 0.3 1" fovy="45"/>
    {extra_world}
    {mocap_xml}
    {cloth_xml(cfg, cloth_pos, cloth_euler)}
  </worldbody>
  <equality>{grab_equalities_xml(list(mocaps))}</equality>
</mujoco>"""


class Grabber:
    """mocap に布の頂点を拘束する。拘束スロットの相手ボディを実行時に差し替える方式。

    grab(): 指定頂点の現在位置を mocap 座標で anchor（body1 側）に入れ、body2 = 頂点ボディ、
    anchor2 = 0 にして有効化する。掴んだ瞬間に跳ばない。
    """

    def __init__(self, model: mujoco.MjModel, mocap: str, cloth: str = "cloth"):
        self.m = model
        self.mocap = mocap
        self.mocap_body = model.body(mocap).id
        self.mocap_id = model.body_mocapid[self.mocap_body]
        self.slots = [model.equality(f"{mocap}_slot{k}").id for k in range(N_SLOTS)]
        self.cloth = cloth
        self.held: list[int] = []

    def grab(self, data: mujoco.MjData, verts: list[int]) -> None:
        assert len(verts) <= N_SLOTS
        mpos = data.mocap_pos[self.mocap_id]
        mquat = data.mocap_quat[self.mocap_id]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, mquat)
        rot = rot.reshape(3, 3)
        for slot, v in zip(self.slots, verts):
            body = self.m.body(f"{self.cloth}_{v}").id
            self.m.eq_obj2id[slot] = body
            vpos = data.xpos[body]
            self.m.eq_data[slot, 0:3] = rot.T @ (vpos - mpos)  # anchor（mocap 座標）
            self.m.eq_data[slot, 3:6] = 0.0  # 頂点ボディ側
            data.eq_active[slot] = 1
        self.held = list(verts)

    def release(self, data: mujoco.MjData) -> None:
        for slot in self.slots:
            data.eq_active[slot] = 0
        self.held = []


# --- 計測 ---------------------------------------------------------------------------


def cloth_verts(model, data, name="cloth") -> np.ndarray:
    f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
    a, n = model.flex_vertadr[f], model.flex_vertnum[f]
    return data.flexvert_xpos[a : a + n].reshape(-1, 3)


def broken(model, data, verts: np.ndarray, radius: float, bound: float = 1.0) -> str | None:
    """破綻の種類（なければ None）。"""
    if not np.isfinite(verts).all() or not np.isfinite(data.qacc).all():
        return "nan"
    if np.abs(verts).max() > bound:
        return "explode"
    if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
        return "badqacc"
    if verts[:, 2].min() < -0.01:  # 机（上面 0）を 1 cm 以上すり抜けた（衝突の瞬間の沈みは許す）
        return "through_table"
    return None


def edge_strain(model, data, name="cloth") -> float:
    """辺の長さの最大相対誤差。"""
    f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
    a, n = model.flex_edgeadr[f], model.flex_edgenum[f]
    l0 = model.flexedge_length0[a : a + n]
    lv = data.flexedge_length[a : a + n]
    return float(np.max(np.abs(lv / l0 - 1.0)))


def strain_stats(model, data, name="cloth") -> tuple[float, float]:
    """辺の相対誤差の (平均, 95 パーセンタイル)。"""
    f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
    a, n = model.flex_edgeadr[f], model.flex_edgenum[f]
    e = np.abs(data.flexedge_length[a : a + n] / model.flexedge_length0[a : a + n] - 1.0)
    return float(e.mean()), float(np.percentile(e, 95))


@dataclass
class StepTimer:
    wall: float = 0.0
    sim: float = 0.0
    ncon: list = field(default_factory=list)
    nefc: list = field(default_factory=list)

    def step(self, model, data, n: int = 1):
        t = time.perf_counter()
        mujoco.mj_step(model, data, n)
        self.wall += time.perf_counter() - t
        self.sim += n * model.opt.timestep
        self.ncon.append(data.ncon)
        self.nefc.append(data.nefc)

    @property
    def rtf(self) -> float:
        return self.sim / max(self.wall, 1e-12)


def fold_layers_half(cfg: ClothCfg, skip_crease: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """二つ折り（i >= n/2 の側を i < n/2 の上へ）の (下層, 上層) 頂点の組。
    skip_crease: 折り目に接する行（i = n/2 - 1。上層は i = n/2）を除く。"""
    n, h = cfg.n, cfg.n // 2
    lo, hi = [], []
    for i in range(h - 1 if skip_crease else h):
        for j in range(n):
            lo.append(vid(cfg, i, j))
            hi.append(vid(cfg, n - 1 - i, j))
    return [(np.array(lo), np.array(hi))]


def fold_layers_quarter(cfg: ClothCfg, skip_crease: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """四つ折り（続けて j >= n/2 の側を j < n/2 の上へ）。下から
    (i, j), (n-1-i, j), (n-1-i, n-1-j), (i, n-1-j) の順に重なる。
    skip_crease: 2 本の折り目に接する行と列（i = n/2 - 1、j = n/2 - 1）を除く。"""
    n, h = cfg.n, cfg.n // 2
    stacks = [[], [], [], []]
    m = h - 1 if skip_crease else h
    for i in range(m):
        for j in range(m):
            stacks[0].append(vid(cfg, i, j))
            stacks[1].append(vid(cfg, n - 1 - i, j))
            stacks[2].append(vid(cfg, n - 1 - i, n - 1 - j))
            stacks[3].append(vid(cfg, i, n - 1 - j))
    s = [np.array(x) for x in stacks]
    return [(s[0], s[1]), (s[1], s[2]), (s[2], s[3])]


def layer_violation(verts: np.ndarray, pairs, tol: float = 0.0) -> float:
    """上層の頂点が対応する下層の頂点より下（z_上 < z_下 - tol）にある割合。"""
    bad = tot = 0
    for lo, hi in pairs:
        bad += int(np.sum(verts[hi, 2] < verts[lo, 2] - tol))
        tot += len(lo)
    return bad / max(tot, 1)


def flex_triangles(model, name="cloth") -> np.ndarray:
    f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
    n = model.flex_elemnum[f]
    d = model.flex_dim[f]
    adr = model.flex_elemdataadr[f]
    return model.flex_elem[adr : adr + n * (d + 1)].reshape(n, d + 1).copy()


def self_intersections(verts: np.ndarray, tris: np.ndarray) -> int:
    """布の中心面どうしの交差の数（辺が、その辺と頂点を共有しない三角形を貫く数）。
    0 なら層のすり抜けはない（層の順序の違反は、めくれや巻き込みによる）。"""
    edges = np.unique(np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    e1, e2 = b - a, c - a
    tmin, tmax = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
    count = 0
    for i, j in edges:
        p, q = verts[i], verts[j]
        lo, hi = np.minimum(p, q), np.maximum(p, q)
        cand = np.all((tmax >= lo) & (tmin <= hi), axis=1)
        cand &= ~np.any((tris == i) | (tris == j), axis=1)
        if not cand.any():
            continue
        d = q - p
        A, E1, E2 = a[cand], e1[cand], e2[cand]
        h = np.cross(d, E2)
        det = np.einsum("ij,ij->i", E1, h)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = p - A
        u = inv * np.einsum("ij,ij->i", sv, h)
        qv = np.cross(sv, E1)
        v = inv * (qv @ d)
        t = inv * np.einsum("ij,ij->i", E2, qv)
        count += int(np.sum(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)))
    return count


def projected_area(verts: np.ndarray, res: float = 0.004) -> float:
    """上から見た投影面積（格子で数える。頂点の三角形を塗るのは重いので、頂点の凸包でなく
    頂点を中心に res 四方を塗った近似）。比較用なので相対値だけ使う。"""
    xy = np.floor(verts[:, :2] / res).astype(int)
    return len({tuple(p) for p in xy}) * res * res


def convex_area(verts: np.ndarray) -> float:
    """上から見た凸包の面積 [m^2]。"""
    pts = verts[:, :2]
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def renderer(model, w=320, h=240):
    return mujoco.Renderer(model, height=h, width=w)
