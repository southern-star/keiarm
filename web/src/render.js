// three.js view of a MuJoCo model, shaded to match mujoco.Renderer as configured in
// keiarm/scene.py: colour * (ambient + diffuse * max(n.l, 0)), no shadows, no specular,
// no colour-space conversion. World coordinates are MuJoCo's (z up); cameras use the same
// convention in both libraries (look along -z, +y up), so poses are copied as they are.
import * as THREE from 'three';

THREE.ColorManagement.enabled = false;

const GEOM = { PLANE: 0, SPHERE: 2, CAPSULE: 3, CYLINDER: 5, BOX: 6, MESH: 7 };
const MAX_VISIBLE_GROUP = 2; // mujoco.Renderer's default: groups 0-2 are drawn, 3+ (collision) are not
const DEFAULT_RGBA = [0.5, 0.5, 0.5, 1];

const VERTEX = /* glsl */ `
  varying vec3 vNormal;
  void main() {
    vNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
  }`;
const FRAGMENT = /* glsl */ `
  uniform vec3 uColor;
  uniform vec3 uToLight;
  uniform float uAmbient;
  uniform float uDiffuse;
  varying vec3 vNormal;
  void main() {
    float lambert = max(dot(normalize(vNormal), uToLight), 0.0);
    gl_FragColor = vec4(uColor * (uAmbient + uDiffuse * lambert), 1.0);
  }`;

function meshGeometry(model, meshId) {
  const faceAdr = model.mesh_faceadr[meshId];
  const faceNum = model.mesh_facenum[meshId];
  const vertAdr = model.mesh_vertadr[meshId];
  const normalAdr = model.mesh_normaladr[meshId];
  const { mesh_vert: vert, mesh_normal: normal, mesh_face: face, mesh_facenormal: faceNormal } = model;
  const positions = new Float32Array(faceNum * 9);
  const normals = new Float32Array(faceNum * 9);
  for (let f = 0; f < faceNum; f++) {
    for (let k = 0; k < 3; k++) {
      const v = 3 * (vertAdr + face[3 * (faceAdr + f) + k]);
      const n = 3 * (normalAdr + faceNormal[3 * (faceAdr + f) + k]);
      positions.set([vert[v], vert[v + 1], vert[v + 2]], 9 * f + 3 * k);
      normals.set([normal[n], normal[n + 1], normal[n + 2]], 9 * f + 3 * k);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return geometry;
}

function primitiveGeometry(type, size) {
  switch (type) {
    case GEOM.SPHERE: return new THREE.SphereGeometry(size[0], 32, 16);
    case GEOM.CAPSULE: return new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 24).rotateX(Math.PI / 2);
    case GEOM.CYLINDER: return new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 48).rotateX(Math.PI / 2);
    case GEOM.BOX: return new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    case GEOM.PLANE: return new THREE.PlaneGeometry(20, 20);
    default: return null;
  }
}

export class SceneView {
  constructor(sim) {
    this.sim = sim;
    const { model, scene: params } = sim;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color().setRGB(...params.background_rgb);
    const toLight = new THREE.Vector3(...params.light_dir).normalize().negate();
    this.geoms = [];
    const meshCache = new Map();
    for (let g = 0; g < model.ngeom; g++) {
      if (model.geom_group[g] > MAX_VISIBLE_GROUP) continue;
      const type = model.geom_type[g];
      const size = [0, 1, 2].map((k) => model.geom_size[3 * g + k]);
      let geometry;
      if (type === GEOM.MESH) {
        const id = model.geom_dataid[g];
        if (!meshCache.has(id)) meshCache.set(id, meshGeometry(model, id));
        geometry = meshCache.get(id);
      } else {
        geometry = primitiveGeometry(type, size);
      }
      if (!geometry) continue;
      let rgba = [0, 1, 2, 3].map((k) => model.geom_rgba[4 * g + k]);
      const mat = model.geom_matid[g];
      const isDefault = rgba.every((v, k) => Math.abs(v - DEFAULT_RGBA[k]) < 1e-6);
      if (mat >= 0 && isDefault) rgba = [0, 1, 2, 3].map((k) => model.mat_rgba[4 * mat + k]);
      if (rgba[3] === 0) continue;
      const material = new THREE.ShaderMaterial({
        vertexShader: VERTEX,
        fragmentShader: FRAGMENT,
        uniforms: {
          uColor: { value: new THREE.Vector3(rgba[0], rgba[1], rgba[2]) },
          uToLight: { value: toLight },
          uAmbient: { value: params.ambient },
          uDiffuse: { value: params.light_diffuse },
        },
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.matrixAutoUpdate = false;
      this.scene.add(mesh);
      this.geoms.push({ id: g, mesh });
    }
    this.geometries = [...meshCache.values()];

    // Policy cameras: fixed-size off-screen targets, read back as RGBA bytes.
    const { image_width: w, image_height: h, cameras } = sim.c;
    this.width = w;
    this.height = h;
    this.target = new THREE.WebGLRenderTarget(w, h, { samples: 4, colorSpace: THREE.NoColorSpace });
    this.cameras = cameras.map((name) => {
      const id = sim.mujoco.mj_name2id(model, sim.mujoco.mjtObj.mjOBJ_CAMERA.value, name);
      const cam = new THREE.PerspectiveCamera(model.cam_fovy[id], w / h, 0.005, 20);
      cam.matrixAutoUpdate = false;
      return { name, id, cam };
    });
    this._pixels = new Uint8Array(w * h * 4);
    this.sync();
  }

  // Copy geom and camera poses from MjData into the scene graph.
  sync() {
    const d = this.sim.data;
    const m = new THREE.Matrix4();
    const set = (target, pos, mat, i) => {
      const p = 3 * i;
      const r = 9 * i;
      m.set(mat[r], mat[r + 1], mat[r + 2], pos[p],
        mat[r + 3], mat[r + 4], mat[r + 5], pos[p + 1],
        mat[r + 6], mat[r + 7], mat[r + 8], pos[p + 2],
        0, 0, 0, 1);
      target.matrix.copy(m);
      target.matrixWorldNeedsUpdate = true;
    };
    for (const { id, mesh } of this.geoms) set(mesh, d.geom_xpos, d.geom_xmat, id);
    for (const { id, cam } of this.cameras) {
      set(cam, d.cam_xpos, d.cam_xmat, id);
      cam.updateMatrixWorld(true);
    }
  }

  // Render one policy camera; returns RGBA bytes, row 0 = top of the image (like mujoco.Renderer).
  renderCamera(renderer, name) {
    const entry = this.cameras.find((c) => c.name === name);
    const { width: w, height: h } = this;
    const previous = renderer.getRenderTarget();
    renderer.setRenderTarget(this.target);
    renderer.render(this.scene, entry.cam);
    renderer.readRenderTargetPixels(this.target, 0, 0, w, h, this._pixels);
    renderer.setRenderTarget(previous);
    const out = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      out.set(this._pixels.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
    }
    return out;
  }

  dispose() {
    for (const { mesh } of this.geoms) {
      mesh.material.dispose();
      if (!this.geometries.includes(mesh.geometry)) mesh.geometry.dispose();
    }
    for (const g of this.geometries) g.dispose();
    this.target.dispose();
  }
}
