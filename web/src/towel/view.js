// three.js view of a towel scene (milestone 3): the robot and table as in render.js, plus the
// towel drawn from the flex vertices every frame. Shading matches mujoco.Renderer as set up
// in keiarm/towel (colour * (ambient + diffuse * max(n.l, 0)), per pixel as render.js; per
// vertex lighting was tried and changed the difference by less than 0.4 / 255):
// - front: the material's builtin checker texture (MuJoCo's 2 x 2 squares, texrepeat times),
//   on the surface `radius` above the mid-surface (MuJoCo draws a 2D flex as a shell)
// - back: the plain back colour, drawn on the other side (keiarm/towel/render.py appends back
//   triangles to the mjvScene for this; here it is a BackSide material)
// Normals are recomputed every frame, so a coarse mesh still looks smooth.
import * as THREE from 'three';
import { SceneView } from '../render.js';

const TEX_VERTEX = /* glsl */ `
  uniform float uOffset;
  uniform float uSide;
  varying vec3 vNormal;
  varying vec3 vPos;
  varying vec2 vUv;
  void main() {
    vNormal = uSide * normal;
    vUv = uv;
    vPos = position + uSide * uOffset * normal;
    gl_Position = projectionMatrix * viewMatrix * vec4(vPos, 1.0);
  }`;
const TEX_FRAGMENT = /* glsl */ `
  uniform vec3 uColor;
  uniform sampler2D uMap;
  uniform float uUseMap;
  uniform vec2 uRepeat;
  uniform vec3 uToLight;
  uniform float uAmbient;
  uniform float uDiffuse;
  uniform float uFlat;
  varying vec3 vNormal;
  varying vec3 vPos;
  varying vec2 vUv;
  void main() {
    // The back is lit flat per triangle, as the mjGEOM_TRIANGLEs of keiarm/towel/render.py;
    // the front smoothly, as MuJoCo's flex. cross(dFdx, dFdy) points towards the viewer, as the
    // visible back surface's outward normal does.
    vec3 faceN = normalize(cross(dFdx(vPos), dFdy(vPos)));
    vec3 n = uFlat > 0.5 ? faceN : normalize(vNormal);
    float shade = uAmbient + uDiffuse * max(dot(n, uToLight), 0.0);
    vec3 base = uUseMap > 0.5 ? texture2D(uMap, vUv * uRepeat).rgb : uColor;
    gl_FragColor = vec4(base * shade, 1.0);
  }`;
// MuJoCo's builtin checker (user_texture.cc): rgb1 in the top-left and bottom-right quarters.
function checkerTexture(rgb1, rgb2, size = 256) {
  const data = new Uint8Array(size * size * 4);
  const half = size / 2;
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const rgb = (r < half) === (c < half) ? rgb1 : rgb2;
      const o = 4 * (r * size + c);
      for (let k = 0; k < 3; k++) data[o + k] = Math.round(255 * rgb[k]);
      data[o + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.colorSpace = THREE.NoColorSpace;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

export class TowelView extends SceneView {
  constructor(sim) {
    super(sim);
    const { model: m, scene: params } = sim;
    const f = sim.flex;
    const nv = sim.nvert;
    const ne = m.flex_elemnum[f];
    const eadr = m.flex_elemdataadr[f];
    // Wind every triangle so that its normal points up at rest (the front faces up).
    const index = new Uint32Array(3 * ne);
    for (let e = 0; e < ne; e++) {
      const t = [0, 1, 2].map((k) => m.flex_elem[eadr + 3 * e + k]);
      const r = t.map((v) => [sim.rest[3 * v], sim.rest[3 * v + 1]]);
      const nz = (r[1][0] - r[0][0]) * (r[2][1] - r[0][1]) - (r[1][1] - r[0][1]) * (r[2][0] - r[0][0]);
      index.set(nz >= 0 ? t : [t[0], t[2], t[1]], 3 * e);
    }
    this.towelPos = new Float32Array(3 * nv);
    const uv = new Float32Array(2 * nv);
    const tadr = m.flex_texcoordadr[f];
    for (let k = 0; k < nv; k++) {
      uv[2 * k] = tadr >= 0 ? m.flex_texcoord[2 * (tadr + k)] : 0;
      uv[2 * k + 1] = tadr >= 0 ? m.flex_texcoord[2 * (tadr + k) + 1] : 0;
    }
    const geom = new THREE.BufferGeometry();
    geom.setIndex(new THREE.BufferAttribute(index, 1));
    this.towelPosAttr = new THREE.BufferAttribute(this.towelPos, 3);
    this.towelPosAttr.setUsage(THREE.DynamicDrawUsage);
    geom.setAttribute('position', this.towelPosAttr);
    geom.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    this.towelGeom = geom;

    const mat = m.flex_matid[f];
    const texId = m.mat_texid[mat * m.mat_texid.length / m.nmat + 1]; // mjTEXROLE_RGB
    const repeat = [m.mat_texrepeat[2 * mat], m.mat_texrepeat[2 * mat + 1]];
    this.towelTex = checkerTexture(params.towel_rgb, params.towel_rgb2, texId >= 0 ? m.tex_width[texId] : 256);
    const toLight = new THREE.Vector3(...params.light_dir).normalize().negate();
    // front: the shell's top (radius above the mid-surface); back: as keiarm/towel/render.py
    const offset = (side) => (side > 0 ? m.flex_radius[f] : sim.c.back_offset ?? m.flex_radius[f]);
    const material = (side, useMap) => new THREE.ShaderMaterial({
      vertexShader: TEX_VERTEX,
      fragmentShader: TEX_FRAGMENT,
      side: side > 0 ? THREE.FrontSide : THREE.BackSide,
      uniforms: {
        uColor: { value: new THREE.Vector3(...params.towel_back_rgb) },
        uMap: { value: this.towelTex },
        uUseMap: { value: useMap ? 1 : 0 },
        uRepeat: { value: new THREE.Vector2(...repeat) },
        uToLight: { value: toLight },
        uAmbient: { value: params.ambient },
        uDiffuse: { value: params.light_diffuse },
        uOffset: { value: offset(side) },
        uFlat: { value: side > 0 ? 0 : 1 },
        uSide: { value: side },
      },
    });
    this.towelMeshes = [new THREE.Mesh(geom, material(1, true)), new THREE.Mesh(geom, material(-1, false))];
    for (const mesh of this.towelMeshes) {
      mesh.frustumCulled = false;
      this.scene.add(mesh);
    }
    this.sync();
  }

  sync() {
    super.sync();
    if (!this.towelGeom) return; // called by SceneView's constructor before the towel exists
    const x = this.sim.data.flexvert_xpos;
    const a = 3 * this.sim.vadr;
    this.towelPos.set(x.subarray(a, a + this.towelPos.length));
    this.towelPosAttr.needsUpdate = true;
    this.towelGeom.computeVertexNormals();
  }

  // Draw a recorded towel shape (record playback, no physics): verts is a flat Float32Array.
  setTowel(verts) {
    this.towelPos.set(verts);
    this.towelPosAttr.needsUpdate = true;
    this.towelGeom.computeVertexNormals();
  }

  dispose() {
    super.dispose();
    for (const mesh of this.towelMeshes) mesh.material.dispose();
    this.towelGeom.dispose();
    this.towelTex.dispose();
  }
}
