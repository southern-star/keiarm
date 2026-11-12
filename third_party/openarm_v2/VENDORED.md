# Vendored from enactic/openarm_mujoco

- Source: https://github.com/enactic/openarm_mujoco (`v2/`)
- Commit: 1daab06716b779376dbe7f01fb0548f8b539bb35 (2026-11-09)
- License: Apache-2.0 (see `LICENSE` in this folder). Copyright 2026 Enactic, Inc.
- Contents: `v2/openarm_bimanual.xml` and the meshes it references (`assets/visual/{arm,gripper}`, `assets/collision`), plus the pedestal collision mesh `assets/collision/body_link0_symp.stl` (from `v2/pedestal/openarm_pedestal.xml`). Other scenes, the cell and the large pedestal visual mesh are not copied.
- Local modifications: none. Do not edit files here; `keiarm/openarm/` builds scenes around `openarm_bimanual.xml` with `MjSpec` (timestep, gravity compensation, materials).
