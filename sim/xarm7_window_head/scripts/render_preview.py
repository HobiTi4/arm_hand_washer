#!/usr/bin/env python3
"""Render a static preview PNG of the scene without OpenGL (matplotlib).
Poses the arm with the blade on the glass, mid-stroke.  python3 render_preview.py"""
import os, importlib.util
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dw", os.path.join(HERE, "demo_wipe.py"))
dw = importlib.util.module_from_spec(spec); spec.loader.exec_module(dw)

m = mujoco.MjModel.from_xml_path(dw.SCENE); d = mujoco.MjData(m)
sid = m.site("blade_tcp").id; gx = m.body("window").pos[0]
q, _ = dw.ik(m, d, sid, np.array([gx, 0.0, 1.30]), dw.desired_R(), np.array([0, -0.2, 0, 1.3, 0, 1.0, 0]))
d.qpos[:7] = q; mujoco.mj_forward(m, d)

def box_faces(size, pos, R):
    s = np.array(size)
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
    v = c @ R.T + pos
    idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [v[i] for i in idx]

def cyl_faces(r, hl, pos, R, n=20):
    a = np.linspace(0, 2*np.pi, n+1)
    ring = np.c_[r*np.cos(a), r*np.sin(a)]
    top = np.c_[ring, np.full(n+1, hl)] @ R.T + pos
    bot = np.c_[ring, np.full(n+1, -hl)] @ R.T + pos
    return [top, bot] + [np.array([top[i], top[i+1], bot[i+1], bot[i]]) for i in range(n)]

fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(111, projection="3d")
for g in range(m.ngeom):
    t = m.geom_type[g]; pos = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3)
    rgba = m.geom_rgba[g].copy(); name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]; va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = m.mesh_vert[va:va+vn] @ R.T + pos; F = m.mesh_face[fa:fa+fn]
        F = F[::3]                                           # decimate for speed
        T = V[F]; n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        lam = 0.45 + 0.55 * np.abs(n @ np.array([-0.4, -0.6, 0.7]) / 1.0)
        cols = np.c_[np.outer(lam, [0.88, 0.88, 0.92]), np.ones(len(T))]
        ax.add_collection3d(Poly3DCollection(T, facecolors=cols, linewidths=0))
    elif m.geom_group[g] == 3 or name in ("floor",):
        continue                                             # skip arm collision proxies and floor
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        if name == "stand": rgba[3] = 0.5
        ax.add_collection3d(Poly3DCollection(box_faces(m.geom_size[g], pos, R), facecolor=rgba[:3], alpha=max(rgba[3], 0.25), edgecolor=(0, 0, 0, 0.15), linewidths=0.3))
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        ax.add_collection3d(Poly3DCollection(cyl_faces(m.geom_size[g][0], m.geom_size[g][1], pos, R), facecolor=rgba[:3], alpha=rgba[3], edgecolor="none"))
# stroke path and spray direction
ax.plot([gx - 0.002]*2, [0, 0], [1.50, 1.10], color="tab:green", lw=2, label="stroke path")
p = d.site("nozzle_tip").xpos; z = d.site("nozzle_tip").xmat.reshape(3, 3)[:, 2]
ax.quiver(*p, *(z*0.15), color="tab:cyan", lw=2, label="spray axis")
ax.set_xlim(-0.35, 0.75); ax.set_ylim(-0.55, 0.55); ax.set_zlim(0.75, 2.05)
ax.set_box_aspect((1.1, 1.1, 1.3)); ax.view_init(elev=18, azim=-128)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_title("xArm7 + squeegee/spray head (no gripper), blade on glass at z = 1.30 m")
ax.legend(loc="upper left")
out = os.path.join(HERE, "..", "preview.png"); plt.tight_layout(); plt.savefig(out, dpi=110)
print("saved", os.path.abspath(out))
