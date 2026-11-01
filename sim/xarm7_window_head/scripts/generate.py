#!/usr/bin/env python3
"""
Generate URDF + MuJoCo MJCF for: UFactory xArm7 (no gripper) + window-cleaning head
(squeegee + spray nozzle + RealSense) bolted directly to the flange.

Arm kinematics / inertias come from bcr_description (arm_subtree.urdf, jinxs).
Everything below the flange (gripper, its camera bracket, key points) is dropped.

All head dimensions are PLACEHOLDERS for a first prototype built around a ~25 cm
household squeegee with the handle cut off. Edit HEAD below and re-run:
    python3 generate.py
Outputs:
    ../urdf/xarm7_window_head.urdf
    ../mjcf/xarm7_window_head.xml   (robot only, base at origin)
    ../mjcf/scene_window.xml        (robot on a stand in front of a window)
"""
import os, math
import numpy as np
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_source_arm_subtree.urdf")
OUT = os.path.dirname(HERE)

# --------------------------------------------------------------------------------------
# Head parameters (flange frame: z points out of the flange, x is along the blade)
# --------------------------------------------------------------------------------------
HEAD = dict(
    head_length=0.25,      # flange -> rubber edge [m]
    blade_width=0.25,      # rubber length [m]
    blade_tilt_deg=0.0,    # built-in tilt of blade holder about x (0 = straight; arm tilts instead)
    # masses [kg] (estimates for PETG adapter + cut household squeegee)
    m_adapter=0.06, m_stem=0.08, m_holder=0.12, m_rubber=0.02, m_nozzle=0.03, m_camera=0.072,
    # nozzle: on the side of the stem, flat-fan, fan plane along x (blade direction)
    nozzle_pos=(0.0, 0.035, 0.12), nozzle_tilt_deg=0.0, spray_fan_deg=80.0,
    # RealSense D4xx-like body on the other side, looking along +z
    camera_pos=(0.0, -0.045, 0.06),
    # passive compliance of the blade holder (MJCF only): spring-loaded slide along tool z.
    # In simulation a rigid holder chatters (stick-slip) while sliding; the spring removes it.
    compliance=True, compliance_stiffness=4000.0, compliance_damping=20.0, compliance_range=0.010,
)
# Torque limits actually used in bcr_core (ArmModel::ArmTorqueLimit), not the URDF efforts
TAU_LIMIT = [80, 80, 60, 48, 42, 24, 18]
# Camera intrinsics taken from bcr_description (arm_color_camera / arm_depth_camera)
COLOR_FY, COLOR_H = 907.18, 720
DEPTH_FY, DEPTH_H = 385.68, 480

# Scene parameters (world frame, metres)
SCENE = dict(
    base_pos=(0.0, 0.0, 1.00),   # xArm link_base on top of a stand / robot column
    glass_x=0.55,                # distance from base axis to glass surface
    glass_w=0.80, glass_h=1.00,  # visible glass
    glass_bottom=0.95,           # glass bottom edge height above floor
    frame_w=0.05, frame_depth=0.06,
)

# --------------------------------------------------------------------------------------
def rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])

def R2quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([w, (R[2, 1]-R[1, 2])/(4*w), (R[0, 2]-R[2, 0])/(4*w), (R[1, 0]-R[0, 1])/(4*w)])
    i = int(np.argmax(np.diag(R)))
    j, k = (i+1) % 3, (i+2) % 3
    s = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4); q[i+1] = s/4; q[0] = (R[k, j]-R[j, k])/s
    q[j+1] = (R[j, i]+R[i, j])/s; q[k+1] = (R[k, i]+R[i, k])/s
    return q

def fmt(v, n=6):
    return " ".join(f"{float(x):.{n}g}" for x in v)

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

# --------------------------------------------------------------------------------------
# Head geometry as primitives: (name, type, size, pos, R, mass, rgba, part)
#   box size = full extents; cylinder size = (radius, length), axis along local z
#   part: 'fixed' (rigid to flange) or 'blade' (moves with optional compliance joint)
# --------------------------------------------------------------------------------------
def head_parts(h=HEAD):
    L = h["head_length"]; W = h["blade_width"]
    rub_out, rub_h = 0.012, 0.022            # rubber sticks out 12 mm below holder
    hold_h, hold_t = 0.045, 0.026
    hold_bottom = L - rub_out                # holder lower face
    stem_top = 0.012
    stem_bottom = hold_bottom - hold_h + 0.005
    tilt = Rx(math.radians(h["blade_tilt_deg"]))
    pivot = np.array([0, 0, stem_bottom])
    def blade_pose(p):                       # apply built-in tilt about pivot
        return pivot + tilt @ (np.array(p) - pivot), tilt
    parts = []
    I3 = np.eye(3)
    parts.append(("adapter", "cylinder", (0.0375, 0.012), np.array([0, 0, 0.006]), I3, h["m_adapter"], (0.85, 0.55, 0.15, 1), "fixed"))
    parts.append(("stem", "box", (0.035, 0.025, stem_bottom - stem_top), np.array([0, 0, (stem_top+stem_bottom)/2]), I3, h["m_stem"], (0.85, 0.55, 0.15, 1), "fixed"))
    p, R = blade_pose([0, 0, hold_bottom - hold_h/2])
    parts.append(("holder", "box", (W + 0.01, hold_t, hold_h), p, R, h["m_holder"], (0.08, 0.08, 0.1, 1), "blade"))
    p, R = blade_pose([0, 0, L - rub_h/2])
    parts.append(("rubber", "box", (W, 0.004, rub_h), p, R, h["m_rubber"], (0.15, 0.25, 0.85, 1), "blade"))
    npos = np.array(h["nozzle_pos"]); nR = Rx(-math.radians(h["nozzle_tilt_deg"]))
    parts.append(("nozzle", "cylinder", (0.008, 0.03), npos, nR, h["m_nozzle"], (0.7, 0.7, 0.75, 1), "fixed"))
    parts.append(("nozzle_bracket", "box", (0.02, 0.025, 0.02), np.array([0, 0.0225, npos[2]]), I3, 0.0, (0.85, 0.55, 0.15, 1), "fixed"))
    parts.append(("camera", "box", (0.09, 0.025, 0.025), np.array(h["camera_pos"]), I3, h["m_camera"], (0.2, 0.2, 0.22, 1), "fixed"))
    # frames
    tcp_p, tcp_R = blade_pose([0, 0, L])
    nozzle_tip = npos + nR @ np.array([0, 0, 0.015])
    frames = dict(
        blade_tcp=(tcp_p, tcp_R),                     # centre of rubber edge; x along blade, z out of tool (into glass)
        nozzle_tip=(nozzle_tip, nR),                  # z = spray direction, x = fan width direction
        head_camera=(np.array(h["camera_pos"]) + np.array([0, 0, 0.0125]), I3),  # optical axis +z
    )
    return parts, frames

def box_inertia(m, s):
    a, b, c = s
    return m/12*np.diag([b*b+c*c, a*a+c*c, a*a+b*b])

def cyl_inertia(m, r, l):
    return np.diag([m*(3*r*r+l*l)/12, m*(3*r*r+l*l)/12, m*r*r/2])

def combine(parts):
    M = sum(p[5] for p in parts)
    com = sum(p[5]*p[3] for p in parts) / M
    I = np.zeros((3, 3))
    for name, t, s, pos, R, m, *_ in parts:
        if m <= 0: continue
        Ic = box_inertia(m, s) if t == "box" else cyl_inertia(m, *s)
        Ic = R @ Ic @ R.T
        d = pos - com
        I += Ic + m*(np.dot(d, d)*np.eye(3) - np.outer(d, d))
    return M, com, I

# --------------------------------------------------------------------------------------
# Read arm chain from the bcr URDF (link_base .. link7)
# --------------------------------------------------------------------------------------
def read_arm():
    root = ET.parse(SRC).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = {j.get("name"): j for j in root.findall("joint")}
    return root, links, joints

ARM_LINKS = ["link_base"] + [f"link{i}" for i in range(1, 8)]
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

def origin(e):
    o = e.find("origin") if e is not None else None
    if o is None: return np.zeros(3), np.zeros(3)
    return (np.array([float(v) for v in o.get("xyz", "0 0 0").split()]),
            np.array([float(v) for v in o.get("rpy", "0 0 0").split()]))

# --------------------------------------------------------------------------------------
def write_urdf():
    root, links, joints = read_arm()
    robot = ET.Element("robot", name="xarm7_window_head")
    ET.SubElement(robot, "material", name="head_orange").append(ET.Element("color", rgba="0.85 0.55 0.15 1"))
    # the vendor arm visuals reference these by name only; declare them so the URDF is standalone
    ET.SubElement(robot, "material", name="White").append(ET.Element("color", rgba="1 1 1 1"))
    ET.SubElement(robot, "material", name="Silver").append(ET.Element("color", rgba="0.753 0.753 0.753 1"))
    for ln in ARM_LINKS:
        l = ET.fromstring(ET.tostring(links[ln]))
        for m in l.iter("mesh"):
            m.set("filename", "../meshes/xarm7/visual/" + os.path.basename(m.get("filename")))
        robot.append(l)
    for jn in ARM_JOINTS:
        j = ET.fromstring(ET.tostring(joints[jn]))
        lim = j.find("limit"); lim.set("effort", str(TAU_LIMIT[int(jn[-1]) - 1]))
        robot.append(j)
    ET.SubElement(robot, "link", name="link_eef")
    j = ET.SubElement(robot, "joint", name="joint_eef", type="fixed")
    ET.SubElement(j, "parent", link="link7"); ET.SubElement(j, "child", link="link_eef")
    ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")

    parts, frames = head_parts()
    M, com, I = combine(parts)
    head = ET.SubElement(robot, "link", name="window_head")
    inert = ET.SubElement(head, "inertial")
    ET.SubElement(inert, "origin", xyz=fmt(com), rpy="0 0 0")
    ET.SubElement(inert, "mass", value=f"{M:.4f}")
    ET.SubElement(inert, "inertia", ixx=f"{I[0,0]:.6e}", ixy=f"{I[0,1]:.6e}", ixz=f"{I[0,2]:.6e}",
                  iyy=f"{I[1,1]:.6e}", iyz=f"{I[1,2]:.6e}", izz=f"{I[2,2]:.6e}")
    for name, t, s, pos, R, m, rgba, _ in parts:
        r, p, y = R_to_rpy(R)
        for tag in ("visual", "collision"):
            e = ET.SubElement(head, tag, name=f"{name}_{tag}")
            ET.SubElement(e, "origin", xyz=fmt(pos), rpy=fmt((r, p, y)))
            g = ET.SubElement(e, "geometry")
            if t == "box": ET.SubElement(g, "box", size=fmt(s))
            else: ET.SubElement(g, "cylinder", radius=f"{s[0]}", length=f"{s[1]}")
            if tag == "visual":
                mat = ET.SubElement(e, "material", name=f"{name}_mat")
                ET.SubElement(mat, "color", rgba=fmt(rgba))
    j = ET.SubElement(robot, "joint", name="window_head_joint", type="fixed")
    ET.SubElement(j, "parent", link="link_eef"); ET.SubElement(j, "child", link="window_head")
    ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
    for fname, (p, R) in frames.items():
        ET.SubElement(robot, "link", name=fname)
        j = ET.SubElement(robot, "joint", name=f"{fname}_joint", type="fixed")
        ET.SubElement(j, "parent", link="window_head"); ET.SubElement(j, "child", link=fname)
        ET.SubElement(j, "origin", xyz=fmt(p), rpy=fmt(R_to_rpy(R)))
    ET.indent(robot)
    path = os.path.join(OUT, "urdf", "xarm7_window_head.urdf")
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<!-- Generated by scripts/generate.py: xArm7 (bcr_description) without gripper,\n'
                '     window-cleaning head (squeegee + spray nozzle + camera) on the flange.\n'
                '     Head inertia is an estimate from primitives. Effort = bcr_core ArmTorqueLimit. -->\n')
        f.write(ET.tostring(robot, encoding="unicode"))
    return path, M, com, I

def R_to_rpy(R):
    p = -math.asin(max(-1, min(1, R[2, 0])))
    if abs(math.cos(p)) > 1e-6:
        r = math.atan2(R[2, 1], R[2, 2]); y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = math.atan2(-R[1, 2], R[1, 1]); y = 0.0
    return (r, p, y)

# --------------------------------------------------------------------------------------
def mj_geom(parent, t, s, pos, R, rgba, name=None, **kw):
    a = dict(type=t, pos=fmt(pos), quat=fmt(R2quat(R)), rgba=fmt(rgba))
    a["size"] = fmt(np.array(s)/2) if t == "box" else f"{s[0]:.6g} {s[1]/2:.6g}"
    if name: a["name"] = name
    a.update(kw)
    return ET.SubElement(parent, "geom", **a)

def build_robot(parent_body, assets_added, h=HEAD):
    root, links, joints = read_arm()
    def add_link(body, ln):
        l = links[ln]
        inn = l.find("inertial")
        if inn is not None:
            p, rpy = origin(inn); i = inn.find("inertia")
            g = lambda k: float(i.get(k))
            ET.SubElement(body, "inertial", pos=fmt(p), mass=inn.find("mass").get("value"),
                          fullinertia=fmt([g("ixx"), g("iyy"), g("izz"), g("ixy"), g("ixz"), g("iyz")]))
        for v in l.findall("visual"):
            p, rpy = origin(v); mesh = os.path.splitext(os.path.basename(v.find("geometry/mesh").get("filename")))[0]
            ET.SubElement(body, "geom", type="mesh", mesh=mesh, pos=fmt(p), quat=fmt(R2quat(rpy2R(*rpy))),
                          **{"class": "visual"})
        for c in l.findall("collision"):
            p, rpy = origin(c); g = c.find("geometry")[0]
            if g.tag == "cylinder":
                s = (float(g.get("radius")), float(g.get("length")))
                mj_geom(body, "cylinder", s, p, rpy2R(*rpy), (0.5, 0.5, 0.5, 0.0), **{"class": "arm_col"})
            elif g.tag == "box":
                s = [float(x) for x in g.get("size").split()]
                mj_geom(body, "box", s, p, rpy2R(*rpy), (0.5, 0.5, 0.5, 0.0), **{"class": "arm_col"})
            elif g.tag == "sphere":
                ET.SubElement(body, "geom", type="sphere", size=g.get("radius"), pos=fmt(p), **{"class": "arm_col"})
    base = ET.SubElement(parent_body, "body", name="link_base")
    add_link(base, "link_base")
    body = base
    for i, jn in enumerate(ARM_JOINTS):
        j = joints[jn]; p, rpy = origin(j)
        child = ET.SubElement(body, "body", name=j.find("child").get("link"), pos=fmt(p), quat=fmt(R2quat(rpy2R(*rpy))))
        lim = j.find("limit"); dyn = j.find("dynamics")
        ET.SubElement(child, "joint", name=jn, axis="0 0 1", range=f"{lim.get('lower')} {lim.get('upper')}",
                      damping=dyn.get("damping") if dyn is not None else "1", armature="0.05")
        add_link(child, j.find("child").get("link"))
        body = child
    # ---- head on the flange (link7) ----
    parts, frames = head_parts(h)
    head = ET.SubElement(body, "body", name="window_head")
    ET.SubElement(head, "site", name="ft_sensor", pos="0 0 0", size="0.01", rgba="1 0 0 0.3")
    fixed = [p for p in parts if p[7] == "fixed"]; blade = [p for p in parts if p[7] == "blade"]
    M, com, I = combine(fixed)
    ET.SubElement(head, "inertial", pos=fmt(com), mass=f"{M:.4f}",
                  fullinertia=fmt([I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]]))
    for name, t, s, pos, R, m, rgba, _ in fixed:
        mj_geom(head, t, s, pos, R, rgba, name=f"head_{name}", **{"class": "head_col"})
    bl = ET.SubElement(head, "body", name="blade_holder")
    if h["compliance"]:
        # slide along the blade normal (the tilted z of the blade frame), not the head-frame z
        ET.SubElement(bl, "joint", name="blade_compliance", type="slide", axis=fmt(frames["blade_tcp"][1][:, 2] + 0.0),
                      range=f"{-h['compliance_range']} {h['compliance_range']}",
                      stiffness=str(h["compliance_stiffness"]), damping=str(h["compliance_damping"]), springref="0")
    Mb, comb, Ib = combine(blade)
    ET.SubElement(bl, "inertial", pos=fmt(comb), mass=f"{Mb:.4f}",
                  fullinertia=fmt([Ib[0,0], Ib[1,1], Ib[2,2], Ib[0,1], Ib[0,2], Ib[1,2]]))
    for name, t, s, pos, R, m, rgba, _ in blade:
        kw = {"class": "rubber"} if name == "rubber" else {"class": "head_col"}
        mj_geom(bl, t, s, pos, R, rgba, name=f"head_{name}", **kw)
    p, R = frames["blade_tcp"]
    ET.SubElement(bl, "site", name="blade_tcp", pos=fmt(p), quat=fmt(R2quat(R)), size="0.006", rgba="0 1 0 1")
    p, R = frames["nozzle_tip"]
    ET.SubElement(head, "site", name="nozzle_tip", pos=fmt(p), quat=fmt(R2quat(R)), size="0.005", rgba="0 0.8 1 1")
    p, R = frames["head_camera"]
    # MuJoCo cameras look along their -z; flip about x so they look along the tool's +z
    Rc = R @ Rx(math.pi)
    ET.SubElement(head, "camera", name="head_color", pos=fmt(p + np.array([-0.0365 + 0.045, 0, 0])),
                  quat=fmt(R2quat(Rc)), fovy=f"{math.degrees(2*math.atan(COLOR_H/2/COLOR_FY)):.2f}")
    ET.SubElement(head, "camera", name="head_depth", pos=fmt(p + np.array([-0.0215 + 0.045, 0, 0])),
                  quat=fmt(R2quat(Rc)), fovy=f"{math.degrees(2*math.atan(DEPTH_H/2/DEPTH_FY)):.2f}")

def mj_root(name, h=HEAD):
    mj = ET.Element("mujoco", model=name)
    ET.SubElement(mj, "compiler", angle="radian", meshdir="../meshes/xarm7/visual", autolimits="true",
                  balanceinertia="true")  # vendor link4 inertia violates A+B>=C slightly
    ET.SubElement(mj, "option", timestep="0.001", integrator="implicitfast", cone="elliptic")
    d = ET.SubElement(mj, "default")
    dv = ET.SubElement(d, "default", **{"class": "visual"}); ET.SubElement(dv, "geom", contype="0", conaffinity="0", group="2", rgba="0.9 0.9 0.92 1")
    # robot geoms collide with the world, not with each other (self-collision is left to the planner)
    da = ET.SubElement(d, "default", **{"class": "arm_col"}); ET.SubElement(da, "geom", contype="1", conaffinity="0", group="3")
    dh = ET.SubElement(d, "default", **{"class": "head_col"}); ET.SubElement(dh, "geom", contype="1", conaffinity="0", group="1", friction="0.6 0.005 0.0001")
    # rubber: soft contact (time-const 10 ms, damping ratio 1) and low friction (wet glass)
    dr = ET.SubElement(d, "default", **{"class": "rubber"}); ET.SubElement(dr, "geom", contype="1", conaffinity="0", group="1",
                                                                             friction="0.35 0.005 0.0001", solref="0.04 1", solimp="0.9 0.95 0.002", condim="3")
    dw = ET.SubElement(d, "default", **{"class": "world"}); ET.SubElement(dw, "geom", contype="0", conaffinity="1", group="0")
    asset = ET.SubElement(mj, "asset")
    for f in sorted(os.listdir(os.path.join(OUT, "meshes/xarm7/visual"))):
        ET.SubElement(asset, "mesh", name=os.path.splitext(f)[0], file=f)
    return mj, asset

def add_actuators_sensors(mj):
    act = ET.SubElement(mj, "actuator")
    for i, jn in enumerate(ARM_JOINTS):
        ET.SubElement(act, "motor", name=f"tau{i+1}", joint=jn, ctrlrange=f"{-TAU_LIMIT[i]} {TAU_LIMIT[i]}")
    sen = ET.SubElement(mj, "sensor")
    ET.SubElement(sen, "force", name="ft_force", site="ft_sensor")
    ET.SubElement(sen, "torque", name="ft_torque", site="ft_sensor")
    ET.SubElement(sen, "framepos", name="blade_pos", objtype="site", objname="blade_tcp")

def write_mjcf_robot():
    mj, _ = mj_root("xarm7_window_head")
    wb = ET.SubElement(mj, "worldbody")
    build_robot(wb, None)
    add_actuators_sensors(mj)
    ET.indent(mj)
    path = os.path.join(OUT, "mjcf", "xarm7_window_head.xml")
    open(path, "w").write("<!-- Generated by scripts/generate.py -->\n" + ET.tostring(mj, encoding="unicode"))
    return path

def write_mjcf_scene(s=SCENE, h=HEAD, fname="scene_window.xml"):
    mj, asset = mj_root("xarm7_window_scene")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.25 0.27 0.3", rgb2="0.3 0.32 0.35", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="8 8")
    ET.SubElement(mj, "visual").append(ET.Element("headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3"))
    wb = ET.SubElement(mj, "worldbody")
    ET.SubElement(wb, "light", pos="0 -1 3", dir="0.2 0.4 -1", directional="true")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="3 3 0.1", material="grid", **{"class": "world"})
    bx, by, bz = s["base_pos"]
    ET.SubElement(wb, "geom", name="stand", type="box", size=f"0.15 0.15 {bz/2:.4f}", pos=f"{bx} {by} {bz/2:.4f}", rgba="0.35 0.35 0.4 1", **{"class": "world"})
    # window: glass surface at x = glass_x (facing the robot), frame around it, sill below
    gx, gw, gh, gb, fw, fd = s["glass_x"], s["glass_w"], s["glass_h"], s["glass_bottom"], s["frame_w"], s["frame_depth"]
    t = 0.006; zc = gb + gh/2
    win = ET.SubElement(wb, "body", name="window", pos=f"{bx + gx:.4f} {by} 0")
    ET.SubElement(win, "geom", name="glass", type="box", size=f"{t/2} {gw/2} {gh/2}", pos=f"{t/2} 0 {zc}",
                  rgba="0.6 0.8 0.95 0.35", friction="0.35 0.005 0.0001", **{"class": "world"})
    # frame protrudes 15 mm towards the robot (realistic lip the blade must not hit)
    lip = 0.015
    for nm, size, pos in [
        ("frame_top", (fd/2, gw/2 + fw, fw/2), (-lip + fd/2, 0, gb + gh + fw/2)),
        ("frame_bottom", (fd/2, gw/2 + fw, fw/2), (-lip + fd/2, 0, gb - fw/2)),
        ("frame_left", (fd/2, fw/2, gh/2), (-lip + fd/2, gw/2 + fw/2, zc)),
        ("frame_right", (fd/2, fw/2, gh/2), (-lip + fd/2, -gw/2 - fw/2, zc))]:
        ET.SubElement(win, "geom", name=nm, type="box", size=fmt(size), pos=fmt(pos), rgba="0.95 0.95 0.95 1", **{"class": "world"})
    ET.SubElement(win, "geom", name="sill", type="box", size=f"0.12 {gw/2 + fw + 0.05} 0.015",
                  pos=f"{-lip - 0.12 + fd:.4f} 0 {gb - fw - 0.015:.4f}", rgba="0.85 0.85 0.85 1", **{"class": "world"})
    ET.SubElement(win, "site", name="glass_center", pos=f"0 0 {zc}", size="0.01", rgba="1 1 0 1")
    mount = ET.SubElement(wb, "body", name="arm_mount", pos=f"{bx} {by} {bz}")
    build_robot(mount, None, h)
    add_actuators_sensors(mj)
    ET.indent(mj)
    path = os.path.join(OUT, "mjcf", fname)
    open(path, "w").write("<!-- Generated by scripts/generate.py -->\n" + ET.tostring(mj, encoding="unicode"))
    return path

if __name__ == "__main__":
    u, M, com, I = write_urdf()
    r = write_mjcf_robot()
    sc = write_mjcf_scene()
    sc2 = write_mjcf_scene(h=dict(HEAD, compliance=False), fname="scene_window_rigid.xml")
    print("wrote", u, r, sc, sc2, sep="\n  ")
    print(f"head mass {M:.3f} kg, CoM {fmt(com, 4)} m (flange frame)")
    print("head inertia about CoM [kg m^2]:\n", np.array2string(I, precision=6))
