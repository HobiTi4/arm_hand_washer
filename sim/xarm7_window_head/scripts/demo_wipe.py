#!/usr/bin/env python3
"""
Sanity demo in MuJoCo: bring the squeegee to the glass, press with ~F_DES newtons using
Cartesian impedance control (torque actuators), pull one vertical stroke, lift off.

Prints contact-force statistics, joint-torque load vs torque limits and frame collisions.
    python3 demo_wipe.py                 # headless, prints metrics, saves demo_log.npz
    python3 demo_wipe.py --view          # opens the MuJoCo viewer (needs a display), waits 5 s, then moves
    python3 demo_wipe.py --view --delay 2   # custom wait before the motion starts
"""
import argparse, os, time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "mjcf", "scene_window.xml")
TAU_LIMIT = np.array([80, 80, 60, 48, 42, 24, 18.0])

F_DES = 20.0            # desired normal force [N]
K_N = 1000.0            # normal stiffness [N/m] -> target 20 mm "inside" the glass
K_T = 1500.0            # tangential stiffness [N/m]
K_R = 60.0              # rotational stiffness [Nm/rad]
TILT = np.radians(30)   # tool leans 30 deg: flange above/behind the blade
STROKE_TOP, STROKE_BOTTOM, STROKE_Y = 1.50, 1.10, 0.0   # world z / y of the blade centre
SPEED = 0.10            # stroke speed [m/s]


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def desired_R():
    # blade x along world +y, blade z (out of tool) into the glass (+x), then lean by TILT
    R0 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)   # columns: x_b, y_b, z_b in world
    return R0 @ Rx(TILT)


def rot_err(R_des, R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, (R_des @ R.T).flatten())
    if q[0] < 0: q = -q
    return 2 * q[1:]          # small-angle axis*angle


def ik(m, d, sid, p_des, R_des, q0, iters=300):
    d.qpos[:7] = q0
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        e = np.r_[p_des - d.site_xpos[sid], rot_err(R_des, d.site_xmat[sid].reshape(3, 3))]
        if np.linalg.norm(e) < 1e-5: break
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.r_[jp[:, :7], jr[:, :7]]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        dq += (np.eye(7) - np.linalg.pinv(J) @ J) @ (0.1 * (q0 - d.qpos[:7]))   # stay near seed
        d.qpos[:7] = np.clip(d.qpos[:7] + dq, lo, hi)
    return d.qpos[:7].copy(), np.linalg.norm(e)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--view", action="store_true")
    ap.add_argument("--rigid", action="store_true", help="use the rigid blade holder (shows stick-slip chatter)")
    ap.add_argument("--delay", type=float, default=5.0, help="seconds to hold the start pose in the viewer before moving")
    args = ap.parse_args()
    scene = SCENE.replace("scene_window.xml", "scene_window_rigid.xml") if args.rigid else SCENE
    print("scene:", os.path.basename(scene))
    m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m)
    sid = m.site("blade_tcp").id
    mujoco.mj_forward(m, d)
    glass_x = m.body("window").pos[0]                       # glass surface (world x)
    rubber = m.geom("head_rubber").id; glass = m.geom("glass").id
    frame_ids = {m.geom(n).id for n in ["frame_top", "frame_bottom", "frame_left", "frame_right", "sill"]}
    R_des = desired_R()

    # ---- start: blade 5 cm in front of the glass at the top of the stroke
    seed = np.array([0.0, -0.2, 0.0, 1.3, 0.0, 1.0, 0.0])
    p0 = np.array([glass_x - 0.05, STROKE_Y, STROKE_TOP])
    q_start, err = ik(m, d, sid, p0, R_des, seed)
    print(f"IK start pose error {err:.2e}, q = {np.round(q_start, 3)}")
    d.qpos[:7] = q_start; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    q_null = q_start.copy()

    # ---- phases: (duration, start target, end target, normal offset start, end)
    approach, press, stroke_t, lift = 1.0, 1.0, (STROKE_TOP - STROKE_BOTTOM) / SPEED, 1.0
    t_tot = approach + press + stroke_t + lift
    def target(t):
        n_in = glass_x + F_DES / K_N            # impedance set-point inside the glass
        n_out = glass_x - 0.05
        if t < approach:                         # settle
            return np.array([n_out, STROKE_Y, STROKE_TOP]), False
        t -= approach
        if t < press:                            # move in
            s = t / press; return np.array([n_out + s * (n_in - n_out), STROKE_Y, STROKE_TOP]), False
        t -= press
        if t < stroke_t:                         # pull down
            s = t / stroke_t; return np.array([n_in, STROKE_Y, STROKE_TOP + s * (STROKE_BOTTOM - STROKE_TOP)]), True
        t -= stroke_t
        s = min(1, t / lift); return np.array([n_in + s * (n_out - n_in), STROKE_Y, STROKE_BOTTOM]), False

    Kp = np.diag([K_N, K_T, K_T]); Dp = 2 * np.sqrt(Kp * 3.0)
    Kr, Dr = K_R, 2 * np.sqrt(K_R * 0.05)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    log = {k: [] for k in ["t", "fn", "fn_ft", "pos", "tau", "stroke", "frame_hit"]}
    f6 = np.zeros(6)

    viewer = None
    if args.view:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(m, d)
        # hold the start pose (physics paused) so the scene can be inspected before the motion starts
        t_end = time.time() + args.delay
        while viewer.is_running() and time.time() < t_end:
            print(f"\rstarting in {t_end - time.time():4.1f} s", end="", flush=True)
            viewer.sync(); time.sleep(0.05)
        print("\rstarting now        ")

    while d.time < t_tot:
        p_des, in_stroke = target(d.time)
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.r_[jp[:, :7], jr[:, :7]]
        v = J @ d.qvel[:7]
        p = d.site_xpos[sid]; R = d.site_xmat[sid].reshape(3, 3)
        F = Kp @ (p_des - p) - Dp @ v[:3]
        Mo = Kr * rot_err(R_des, R) - Dr * v[3:]
        tau = J.T @ np.r_[F, Mo] + d.qfrc_bias[:7] + m.dof_damping[:7] * d.qvel[:7]
        Jpinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        tau += (np.eye(7) - J.T @ Jpinv.T) @ (5.0 * (q_null - d.qpos[:7]) - 1.0 * d.qvel[:7])
        d.ctrl[:7] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)
        mujoco.mj_step(m, d)

        fn, hit = 0.0, False
        for i in range(d.ncon):
            c = d.contact[i]; pair = {c.geom1, c.geom2}
            if pair == {rubber, glass}:
                mujoco.mj_contactForce(m, d, i, f6)
                fn += abs(f6[0])
            elif pair & frame_ids:
                hit = True
        # F/T sensor at the flange, expressed in world; component along glass normal (+x)
        R_ft = d.site("ft_sensor").xmat.reshape(3, 3)
        f_world = R_ft @ d.sensor("ft_force").data
        log["t"].append(d.time); log["fn"].append(fn); log["fn_ft"].append(f_world[0])
        log["pos"].append(p.copy()); log["tau"].append(d.qfrc_actuator[:7].copy())
        log["stroke"].append(in_stroke); log["frame_hit"].append(hit)
        if viewer is not None and int(d.time * 1000) % 16 == 0:
            viewer.sync(); time.sleep(0.016)

    L = {k: np.array(v) for k, v in log.items()}
    st = L["stroke"] & (L["t"] > L["t"][L["stroke"]][0] + 0.3)          # skip first 0.3 s of stroke
    fn = L["fn"][st]
    print(f"\nStroke {STROKE_TOP:.2f} -> {STROKE_BOTTOM:.2f} m at {SPEED} m/s, tilt {np.degrees(TILT):.0f} deg, target {F_DES} N")
    print(f"  contact normal force: mean {fn.mean():.1f} N, std {fn.std():.2f} N, min {fn.min():.1f}, max {fn.max():.1f}")
    print(f"  contact lost in stroke: {np.mean(fn < 1.0) * 100:.1f}% of samples")
    print(f"  blade x deviation from glass plane: {np.ptp(L['pos'][st, 0]) * 1000:.1f} mm peak-to-peak")
    tau_ratio = np.max(np.abs(L["tau"][L["stroke"]]), axis=0) / TAU_LIMIT
    print("  peak |tau| / torque limit during stroke: " + "  ".join(f"j{i+1} {r*100:3.0f}%" for i, r in enumerate(tau_ratio)))
    print(f"  frame/sill contacts: {int(L['frame_hit'].sum())} samples")
    np.savez(os.path.join(HERE, "demo_log.npz"), **L)
    if viewer is not None: viewer.close()


if __name__ == "__main__":
    main()
