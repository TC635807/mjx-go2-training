#!/usr/bin/env python
"""probe_go2_lidar_occlusion.py —— Go2 倒装 L1 的**自遮挡**估算（纯 CPU MuJoCo，无需 GPU）。

为什么需要它：决策 #4 是"必须依赖雷达"，而雷达装在 Go2 头部（官方 URDF
\`radar_joint: xyz=(0.28945,0,-0.046825), rpy=(0, 2.8782, 0)\` → **倒装**, 传感器 +Z 半球
指向地面、前偏 15.1°）。机器人自己的躯干/大腿会挡住多少格子，只能靠射线投射量化。

做法：
  1. 用 Go2 MJCF（unitree_mujoco）+ PD kp=20/kd=1 静置到真实站姿；
  2. 按 URDF 外参放虚拟 L1，对训练用的 35 点扫描网格（x∈[-0.3,0.9], y∈[-0.4,0.4]）逐个
     从雷达到地面点做射线投射（头部壳体当作孔径排除，射线穿过后继续前进）；
  3. 半球采样 4000 条射线，统计一帧里的地面点数与每格覆盖。

用法: python sim/probe_go2_lidar_occlusion.py
"""
import numpy as np
import mujoco

XML = "/home/tc63/mujoco/mjx-go2-mujoco2ros2/sim_env/models/go2/scene_flat.xml"
HOME = {"FL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5}
LIDAR_XYZ = np.array([0.28945, 0.0, -0.046825])
LIDAR_PITCH = 2.8782          # rad, 官方 URDF radar_joint
N_RAYS = 4000                 # 采样射线数（真机 ≈1964 点/帧 @11Hz）

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
JID = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HOME}
AID = {m.actuator_trnid[a, 0]: a for a in range(m.nu)}
GG = np.array([1, 0, 0, 1, 0, 0], np.uint8)      # group0=地板, group3=机器人碰撞体
BASE = m.body("base_link").id
SHELL = [g for g in range(m.ngeom) if m.geom_bodyid[g] == BASE and m.geom_group[g] == 3
         and m.geom_pos[g][0] > 0.27]            # 头部外壳 = L1 孔径


def settle(pitch=0.0, extra=None, steps=2000):
    d.qpos[:] = 0.0
    d.qpos[2] = 0.45
    d.qpos[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
    tgt = dict(HOME)
    tgt.update(extra or {})
    for n, j in JID.items():
        d.qpos[m.jnt_qposadr[j]] = tgt[n]
    mujoco.mj_forward(m, d)
    for _ in range(steps):
        for n, j in JID.items():
            a = AID[j]
            tau = 20.0 * (tgt[n] - d.qpos[m.jnt_qposadr[j]]) - 1.0 * d.qvel[m.jnt_dofadr[j]]
            lo, hi = m.actuator_ctrlrange[a]
            d.ctrl[a] = np.clip(tau, lo, hi)
        mujoco.mj_step(m, d)


def lidar_pose():
    R = d.xmat[BASE].reshape(3, 3)
    th = LIDAR_PITCH
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    return d.xpos[BASE] + R @ LIDAR_XYZ, R @ (Ry @ np.array([0.0, 0.0, 1.0]))


def trace(origin, target, eps=0.002):
    """沿 origin->target 前进（穿过头部外壳），返回 'ground' / 'body' / 'leg:<name>' / 'none'。"""
    v = target - origin
    v = v / np.linalg.norm(v)
    o = origin.copy()
    for _ in range(20):
        gid = np.array([-1], np.int32)
        dd = mujoco.mj_ray(m, d, o, v, GG, 1, -1, gid)
        if not (dd >= 0.0) or gid[0] < 0:
            return "none"
        if np.linalg.norm(o + v * dd - target) < 0.01:
            return "ground"                      # 命中点就是目标 -> 没被挡
        if gid[0] in SHELL:
            o = o + v * (dd + eps)
            continue
        b = m.body(m.geom_bodyid[gid[0]]).name
        return "body" if b == "base_link" else "leg:" + b
    return "none"


def main():
    from collections import Counter
    gx = np.linspace(-0.3, 0.9, 7)
    gy = np.linspace(-0.4, 0.4, 5)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    XY = np.stack([GX.ravel(), GY.ravel()], -1)          # 35 点，x 外层 y 内层
    print("35 点扫描网格可见性（. = 可见, X = 被机器人自身遮挡）")
    views = {}
    for title, kw in (("A) 名义四足站姿", {}),
                      ("B) 三足(FL 摆动相)", {"extra": {"FL_thigh_joint": 0.3, "FL_calf_joint": -1.0}}),
                      ("C) 躯干前倾 +5°", {"pitch": np.radians(5)}),
                      ("D) 躯干后仰 -5°", {"pitch": np.radians(-5)})):
        settle(**kw)
        p, _ = lidar_pose()
        bx = d.qpos[:2]
        res = [trace(p, np.array([bx[0] + x, bx[1] + y, 0.0])) for x, y in XY]
        vis = np.array([r == "ground" for r in res])
        views[title] = vis
        up = d.xmat[BASE].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        print("\n=== %s  (trunk z=%.3f m, 倾角 %.2f°, 雷达离地 %.3f m)"
              % (title, d.qpos[2], np.degrees(np.arccos(np.clip(up[2], -1, 1))), p[2]))
        for i, x in enumerate(gx):
            print("      x=%+.1f  %s" % (x, " ".join("." if vis[i * 5 + j] else "X" for j in range(5))))
        print("      可见 %2d/35 = %4.1f%%   遮挡来源: %s"
              % (vis.sum(), 100 * vis.mean(), dict(Counter(r for r in res if r != "ground"))))
    V = np.array(list(views.values()))
    print("\n=== 四个姿态全都可见: %d/35   至少一种姿态可见: %d/35"
          % (int(V.all(0).sum()), int(V.any(0).sum())))

    # ---- 半球采样：一帧的点落在哪里 ----
    settle()
    p, a = lidar_pose()
    tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    rng = np.random.default_rng(0)
    u = rng.random(N_RAYS); phi = 2 * np.pi * rng.random(N_RAYS)
    th = np.arccos(u)
    dirs = (np.cos(th)[:, None] * a
            + np.sin(th)[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
    cnt = Counter(); pts = []
    for v in dirs:
        o = p.copy()
        for _ in range(20):
            gid = np.array([-1], np.int32)
            dd = mujoco.mj_ray(m, d, o, v, GG, 1, -1, gid)
            if not (dd >= 0.0) or gid[0] < 0:
                cnt["无回波(>量程/天空)"] += 1; break
            o = o + v * dd
            if gid[0] in SHELL:
                o = o + v * 0.002; continue
            b = m.body(m.geom_bodyid[gid[0]]).name
            if b == "world":
                (cnt.__setitem__("地面", cnt["地面"] + 1) if np.linalg.norm(o) < 8.0
                 else cnt.__setitem__("地面>8m", cnt["地面>8m"] + 1))
                if np.linalg.norm(o) < 8.0:
                    pts.append(o.copy())
            elif b == "base_link":
                cnt["躯干"] += 1
            else:
                cnt["腿(%s)" % b] += 1
            break
    print("\n=== 半球采样 %d 条射线（真机一帧 ≈1964 点）" % N_RAYS)
    print("    ", dict(cnt))
    pts = np.array(pts)
    bx = d.qpos[:2]
    hit = np.zeros(35)
    for q in pts:
        dx, dy = q[0] - bx[0], q[1] - bx[1]
        if gx[0] - 0.1 <= dx <= gx[-1] + 0.1 and gy[0] - 0.1 <= dy <= gy[-1] + 0.1:
            i = int(np.argmin(np.abs(gx - dx))); j = int(np.argmin(np.abs(gy - dy)))
            hit[i * 5 + j] += 1
    print("    一帧内被点到 >=1 次的格子: %d/35   每格平均 %.1f 点/帧" % ((hit > 0).sum(), hit.mean()))
    print("    每格点数(升序): %s" % np.round(np.sort(hit), 0).astype(int).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
