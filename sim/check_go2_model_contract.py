#!/usr/bin/env python
"""Go2 训练模型契约检查 + 沉降测试.

envs/go2_walk.py 按**名字**索引模型里的一堆对象 (site/geom/sensor)。名字错了
不会在加载时炸, 而是在 _get_obs / 奖励里静默取错或抛 KeyError —— 所以把契约
固化成这个脚本, 每次改 XML 都跑一遍。

用法:
  python sim/check_go2_model_contract.py
  python sim/check_go2_model_contract.py --xml models/go2/scene_mjx_terrain.xml
"""
import argparse
import os
import sys

import numpy as np
import mujoco

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FEET = ["FR", "FL", "RR", "RL"]
SENSORS = ["gyro", "local_linvel", "accelerometer", "global_linvel",
           "global_angvel", "upvector"]


def check(xml_path, key_z=0.32):
    print(f"=== 契约检查: {xml_path} ===")
    m = mujoco.MjModel.from_xml_path(xml_path)
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
          f"nsite={m.nsite} nsensor={m.nsensor} nhfield={m.nhfield}")

    fails, warns = [], []

    def has(fn, kind, name, why):
        try:
            fn(name)
            return True
        except Exception as e:
            fails.append(f"{kind} '{name}' 缺失 ({why}): {e}")
            return False

    # ---- 1. 足端 site / geom ----
    for f in FEET:
        has(m.site, "site", f, "env 的 _feet_site_id / {f}_global_linvel")
        has(m.geom, "geom", f, "env 的 _feet_geom_id / {f}_floor_found")
    # ---- 2. 躯干 / imu ----
    has(m.body, "body", "base_link", "env 的 _torso_body_id (Go1 里叫 trunk)")
    has(m.site, "site", "imu", "所有本体感知传感器")
    # ---- 3. 地面 ----
    has(m.geom, "geom", "floor", "足端接触传感器 geom2 / terrain_height")
    # ---- 4. 传感器 ----
    for s in SENSORS:
        has(m.sensor, "sensor", s, "本体系/全局速度、重力方向、角速度")
    for f in FEET:
        has(m.sensor, "sensor", f"{f}_global_linvel", "足端滑移/抬脚奖励")
        has(m.sensor, "sensor", f"{f}_floor_found", "feet_air_time / 触地判定")

    # ---- 5. keyframe ----
    try:
        kf = m.keyframe("home")
        q = np.array(kf.qpos)
        c = np.array(kf.ctrl)
        print(f"keyframe home: z={q[2]:.4f}  joints={np.round(q[7:],3)}")
        print(f"                ctrl  ={np.round(c,3)}")
        if abs(q[2] - key_z) > 1e-6:
            warns.append(f"keyframe z={q[2]:.4f} != 期望 {key_z}")
        if len(q) != 7 + m.nu:
            fails.append(f"keyframe qpos 维度 {len(q)} != 7+nu={7+m.nu}")
        if not np.allclose(q[7:], c, atol=1e-6):
            warns.append("keyframe qpos[7:] != ctrl (obs 的 qpos0 取的是 ctrl)")
    except Exception as e:
        fails.append(f"keyframe 'home' 缺失: {e}")

    # ---- 6. 执行器: 位置伺服, ctrlrange 覆盖关节范围 ----
    print("\n执行器:")
    for i in range(m.nu):
        nm = m.actuator(i).name
        jid = m.actuator_trnid[i, 0]
        jn = m.joint(jid).name
        jr = m.jnt_range[jid]
        cr = m.actuator_ctrlrange[i]
        fr = m.actuator_forcerange[i]
        ok_c = (cr[0] <= jr[0] + 1e-6) and (cr[1] >= jr[1] - 1e-6)
        if not ok_c:
            warns.append(f"actuator {nm} ctrlrange {cr} 未覆盖关节范围 {jr}")
        print(f"  {nm:10s} joint={jn:16s} jnt={np.round(jr,3)} "
              f"ctrl={np.round(cr,3)} forcerange={np.round(fr,2)}")
        if m.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED:
            warns.append(f"actuator {nm} 不是 FIXED 增益 (期望 position 伺服)")
        gap = m.actuator_dyntype[i]
        if gap != mujoco.mjtDyn.mjDYN_NONE:
            warns.append(f"actuator {nm} 有动力学期望 NONE")

    # ---- 7. 足端接触参数 ----
    print("\n足端接触:")
    for f in FEET:
        gid = m.geom(f).id
        print(f"  {f}: condim={m.geom_condim[gid]} solimp={np.round(m.geom_solimp[gid],4)} "
              f"friction={np.round(m.geom_friction[gid],3)} prio={m.geom_priority[gid]}")

    # ---- 8. 沉降测试 ----
    print("\n=== 沉降测试 (home keyframe, 1s) ===")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
    d.ctrl[:] = np.array(m.keyframe("home").ctrl)
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    z = float(d.qpos[2])
    up = 1.0 - 2.0 * (d.qpos[5] ** 2 + d.qpos[6] ** 2)
    tilt = float(np.degrees(np.arccos(np.clip(up, -1, 1))))
    print(f"  躯干 z = {z:.4f} m   倾角 = {tilt:.2f}°   ncon = {d.ncon}   "
          f"|tau|max = {np.abs(d.actuator_force).max():.2f} N·m")
    if not (0.20 < z < 0.35):
        fails.append(f"沉降后 z={z:.4f} 不在 [0.20, 0.35]")
    if tilt > 6.0:
        fails.append(f"沉降后倾角 {tilt:.2f}° 过大")
    if d.ncon != 4:
        warns.append(f"ncon={d.ncon} != 4 (可能不是四足均匀着地)")

    print("\n=== 结论 ===")
    for w in warns:
        print(f"  [warn] {w}")
    for f in fails:
        print(f"  [FAIL] {f}")
    if not fails:
        print("  契约通过。")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(
        _ROOT, "models", "go2", "scene_mjx_position.xml"))
    ap.add_argument("--key_z", type=float, default=0.32)
    args = ap.parse_args()
    return check(args.xml, args.key_z)


if __name__ == "__main__":
    sys.exit(main())
