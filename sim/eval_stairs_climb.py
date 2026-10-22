#!/usr/bin/env python
"""台阶爬升能力评测 (warp 训练物理): 不只存活, 还量"实际爬了多少".

动机: probe_sweep_params 只给存活/vx/z, 而 spawn 在楼梯口时"站着不动不摔"也能
100% 存活 -> 台阶能力被掩盖。本脚本记录每个 env 的**最大爬升高度**
(max base_z - 初始 base_z) 与最终高度, 输出爬升分位数与达到各高度的比例。

用法 (GPU 单作业, gpu_guard):
  python sim/eval_stairs_climb.py --policy models/go2_v28c_cons_logs/checkpoints/best \
    --terrain_xml models/go2/scene_mjx_terrain_bands1.xml \
    --boxgrid_json models/go2/assets/bands1.json --spawn_stairs_prob 0.8 \
    --envs 128 --steps 300 --out logs/stairs_climb_bands1.tsv
输出 TSV: policy terrain surv% climb_mean/p50/p90 climb_gt{0.10,0.20,0.30}%
"""
import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.go2_walk import Go2Walk, default_config  # noqa: E402
from common import load_policy  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--terrain_xml", required=True)
    ap.add_argument("--boxgrid_json", default="")
    ap.add_argument("--contact_mode", default="clearance")
    ap.add_argument("--spawn_stairs_prob", type=float, default=0.8)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--cmd", type=float, nargs=3, default=[0.4, 0.0, 0.0])
    ap.add_argument("--scan_nx", type=int, default=17)
    ap.add_argument("--scan_ny", type=int, default=11)
    ap.add_argument("--scan_x", type=float, nargs=2, default=[-0.4, 1.2])
    ap.add_argument("--scan_y", type=float, nargs=2, default=[-0.5, 0.5])
    ap.add_argument("--lidar", action="store_true")
    ap.add_argument("--lidar_occ_recall", type=float, default=None)
    ap.add_argument("--lidar_dropout", type=float, default=None)
    ap.add_argument("--lidar_mount_jitter", type=float, default=None)
    ap.add_argument("--lidar_vfov_jitter", type=float, default=None)
    ap.add_argument("--lidar_sector_prob", type=float, default=None)
    ap.add_argument("--spawn_ring", action="store_true")
    ap.add_argument("--riser", type=float, default=0.0,
                    help="单级 riser 高度 (m); >0 时统计上 1 级 / 2 级比例")
    ap.add_argument("--top-height", type=float, default=0.0,
                    help="楼梯总高 (m); >0 时统计到顶比例 (climb >= 0.8*top)")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    cfg = default_config()
    cfg.terrain = True
    cfg.terrain_xml = args.terrain_xml
    cfg.terrain_boxgrid_json = args.boxgrid_json
    cfg.contact_mode = args.contact_mode
    cfg.random_init.enable = True
    cfg.random_init.spawn_ring = args.spawn_ring
    cfg.random_init.spawn_stairs_prob = float(args.spawn_stairs_prob)
    cfg.height_scan.enable = True
    cfg.height_scan.nx = args.scan_nx
    cfg.height_scan.ny = args.scan_ny
    cfg.height_scan.x_min, cfg.height_scan.x_max = args.scan_x
    cfg.height_scan.y_min, cfg.height_scan.y_max = args.scan_y
    if args.lidar:
        ld = cfg.height_scan.lidar
        ld.enable = True
        ld.occ_recall = 0.5
        ld.dropout = 0.05
        ld.mount_jitter = [0.02] * 3
        ld.vfov_jitter = 0.15
        ld.sector_prob = 0.25
        if args.lidar_occ_recall is not None:
            ld.occ_recall = args.lidar_occ_recall
        if args.lidar_dropout is not None:
            ld.dropout = args.lidar_dropout
        if args.lidar_mount_jitter is not None:
            ld.mount_jitter = [args.lidar_mount_jitter] * 3
        if args.lidar_vfov_jitter is not None:
            ld.vfov_jitter = args.lidar_vfov_jitter
        if args.lidar_sector_prob is not None:
            ld.sector_prob = args.lidar_sector_prob
    cfg.foot_soft_contact = False
    cfg.reward_config.scales.feet_phase = 2.0
    cfg.reward_config.gait_swing_height = 0.15
    cfg.command_config.sample = False
    cfg.command_config.fixed = [float(v) for v in args.cmd]
    cfg.healthy_z_ref = "feet"
    env = Go2Walk(cfg)
    infer, ns = load_policy(os.path.abspath(args.policy))
    print(f"env ok, obs={env.observation_size}, policy obs dim={ns}, "
          f"envs={args.envs} steps={args.steps}", flush=True)

    rngs = jax.random.split(jax.random.PRNGKey(0), args.envs)

    def one(rng):
        st = env.reset(rng)
        z0 = st.data.qpos[2]

        def body(carry, _):
            st, alive, zmax, vx_sum = carry
            act, _ = infer(st.obs, rng)
            st2 = env.step(st, act)
            z = st2.data.qpos[2]
            new_alive = alive & (~st2.done.astype(bool))
            R = st2.data.site_xmat[env._imu_site_id].reshape(3, 3)
            vx = (R.T @ st2.data.qvel[:3])[0]
            return (st2, new_alive,
                    jp.where(new_alive, jp.maximum(zmax, z), zmax),
                    vx_sum + jp.where(new_alive, vx, 0.0)), None

        (stf, alive, zmax, vx_sum), _ = jax.lax.scan(
            body, (st, jp.array(True), z0, jp.zeros(())), None, length=args.steps)
        return alive, zmax - z0, vx_sum, stf.data.qpos

    alive, climb, vx, qpos = jax.jit(jax.vmap(one))(rngs)
    alive = np.asarray(alive)
    climb = np.asarray(climb)
    vx = np.asarray(vx) / args.steps
    qpos = np.asarray(qpos)                 # (B, nq) 最终状态
    # 初始 xy/yaw: 用 env.reset 逐 env 太贵; 从 spawn 记录重算 (与 reset 同 rng 序列)
    yaw = 2.0 * np.arctan2(qpos[:, 6], qpos[:, 3])
    yaw_deg = np.degrees(np.abs(yaw))
    surv = 100.0 * alive.mean()
    ok = alive
    cl = climb[ok] if ok.any() else climb
    row = dict(policy=os.path.basename(os.path.dirname(os.path.dirname(
        os.path.abspath(args.policy)))) or args.tag,
        terrain=os.path.basename(args.terrain_xml).replace("scene_mjx_terrain_", "").replace(".xml", ""),
        n=args.envs, steps=args.steps, surv=round(float(surv), 1),
        climb_mean=round(float(cl.mean()), 3),
        climb_p50=round(float(np.median(cl)), 3),
        climb_p90=round(float(np.percentile(cl, 90)), 3),
        frac_gt10=round(100.0 * float((cl > 0.10).mean()), 1) if args.riser <= 0 else 0.0,
        frac_gt20=round(100.0 * float((cl > 0.20).mean()), 1) if args.riser <= 0 else 0.0,
        frac_gt30=round(100.0 * float((cl > 0.30).mean()), 1) if args.riser <= 0 else 0.0,
        yaw_abs_p90=round(float(np.percentile(yaw_deg, 90)), 1),
        vx=round(float(vx.mean()), 3))
    if args.riser > 0:
        row["frac_ge1"] = round(100.0 * float((cl >= 0.9 * args.riser).mean()), 1)
        row["frac_ge2"] = round(100.0 * float((cl >= 1.9 * args.riser).mean()), 1)
    if args.top_height > 0:
        row["frac_top"] = round(100.0 * float((cl >= 0.8 * args.top_height).mean()), 1)
    print("[climb] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    if args.out:
        outp = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        new = not os.path.isfile(outp)
        with open(outp, "a", encoding="utf-8") as fh:
            if new:
                fh.write("\t".join(row.keys()) + "\n")
            fh.write("\t".join(str(v) for v in row.values()) + "\n")
        print("[climb] ->", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
