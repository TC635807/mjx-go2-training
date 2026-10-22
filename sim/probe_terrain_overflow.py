#!/usr/bin/env python3
"""在 warp (训练物理) 里跑随机动作, 检查地形接触溢出 (geom-hfield 每对 50 接触上限)。

溢出后果: 超过上限的接触被丢弃 -> 物理静默退化 (和 naconmax 溢出同性质)。
warp 的提示会直接 printf 到 stdout, 所以本脚本只负责制造接触; 计数在调用方 grep:
  python sim/probe_terrain_overflow.py --scene models/go2/scene_mjx_terrain_160.xml \
      > /tmp/p.log 2>&1; grep -c 'height field collision overflow' /tmp/p.log

用法: /home/tc63/mujoco/.venv/bin/python sim/probe_terrain_overflow.py --scene X [--envs 16] [--steps 300]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax, jax.numpy as jp, numpy as np
from envs.go2_walk import Go2Walk, default_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = default_config()
    cfg.terrain = True
    cfg.terrain_xml = args.scene
    cfg.height_scan.enable = True
    cfg.random_init.enable = True
    cfg.random_init.xy_range = 5.0
    cfg.push_config.enable = False
    cfg.obs_noise.enable = False
    cfg.naconmax = 32 * args.envs
    env = Go2Walk(config=cfg)
    print("[probe] scene=%s hfield=%dx%d" % (
        args.scene, env.mj_model.hfield_ncol[0], env.mj_model.hfield_nrow[0]))

    N = args.envs
    keys = jax.random.split(jax.random.PRNGKey(args.seed), N)
    state = jax.vmap(env.reset)(keys)
    step = jax.vmap(env.step)
    lo = env.mj_model.actuator_ctrlrange[:, 0]
    hi = env.mj_model.actuator_ctrlrange[:, 1]
    key = jax.random.PRNGKey(args.seed + 1)
    rng_act = np.random.default_rng(args.seed)

    for i in range(args.steps):
        key, k = jax.random.split(key)
        act = jax.random.uniform(k, (N, env.mjx_model.nu), minval=lo, maxval=hi)
        # 一半环境用随机目标 (摔/撞), 一半用 home (站/走) -> 覆盖接触形态
        home = jp.tile(jp.array(env.mj_model.key_ctrl[0]), (N, 1))
        mask = (jp.arange(N) % 2)[:, None]
        act = jp.where(mask == 0, act, home + 0.3 * jax.random.normal(k, (N, env.mjx_model.nu)))
        state = step(state, act)
        if i % 100 == 0:
            q = np.asarray(state.data.qpos)
            print("[probe] step %3d finite=%s z_min=%.3f" % (
                i, bool(np.all(np.isfinite(q))), float(q[:, 2].min())))
    print("[probe] done %d steps" % args.steps)


if __name__ == "__main__":
    main()
