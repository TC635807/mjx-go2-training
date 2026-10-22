#!/usr/bin/env python
"""地形物理吞吐探针: 只测 jit 后的 rollout 墙钟时间 (env-steps/s), 用来判断
box geom 地形 (R1) 的宽相/接触开销能不能支撑训练。

用法:
  python sim/probe_perf_terrain.py --terrain_xml models/go2/scene_mjx_terrain_mix.xml
  python sim/probe_perf_terrain.py --terrain_xml models/go2/scene_mjx_terrain_bg200.xml \
      --boxgrid_json models/go2/assets/bg200.json --contact_mode clearance --spawn_ring
"""
import argparse
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(_ROOT))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from envs.go2_walk import Go2Walk, default_config  # noqa: E402
from common import load_policy  # noqa: E402


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--terrain_xml", required=True)
  ap.add_argument("--boxgrid_json", default="")
  ap.add_argument("--contact_mode", choices=["sensor", "clearance"], default="sensor")
  ap.add_argument("--spawn_ring", action="store_true")
  ap.add_argument("--ring_r", type=float, nargs=2, default=None)
  ap.add_argument("--policy", default="models/go2_v6a_logs/checkpoints/000007372800")
  ap.add_argument("--envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--repeats", type=int, default=3)
  args = ap.parse_args(argv)

  cfg = default_config()
  cfg.terrain = True
  cfg.terrain_xml = args.terrain_xml
  cfg.terrain_boxgrid_json = args.boxgrid_json
  cfg.contact_mode = args.contact_mode
  cfg.random_init.enable = True
  cfg.random_init.spawn_ring = args.spawn_ring
  if args.ring_r is not None:
    cfg.random_init.ring_r = tuple(args.ring_r)
  cfg.height_scan.enable = True
  cfg.height_scan.nx, cfg.height_scan.ny = 17, 11
  cfg.height_scan.x_min, cfg.height_scan.x_max = -0.4, 1.2
  cfg.height_scan.y_min, cfg.height_scan.y_max = -0.5, 0.5
  cfg.height_scan.lidar.enable = True
  cfg.healthy_z_ref = "feet"
  cfg.foot_soft_contact = False
  cfg.reward_config.scales.feet_phase = 2.0
  cfg.reward_config.scales.feet_slip = -0.5
  cfg.reward_config.scales.feet_height = -1.0
  cfg.reward_config.scales.feet_dangle = -0.5
  cfg.reward_config.scales.feet_impact = -2.0
  cfg.reward_config.scales.body_contact = -150.0
  cfg.reward_config.scales.ascent = 4.0
  cfg.reward_config.scales.symmetry = -0.4
  cfg.command_config.sample = False
  cfg.command_config.fixed = [0.5, 0.0, 0.0]

  env = Go2Walk(cfg)
  infer, _ = load_policy(os.path.abspath(args.policy))

  def one(rng):
    state = env.reset(rng)

    def body(carry, i):
      st, alive = carry
      act, _ = infer(st.obs, rng)
      st2 = env.step(st, act)
      return (st2, alive & (~st2.done.astype(bool))), None

    carry, _ = jax.lax.scan(body, (state, jp.array(True)),
                            jp.arange(args.steps), length=args.steps)
    return carry[1]

  rngs = jax.random.split(jax.random.PRNGKey(0), args.envs)
  f = jax.jit(jax.vmap(one))
  t0 = time.perf_counter()
  f(rngs).block_until_ready()                     # 编译 + 首次
  t_compile = time.perf_counter() - t0
  ts = []
  for _ in range(args.repeats):
    t0 = time.perf_counter()
    f(rngs).block_until_ready()
    ts.append(time.perf_counter() - t0)
  fps = args.envs * args.steps / float(np.median(ts))
  print("[perf] %s boxgrid=%s contact=%s envs=%d steps=%d"
        % (os.path.basename(args.terrain_xml), bool(args.boxgrid_json),
           args.contact_mode, args.envs, args.steps))
  print("[perf] compile+first=%.1fs  rollout median=%.3fs  -> %.0f env-steps/s"
        % (t_compile, float(np.median(ts)), fps))
  return 0


if __name__ == "__main__":
  sys.exit(main())
