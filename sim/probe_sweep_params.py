#!/usr/bin/env python
"""多检查点 x 多地形一次评测 (warp 训练物理) —— 比逐个起 probe 进程快得多。

关键: 把策略参数作为 jit 的**参数**(而不是闭包常量), 于是同一个地形的 rollout
只编译一次, 所有检查点复用同一份编译产物 (probe_go2_warp_eval.py 是"一个进程
一个策略", 每个 (策略, 地形) 都要重编译, 扫 5 个检查点要 ~30 min; 本脚本 ~1-2 min/地形)。

用法:
  python sim/probe_sweep_params.py \
    --ckpts "v6a=models/go2_v6a_logs/checkpoints/000007372800,v7@4M=models/go2_v7_logs/checkpoints/000004096000" \
    --terrains "v7=models/go2/scene_mjx_terrain_v7.xml,boxes=models/go2/scene_mjx_terrain_boxes_only.xml" \
    --envs 128 --steps 300

输出 TSV: policy terrain surv vx z curve
"""
import argparse
import os
import pickle
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(_ROOT))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from envs.go2_walk import Go2Walk, default_config  # noqa: E402
from common import infer_layers_and_dims  # noqa: E402

DEFAULT_TERRAINS = {
    "mix": "models/go2/scene_mjx_terrain_mix.xml",
    "stairs": "models/go2/scene_mjx_terrain_stairs_only.xml",
    "slope": "models/go2/scene_mjx_terrain_slope_only.xml",
    "rough": "models/go2/scene_mjx_terrain_rough_only.xml",
    "boxes": "models/go2/scene_mjx_terrain_boxes_only.xml",
    "v7": "models/go2/scene_mjx_terrain_v7.xml",
}


def load_params(path):
  """返回 (params, ns, nv); 兼容 orbax 检查点目录与 pkl。"""
  path = os.path.abspath(path)
  if os.path.isdir(path):
    import json
    from brax.training.agents.ppo import checkpoint as ppo_ckpt
    cfg = json.load(open(os.path.join(path, "ppo_network_config.json")))
    params = ppo_ckpt.load(path)
    kw = cfg["network_factory_kwargs"]
    layers = tuple(kw["policy_hidden_layer_sizes"])
    return params, layers, cfg["action_size"]
  with open(path, "rb") as fh:
    params = pickle.load(fh)
  layers, ns, nv = infer_layers_and_dims(params)
  return params, layers, ns


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpts", required=True,
                  help="tag=path,tag=path (orbax 目录或 pkl)")
  ap.add_argument("--terrains", default="",
                  help="name=path,name=path; 空 = 默认 5 地形 + v7")
  ap.add_argument("--envs", type=int, default=128)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
  ap.add_argument("--spawn", choices=["origin", "random"], default="random")
  ap.add_argument("--spawn_max_slope", type=float, default=0.0)
  ap.add_argument("--boxgrid_json", type=str, default="",
                  help="v26/R1: box geom 地形解析高度表 json (对所有地形生效)")
  ap.add_argument("--boxgrid_map", type=str, default="",
                  help="按地形名给不同高度表: 'bg200=path1,stg1=path2'")
  ap.add_argument("--contact_mode", choices=["sensor", "clearance"],
                  default="sensor")
  ap.add_argument("--spawn_ring", action="store_true")
  ap.add_argument("--spawn_stairs_prob", type=float, default=0.0,
                  help="v28: 楼梯口出生概率 (需地形 json 有 spawn_entries)")
  ap.add_argument("--obs_history_frames", type=int, default=0)
  ap.add_argument("--ring_r", type=float, nargs=2, default=None)
  ap.add_argument("--foot-contact", choices=["soft", "hard"], default="hard")
  ap.add_argument("--scan_nx", type=int, default=17)
  ap.add_argument("--scan_ny", type=int, default=11)
  ap.add_argument("--scan_x", type=float, nargs=2, default=[-0.4, 1.2])
  ap.add_argument("--scan_y", type=float, nargs=2, default=[-0.5, 0.5])
  ap.add_argument("--healthy_z_ref", choices=["feet", "body"], default="feet")
  ap.add_argument("--lidar", action="store_true")
  ap.add_argument("--lidar_occ_recall", type=float, default=None)
  ap.add_argument("--lidar_dropout", type=float, default=None)
  ap.add_argument("--lidar_sector_prob", type=float, default=None)
  ap.add_argument("--lidar_mount_jitter", type=float, default=None)
  ap.add_argument("--lidar_vfov_jitter", type=float, default=None)
  ap.add_argument("--out", default="logs/go2_sweep_params.tsv")
  args = ap.parse_args(argv)

  bg_map = {}
  for spec in (args.boxgrid_map or "").split(","):
    spec = spec.strip()
    if spec and "=" in spec:
      k, v = spec.split("=", 1)
      bg_map[k.strip()] = v.strip()

  ckpt_specs = []
  for spec in args.ckpts.split(","):
    spec = spec.strip()
    if not spec:
      continue
    tag, path = spec.split("=", 1)
    ckpt_specs.append((tag.strip(), path.strip()))
  if not ckpt_specs:
    raise SystemExit("--ckpts 为空")

  if args.terrains.strip():
    terr_specs = []
    for spec in args.terrains.split(","):
      tag, path = spec.strip().split("=", 1)
      terr_specs.append((tag.strip(), path.strip()))
  else:
    terr_specs = list(DEFAULT_TERRAINS.items())

  # v27: 支持**不同网络结构**的策略同进程评测 (如 v13 128x128 vs v17 512x256):
  # 按 layers 分组, 每组建一个网络 + 一次 jit, 组内复用, 避免形状不匹配。
  from brax.training.agents.ppo import networks as ppo_networks
  policies = []
  for tag, path in ckpt_specs:
    p, ly, _m = load_params(path)
    policies.append((tag, p, tuple(ly)))
    print("[sweep] policy %-18s layers=%s" % (tag, tuple(ly)))
  groups = {}
  for tag, p, ly in policies:
    groups.setdefault(tuple(ly), []).append((tag, p))
  _, ns, nv = infer_layers_and_dims(policies[0][1])
  print("[sweep] obs state=%d priv=%d; layer groups=%s"
        % (ns, nv, {k: len(v) for k, v in groups.items()}))

  rows = []
  for terr_tag, terr_path in terr_specs:
    cfg = default_config()
    cfg.terrain = True
    cfg.terrain_xml = terr_path
    cfg.height_scan.enable = True
    cfg.height_scan.nx = args.scan_nx
    cfg.height_scan.ny = args.scan_ny
    cfg.height_scan.x_min, cfg.height_scan.x_max = args.scan_x
    cfg.height_scan.y_min, cfg.height_scan.y_max = args.scan_y
    cfg.healthy_z_ref = args.healthy_z_ref
    cfg.terrain_boxgrid_json = bg_map.get(terr_tag, args.boxgrid_json)
    cfg.contact_mode = args.contact_mode
    cfg.random_init.spawn_ring = args.spawn_ring
    cfg.random_init.spawn_stairs_prob = float(args.spawn_stairs_prob)
    if args.obs_history_frames > 0:
      cfg.obs_history.enable = True
      cfg.obs_history.n_frames = int(args.obs_history_frames)
    if args.ring_r is not None:
      cfg.random_init.ring_r = tuple(args.ring_r)
    cfg.random_init.spawn_max_slope = args.spawn_max_slope
    if args.lidar:
      ld = cfg.height_scan.lidar
      ld.enable = True
      if args.lidar_occ_recall is not None:
        ld.occ_recall = args.lidar_occ_recall
      if args.lidar_dropout is not None:
        ld.dropout = args.lidar_dropout
      if args.lidar_sector_prob is not None:
        ld.sector_prob = args.lidar_sector_prob
      if args.lidar_mount_jitter is not None:
        ld.mount_jitter = [args.lidar_mount_jitter] * 3
      if args.lidar_vfov_jitter is not None:
        ld.vfov_jitter = args.lidar_vfov_jitter
    cfg.reward_config.scales.feet_phase = 2.0
    cfg.reward_config.scales.feet_slip = -0.5
    cfg.reward_config.scales.feet_height = -1.0
    cfg.reward_config.scales.feet_dangle = -0.5
    cfg.reward_config.scales.feet_impact = -2.0
    cfg.reward_config.scales.body_contact = -150.0
    cfg.reward_config.scales.ascent = 4.0
    cfg.reward_config.scales.symmetry = -0.4
    cfg.command_config.sample = False
    cfg.command_config.fixed = list(args.cmd)
    cfg.random_init.enable = (args.spawn == "random")
    cfg.foot_soft_contact = (args.foot_contact == "soft")

    env = Go2Walk(cfg)
    print("[sweep] terrain %-6s envs=%d steps=%d  xml=%s"
          % (terr_tag, args.envs, args.steps, terr_path))
    steps = args.steps
    pts = [int(round(x / 0.02)) for x in (0.5, 1, 2, 3, 4, 6)]
    pts = [p for p in pts if 0 < p <= steps]

    rngs = jax.random.split(jax.random.PRNGKey(0), args.envs)

    def _eval_group(layers_key, group):
      net = ppo_networks.make_ppo_networks(
          observation_size={"state": (ns,), "privileged_state": (nv,)},
          action_size=12, policy_hidden_layer_sizes=layers_key,
          value_hidden_layer_sizes=layers_key, policy_obs_key="state",
          value_obs_key="privileged_state", distribution_type="normal",
          activation=jax.nn.tanh)
      dist = net.parametric_action_distribution
      pnet = net.policy_network

      def act(params, obs):
        logits = pnet.apply(params[0], params[1], obs)
        return jp.clip(dist.mode(logits), -1.0e6, 1.0e6)

      def one(rng, params):
        state = env.reset(rng)

        def body(carry, i):
          st, alive, n_alive, vx_sum, z_sum, _hist = carry
          a = act(params, st.obs)
          st2 = env.step(st, a)
          d = st2.data
          R = d.site_xmat[env._imu_site_id].reshape(3, 3)
          vxl = (R.T @ d.qvel[:3])[0]
          new_alive = alive & (~st2.done.astype(bool))
          hist = _hist.at[i].set(new_alive.astype(jp.float32))
          return (st2, new_alive, n_alive + new_alive.astype(jp.float32),
                  vx_sum + jp.where(new_alive, vxl, 0.0),
                  z_sum + jp.where(new_alive, d.qpos[2], 0.0), hist), None

        hist0 = jp.zeros((steps,))
        carry = (state, jp.array(True), jp.zeros(()), jp.zeros(()),
                 jp.zeros(()), hist0)
        carry, _ = jax.lax.scan(body, carry, jp.arange(steps), length=steps)
        st, alive, n_alive, vx_sum, z_sum, hist = carry
        return alive, n_alive, vx_sum, z_sum, hist

      f = jax.jit(jax.vmap(one, in_axes=(0, None)))
      for tag, p in group:
        alive, n_alive, vx_sum, z_sum, hist = f(rngs, p)
        alive = np.asarray(alive)
        n_alive = np.asarray(n_alive)
        hist = np.asarray(hist)
        vx = np.asarray(vx_sum) / np.maximum(n_alive, 1)
        z = np.asarray(z_sum) / np.maximum(n_alive, 1)
        ok = n_alive > 50
        surv = 100.0 * alive.mean()
        curve = ""
        if pts:
          curve = " ".join("%.1fs=%.1f%%" % (pp * 0.02, 100.0 * hist[:, pp - 1].mean())
                           for pp in pts)
        vxm = float(vx[ok].mean()) if ok.any() else float("nan")
        zm = float(z[ok].mean()) if ok.any() else float("nan")
        rows.append((tag, terr_tag, surv, vxm, zm, curve))
        print("  %-18s surv=%5.1f%%  vx=%+.3f  z=%.3f  %s"
              % (tag, surv, vxm, zm, curve))

    for ly_key, grp in groups.items():
      _eval_group(ly_key, grp)
    del env

  out = args.out
  os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  with open(out, "w", encoding="utf-8") as fh:
    fh.write("policy\tterrain\tsurv\tvx\tz\tcurve\n")
    for r in rows:
      fh.write("%s\t%s\t%.1f\t%+.3f\t%.3f\t%s\n" % r)
  print("DONE -> %s" % out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
