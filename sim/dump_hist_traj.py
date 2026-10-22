#!/usr/bin/env python
"""dump 一条训练环境 (warp) 轨迹的 qpos/qvel/gyro/obs, 供 CPU 侧对比部署 obs builder。

  python sim/dump_hist_traj.py --policy models/go2_v18h_hist3_logs/checkpoints/000020070400 \
      --out /tmp/hist_traj.npz --steps 40
"""
import argparse, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")
import numpy as np
import jax
import jax.numpy as jp
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(_ROOT)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.go2_walk import Go2Walk, default_config
from common import load_policy

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", required=True)
  ap.add_argument("--terrain_xml", default="models/go2/scene_mjx_terrain_hyb2.xml")
  ap.add_argument("--boxgrid_json", default="models/go2/assets/hyb2.json")
  ap.add_argument("--obs_history_frames", type=int, default=3)
  ap.add_argument("--steps", type=int, default=40)
  ap.add_argument("--out", default="/tmp/hist_traj.npz")
  args = ap.parse_args()
  cfg = default_config()
  cfg.terrain = True; cfg.terrain_xml = args.terrain_xml
  cfg.terrain_boxgrid_json = args.boxgrid_json
  cfg.contact_mode = "clearance"
  cfg.random_init.enable = True
  cfg.random_init.spawn_ring = True
  cfg.height_scan.enable = True
  cfg.height_scan.nx, cfg.height_scan.ny = 17, 11
  cfg.height_scan.x_min, cfg.height_scan.x_max = -0.4, 1.2
  cfg.height_scan.y_min, cfg.height_scan.y_max = -0.5, 0.5
  cfg.healthy_z_ref = "feet"
  cfg.obs_history.enable = True
  cfg.obs_history.n_frames = args.obs_history_frames
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
  infer, ns = load_policy(os.path.abspath(args.policy))
  print("obs dim", ns, "env", env.observation_size["state"])
  rng = jax.random.PRNGKey(0)
  st = jax.jit(env.reset)(rng)
  gyro_adr = int(env._mj_model.sensor_adr[env._mj_model.sensor("gyro").id])
  qpos, qvel, gyro, obs_l, acts = [], [], [], [], []
  for i in range(args.steps):
    act, _ = infer(st.obs, rng)
    acts.append(np.asarray(act))
    st = jax.jit(env.step)(st, act)
    d = st.data
    qpos.append(np.asarray(d.qpos)); qvel.append(np.asarray(d.qvel))
    gyro.append(np.asarray(d.sensordata[gyro_adr:gyro_adr+3]))
    obs_l.append(np.asarray(st.obs["state"]))
  out = dict(qpos=np.stack(qpos), qvel=np.stack(qvel), gyro=np.stack(gyro),
             obs=np.stack(obs_l), act=np.stack(acts),
             cmd=np.array([0.5,0,0], np.float64))
  np.savez(args.out, **out)
  print("saved", args.out, "T", len(qpos))

if __name__ == "__main__":
  main()
