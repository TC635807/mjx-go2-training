#!/usr/bin/env python3
"""验证 v3 速度估计误差注入: 统计/时序/与真值的偏差, 并检查 knobs=0 时零影响。

跑法:
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    /home/tc63/mujoco/.venv/bin/python sim/probe_obs_vel_error.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax, jax.numpy as jp, numpy as np
from envs.go2_walk import Go2Walk, default_config


def make(bias=0.0, drift=0.0, tau=2.0, delay=0, white=0.05, terrain=False):
  cfg = default_config()
  cfg.terrain = terrain
  cfg.obs_noise.enable = True
  cfg.obs_noise.lin_vel = white
  cfg.obs_noise.lin_vel_bias = bias
  cfg.obs_noise.lin_vel_drift_sigma = drift
  cfg.obs_noise.lin_vel_drift_tau = tau
  cfg.obs_noise.lin_vel_delay_steps = delay
  cfg.naconmax = 32 * 64          # probe 只用 8 env, 不占大块显存
  return Go2Walk(config=cfg)


def rollout(env, n_envs, n_steps, seed=0):
  rngs = jax.random.split(jax.random.PRNGKey(seed), n_envs)
  state = jax.vmap(env.reset)(rngs)
  step = jax.vmap(env.step)
  act = jp.zeros((n_envs, env.mjx_model.nu))
  errs, ests, truths, delays = [], [], [], []
  for i in range(n_steps):
    state = step(state, act)
    est = state.info["vel_est_world"]
    truth = state.data.qvel[:, :3]
    errs.append(np.asarray(est - truth))
    ests.append(np.asarray(est)); truths.append(np.asarray(truth))
    if i == 0:
      delays.append(np.asarray(state.info["vel_delay"]))
  return np.stack(errs), np.stack(ests), np.stack(truths), np.stack(delays)


def report(tag, errs, delays=None):
  # errs: (T, N, 3)
  e = errs
  bias = e.mean(axis=0)                       # per-env mean
  inst = e - bias[None]                       # 去均值后的即时波动
  print(f"[{tag}]")
  print(f"  每回合偏置 |mean| = {np.abs(bias).mean():.3f} m/s  (逐轴: %s)"
        % np.round(bias.mean(axis=0), 3))
  print(f"  即时误差 std      = {inst.std():.3f} m/s  (逐轴 %s)"
        % np.round(inst.std(axis=(0, 1)), 3))
  print(f"  总误差 RMS        = {np.sqrt((e ** 2).mean()):.3f} m/s")
  # 自相关: 1 步 / 10 步 / 1s(50步) -> 慢漂占比
  def ac(lag):
    a = e[:-lag].reshape(-1, 3); b = e[lag:].reshape(-1, 3)
    return float(((a - a.mean(0)) * (b - b.mean(0))).mean()
                 / (a.var(0).mean() + 1e-12))
  print(f"  自相关 lag1={ac(1):+.2f} lag10={ac(10):+.2f} lag50={ac(50):+.2f}"
        "  (lag50 高 => 慢漂明显)")
  if delays is not None:
    d = delays.ravel()
    print(f"  延迟采样: min={d.min()} max={d.max()} mean={d.mean():.2f} 步, "
          f"分布={np.bincount(d.astype(int)).tolist()}")


def main():
  N = 8
  # ---- A) knobs=0: 估计值必须 == 真值 (证明对 v2 零影响) ----
  env0 = make()
  e0, _, _, _ = rollout(env0, N, 5)
  print("A) 全 0 档: max|est-truth| = %.3e m/s  (应为 0)" % np.abs(e0).max())

  # ---- B) v3 推荐档 ----
  env1 = make(bias=0.12, drift=0.08, tau=2.0, delay=2, white=0.15)
  e1, est, tru, dl = rollout(env1, N, 120)
  report("B) bias=0.12 drift=0.08 tau=2 delay<=2 white=0.15", e1, dl)
  print(f"   速度量级: |est|={np.linalg.norm(est,axis=-1).mean():.3f}  "
        f"|truth|={np.linalg.norm(tru,axis=-1).mean():.3f} m/s")

  # ---- C) 实测口径 (sim 内 LegOdometry: bias~0.07, 即时 std~0.23, 平滑 std~0.06) ----
  env2 = make(bias=0.07, drift=0.06, tau=2.0, delay=1, white=0.20)
  e2, _, _, _ = rollout(env2, N, 120)
  report("C) 对齐实测 LegOdometry (bias=0.07 drift=0.06 delay<=1 white=0.20)", e2)

  # ---- D) actor obs 0:3 是否真的用了估计值 (obs/2 == est_body) ----
  st = jax.vmap(env1.reset)(jax.random.split(jax.random.PRNGKey(1), N))
  st = jax.vmap(env1.step)(st, jp.zeros((N, env1.mjx_model.nu)))
  obs = np.asarray(st.obs["state"])[:, :3] / 2.0
  est_body = np.asarray(
      jax.vmap(lambda d, e: d.site_xmat[env1._imu_site_id].T @ e)(
          st.data, st.info["vel_est_world"]))
  print("D) actor obs[0:3]/2 vs est_body: max|d| = %.3e (应≈0, 只剩白噪声差异)"
        % np.abs(obs - est_body).max())
  print("   注: obs 上还叠了 ±%.2f m/s 白噪声, 所以 D 不是严格 0" % 0.15)


if __name__ == "__main__":
  main()
