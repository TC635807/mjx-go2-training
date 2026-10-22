#!/usr/bin/env python
"""v30: brax 训练用 per-env 地形课程包装器 (legged_gym 同款语义)。

为什么需要自己写: playground 的 wrap_for_brax_training(full_reset=True) 每回合调
env.reset(rng), 但 reset 拿不到上一回合的课程 level, 也没法把 level 传进去。
这里在 AutoReset 位置插一层:
  - reset: 每个 env 从 U{0..init_max} 采样初始 level, 调 env.reset_with_level(rng, level)
  - step: 先走一步 -> 用 (位移, 是否摔倒) 升降级 -> 只对 done 的 env 用新 level 重开
公式 (legged_gym legged_robot.py::_update_terrain_curriculum / mjlab curriculums.py):
  move_up   = (dist > success_dist) & ~fall
  move_down = (fall | (dist < min_dist)) & ~move_up
  level    += up - down, clip 到 [0, max_level]
"""
import jax
from jax import numpy as jp

from brax.envs.wrappers import training as brax_training
from mujoco_playground._src.wrapper import Wrapper as MjxWrapper


class CurriculumVmapWrapper(brax_training.VmapWrapper):
  """批量 reset/step, reset 时按 per-env level 调 env.reset_with_level。"""

  def reset(self, rng_batch, ):
    levels = self.env.sample_init_levels(rng_batch)
    return jax.vmap(self.env.reset_with_level)(rng_batch, levels)

  def reset_with_level(self, rng_batch, levels):
    return jax.vmap(self.env.reset_with_level)(rng_batch, levels)


class CurriculumAutoResetWrapper(MjxWrapper):
  """EpisodeWrapper 外层: done 的 env 按课程 level 重开。

  与 playground BraxAutoResetWrapper 的区别:
  * done 的 env 用 **新 level** 的 spawn_entries 重开 (课程晋升/降级);
  * episode 簿记 (steps/truncation/episode_done/episode_metrics) 手工补上;
  * 非 done 的 env 原样保留 (data/obs/info/metrics 用 where 合并)。
  """

  def __init__(self, env, success_dist=3.0, min_dist=1.0, max_level=5):
    super().__init__(env)
    self._succ = float(success_dist)
    self._min = float(min_dist)
    self._max_level = int(max_level)

  def reset(self, rng_batch):
    state = self.env.reset(rng_batch)
    state.info["first_data"] = state.data
    state.info["first_obs"] = state.obs
    return state

  def step(self, state, action):
    # 与 playground wrapper 一致: done 的 env 先把 steps 归零再走
    if "steps" in state.info:
      state.info["steps"] = jp.where(
          state.done, jp.zeros_like(state.info["steps"]), state.info["steps"])
    state = state.replace(done=jp.zeros_like(state.done))
    nstate = self.env.step(state, action)

    # ---- 课程晋升/降级 (用本步结束时的状态) ----
    dist = jp.linalg.norm(
        nstate.data.qpos[..., :2] - nstate.info["spawn_xy"], axis=-1)
    trunc = nstate.info.get("truncation", jp.zeros_like(nstate.done))
    done = nstate.done.astype(bool)
    fall = done & (trunc < 0.5)
    up = (dist > self._succ) & (~fall)
    down = (fall | (dist < self._min)) & (~up)
    lvl = jp.clip(
        nstate.info["terrain_level"]
        + up.astype(jp.int32) - down.astype(jp.int32), 0, self._max_level)
    lvl = jp.where(done, lvl, nstate.info["terrain_level"])

    # ---- 只对 done 的 env 用新 level 重开 (全部算, 用 where 合并) ----
    rngs = jax.vmap(jax.random.split)(nstate.info["curr_rng"])
    reset_rng, next_rng = rngs[..., 0], rngs[..., 1]
    rs = self.env.env.reset_with_level(reset_rng, jp.minimum(lvl, self._max_level))
    rs.info["steps"] = jp.zeros_like(nstate.info["steps"])
    rs.info["truncation"] = jp.zeros_like(nstate.info["truncation"])
    rs.info["episode_done"] = jp.zeros_like(nstate.info["episode_done"])
    rs.info["episode_metrics"] = jax.tree.map(
        jp.zeros_like, nstate.info["episode_metrics"])
    rs.info["curr_rng"] = next_rng
    # 与外层 reset 保持相同 info 键 (否则 tree.map 合并时 dict 键不匹配)
    rs.info["first_data"] = rs.data
    rs.info["first_obs"] = rs.obs

    def wd(x, y):
      if (hasattr(x, "shape") and x.shape and done_shape[0] == x.shape[0]):
        d = jp.reshape(nstate.done, [done_shape[0]] + [1] * (x.ndim - 1))
        return jp.where(d, x, y)
      return y

    done_shape = nstate.done.shape
    data = jax.tree.map(wd, rs.data, nstate.data)
    obs = jax.tree.map(wd, rs.obs, nstate.obs)
    info = jax.tree.map(wd, rs.info, nstate.info)
    # 关键: brax 的 logger 在 **done 那一步** 读取 episode_metrics/steps 来算
    # ep_len/指标, 所以这些簿记必须保留 nstate (完成的那一回合) 的值,
    # 不能换成 reset 的零值; 下一回合开始时 EpisodeWrapper/AutoReset 逻辑会归零。
    for _k in ("steps", "truncation", "episode_done", "episode_metrics"):
      info[_k] = nstate.info[_k]
    # metrics 键可能不一致 (训练循环会额外加 'reward'), 只合并两边都有的键
    metrics = dict(nstate.metrics)
    for k in rs.metrics:
      if k in nstate.metrics:
        metrics[k] = wd(rs.metrics[k], nstate.metrics[k])
    return nstate.replace(data=data, obs=obs, info=info, metrics=metrics)


def wrap_for_curriculum_training(
    env, episode_length=750, action_repeat=1, randomization_fn=None,
    init_level_max=2, success_dist=3.0, min_dist=1.0, max_level=5):
  """构造 课程 AutoReset(Episode(Vmap(env))) 栈; ppo.train 的 wrap_env_fn 直接用它。"""
  vm = CurriculumVmapWrapper(env)
  ep = brax_training.EpisodeWrapper(vm, episode_length, action_repeat)
  return CurriculumAutoResetWrapper(
      ep, success_dist=success_dist, min_dist=min_dist, max_level=max_level)
