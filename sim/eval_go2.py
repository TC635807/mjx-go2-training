#!/usr/bin/env python
"""eval_go2.py —— Go2 策略闭环评测工具 (CPU MuJoCo, 可重复, 无 GPU 依赖)。

目的
----
回答一个问题: "这个策略能不能在一定复杂地形上稳定行走?"
为此本脚本用 **CPU MuJoCo**(不是 MJX/warp, 不占 GPU) 跑 50Hz 关节位置闭环,
逐 episode 记录轨迹, 汇总 6 类核心指标 (速度跟踪 / 存活 / 姿态 / 步态 /
地形通过 / 曲线图), 输出 JSON + PNG。

观测与动作的"唯一真值来源"
--------------------------
逐项对齐 envs/go2_walk.py (= envs/go1_walk.py 的 Go2 移植), 关键代码位置:
  _get_obs (go2_walk.py:1070-1162)      obs 布局 + 缩放 + clip
  height_scan_features (go2_walk.py:714-745)  35 点高度扫描
  terrain_height (go2_walk.py:670-712)  hfield 双线性采样 (逐点复刻)
  step (go2_walk.py:901-907)            动作 = 绝对位置目标, clip 到 ctrlrange
  reset (go2_walk.py:789-843)           出生噪声 / 指令 / 相位初值
  _is_healthy (go2_walk.py:753-785)     终止判据
  default_config (go2_walk.py:63-399)   标量: base=0.268 / clip_m=0.35 / scale=5.0
                                        gait_freq=1.8 / obs_clip=100 / healthy 阈值

actor obs (48 + 35 + 8, 由策略权重的第一层维度自动决定):
  0:3    base 线速度 * 2.0        (frame="body" -> R^T @ qvel[:3]; "global" -> qvel[:3])
  3:6    base 角速度 * 0.25       (qvel[3:6])
  6:9   机体系重力 R^T @ [0,0,-1]
  9:12   指令 * 2.0
  12:24  qpos[7:] - key_ctrl[0]  (关节角相对 home ctrl 的偏移)
  24:36  qvel[6:]                (**原样, 不乘 0.05** —— 见下方"已知坑")
  36:48  last_action             (上一步策略原始输出, 未 clip)
  48:83  高度扫描 (可选)          clip(z - h_ground - 0.268, ±0.35) * 5.0
  83:91  相位编码 (可选)          [cos(phi_i), sin(phi_i)] x 4 足 (trot = [0,pi,pi,0])
  最后整体 clip ±100。

动作: 12 维**绝对关节位置目标** (ctrl = clip(a, 关节范围)); 若模型用的是
力矩电机 (<motor>) 而不是 <position>, 则退化为显式 PD
tau = clip(kp*(q_des-q) - kd*qd, ctrlrange) (与 mjx-go2-mujoco2ros2 部署侧
LowCmd 的 kp/kd 语义一致, 默认 kp=20 kd=1)。

物理档位 (--solver, 默认 cpu) —— 实测对比, 见 docs/go2-eval-harness.md
--------------------------------------------------------------
  cpu  100/50  CPU MuJoCo 完整 Newton。**推荐**: 静置 home 得 z=0.267 m /
              pitch 1.5° (训练模型) 与 z=0.273 m / 1.5° (部署模型, 与任务给的
              实测值 0.268 m / 1.58° 一致)
  env   20/10  envs/go2_walk 的 hard_contact_iters 档。对**软接触足端**的
              Go2 训练模型 (go2_mjx_position.xml 里 solimp=0.9/0.95) 数值上
              站不住: z 掉到 0.226 m、pitch 漂到 8.5°/19.8° → 8s 内触发终止。
              这是求解器迭代不够 (接触约束未收敛), 不是策略问题 —— 评测请用 cpu。
  soft   2/5   envs/go2_walk 的 MJX 软接触档 (warp 后端用它), CPU Newton 下
              下沉更深 (z≈0.22), 不建议 CPU 评测使用。
  xml         保持模型自带/引擎默认。

已知坑 (读源码发现的与直觉不符之处, 已在代码里按源码实现并标注)
----------------------------------------------------------------
1. obs_scale.dofs_velocity=0.05 **在代码里没有被使用**: _get_obs 里
   dofs_velocity = data.qvel[6:] 原样进 obs。本脚本照源码 (不缩放)。
   部署侧 src/go2_bridge/obs.py:343 的注释也确认了这一点 ("不缩放!")。
2. 动作**没有**"soft range 内缩 5%"的线性映射 —— 5% 内缩只用在 joint_limit
   奖励 (_soft_lowers/_soft_uppers) 上; step() 是 clip(action, ctrlrange)。
   若要测 [-1,1] 的 tanh 策略, 用 --action-mode tanh_soft。
3. 终止判据用的是**四元数分量** q[1]/q[2] < deg2rad(10) (≈ 真欧拉 ±20°),
   不是真 roll/pitch。本脚本默认 --fall-mode env 复刻它, 同时另报真 roll/pitch。

用法
----
  # (1) 纯链路自检: 无策略, 顶住 home 姿态站立 (验证物理+指标+出图)
  python sim/eval_go2.py --policy none --dummy hold_home --terrain flat \
      --seeds 4 --duration 6 --tag selfcheck_hold

  # (2) 自检: 零动作/随机动作 (验证摔倒检测)
  python sim/eval_go2.py --policy none --dummy zero --seeds 4 --duration 5
  python sim/eval_go2.py --policy none --dummy random --seeds 4 --duration 5

  # (3) 真策略 (pkl 或 orbax checkpoint 目录)
  python sim/eval_go2.py --policy policies/go2_walk_policy.pkl \
      --terrain terrain --cmd 0.5 0 0 --seeds 20 --duration 15

  # (4) 交叉验证 (harness 自身正确性): Go1 pkl 在 Go1 模型上应当会走
  python sim/eval_go2.py --policy policies/go1_walk_policy.pkl \
      --xml models/go1/scene_mjx_position.xml --vel-frame world \
      --cmd 0.5 0 0 --seeds 4 --duration 8

输出 (默认 mjx-go1-getup/logs/go2_eval/):
  <tag>.json   全部标量指标
  <tag>.png    4 联图: 速度跟踪 / 躯干高度 / 每关节力矩 / FR 关节力矩曲线
  <tag>.trace.npz  可选 (--dump-trace)

注意: 本脚本默认强制 JAX 跑 CPU (JAX_PLATFORM_NAME=cpu), 不碰 GPU。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_DEPLOY_GO2 = os.environ.get(
    "GO2_DEPLOY_DIR", "/home/tc63/mujoco/mjx-go2-mujoco2ros2/models/go2")

# ---------------------------------------------------------------- 常量
# 全部来自 envs/go2_walk.py default_config() / __init__ / _get_obs
FEET = ["FR", "FL", "RR", "RL"]          # 训练侧足序 (go2_walk.py:60)
CTRL_DT = 0.02                            # 50Hz
SIM_DT = 0.002                           # XML timestep
EPISODE_LEN = 750                        # 15s
N_SUBSTEPS = int(round(CTRL_DT / SIM_DT))
OBS_CLIP = 100.0
SC_LIN = 2.0                             # obs_scale.linear_velocity
SC_ANG = 0.25                            # obs_scale.angular_velocity
HEALTHY_Z = (0.19, 0.65)                 # healthy_z_range (Go2; Go1 是 0.22)
HEALTHY_QXY_DEG = 10.0                   # healthy_roll/pitch_range (卡四元数分量)
SCAN = dict(nx=7, ny=5, x_min=-0.3, x_max=0.9, y_min=-0.4, y_max=0.4,
            base=0.268, clip_m=0.35, scale=5.0)
TROT_PHASE0 = np.array([0.0, np.pi, np.pi, 0.0])   # 与 FEET 同序
GAIT_FREQ = 1.8                          # reward_config.gait_freq (Hz)
GAIT_FREQ_SCALED = False                 # gait_freq_scaled_by_cmd
GAIT_FREQ_MIN = 0.6
SOFT_INSET = 0.05                        # 0.5*(1-0.9): ctrlrange 内缩 5%
SOLVER_ITERS = 20                        # env 的 hard_contact_iters
SOLVER_LS = 10
# 求解器档位: 名称 -> (iterations, ls_iterations)
SOLVER_PRESETS = {
    "cpu": (100, 50),    # CPU MuJoCo 默认 = 参考仓库的完整 Newton (推荐)
    "env": (SOLVER_ITERS, SOLVER_LS),   # envs/go2_walk 硬接触档 (20/10)
    "soft": (2, 5),      # envs/go2_walk MJX 软接触档 (2/5)
}
# MJX 软接触足端覆写 (foot_soft_contact=True 时训练用的值)
FOOT_SOLIMP = np.array([0.9, 0.95, 0.023, 0.5, 2.0])
FOOT_CONDIM = 3
# unitree_rl_gym / 官方部署惯例 home 姿态 (Go2, 12 关节)
HOME_UNITREE = dict(
    FL_hip=0.1, FR_hip=-0.1, RL_hip=0.1, RR_hip=-0.1,
    FL_thigh=0.8, FR_thigh=0.8, RL_thigh=1.0, RR_thigh=1.0,
    FL_calf=-1.5, FR_calf=-1.5, RL_calf=-1.5, RR_calf=-1.5)


def unitree_home_value(joint_name):
  """关节名 -> unitree_rl_gym home 值。兼容 "FL_hip" 与 "FL_hip_joint" 两种命名。"""
  if joint_name is None:
    return None
  n = str(joint_name).replace("_joint", "")
  for leg in ("FL", "FR", "RL", "RR"):
    if n.startswith(leg):
      part = n[len(leg):].lstrip("_")
      key = f"{leg}_{part}"
      if key in HOME_UNITREE:
        return HOME_UNITREE[key]
  return None


# ---------------------------------------------------------------- 小工具
def log(msg):
  print(msg, flush=True)


def rpy_from_quat(q):
  """quat (w,x,y,z) -> (roll, pitch, yaw) rad。"""
  w, x, y, z = (float(v) for v in q)
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  sp = 2 * (w * y - z * x)
  pitch = np.arcsin(np.clip(sp, -1.0, 1.0))
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return roll, pitch, yaw


# ---------------------------------------------------------------- 策略加载
def infer_layers_and_dims(params):
  """从参数形状反推隐层与 obs 维度 (与 sim/common.py 同逻辑, 独立实现)。"""
  roots = [params]
  if isinstance(params, (tuple, list)) and len(params) > 1:
    roots = [params[1]]
  found = {}

  def walk(node):
    if isinstance(node, (tuple, list)):
      for v in node:
        walk(v)
      return
    if not hasattr(node, "items"):
      return
    for k, v in node.items():
      if (k.startswith("hidden_") and hasattr(v, "get") and "kernel" in v
          and hasattr(v["kernel"], "shape")):
        found[int(k.split("_")[1])] = int(v["kernel"].shape[1])
      walk(v)

  for r in roots:
    walk(r)
  layers = tuple(found[i] for i in sorted(found)) if found else (128, 128)

  def first_dim(node):
    best = {}

    def w(n):
      if isinstance(n, (tuple, list)):
        for v in n:
          w(v)
        return
      if not hasattr(n, "items"):
        return
      for k, v in n.items():
        if (k == "hidden_0" and hasattr(v, "get") and "kernel" in v
            and hasattr(v["kernel"], "shape")):
          best.setdefault("dim", int(v["kernel"].shape[0]))
        w(v)

    w(node)
    return best.get("dim")

  if isinstance(params, (tuple, list)) and len(params) >= 3:
    ns, nv = first_dim(params[1]) or 48, first_dim(params[2]) or 119
  else:
    ns, nv = (first_dim(params) or 48), 119
  return layers, ns, nv


def _policy_params_of(params):
  p = params[1] if isinstance(params, (tuple, list)) else params
  if hasattr(p, "get") and "params" in p:
    p = p["params"]
  return p


def _has_std_param(node):
  if hasattr(node, "keys"):
    if "std_param" in node:
      return True
    return any(_has_std_param(node[k]) for k in node)
  if isinstance(node, (tuple, list)):
    return any(_has_std_param(v) for v in node)
  return False


def _normalizer_info(norm_params):
  """返回 (enabled, is_multi_key)。enabled = 训练时真的更新过 running stats。"""
  if norm_params is None:
    return False, False
  cnt = getattr(norm_params, "count", None)
  multi = hasattr(norm_params, "mean") and hasattr(norm_params.mean, "keys")
  if cnt is None:
    return False, multi
  try:
    import numpy as _np
    n = int(_np.asarray(cnt))
  except Exception:  # pragma: no cover
    n = 0
  return n > 0, multi


class Policy:
  """统一的策略接口: pkl / orbax checkpoint 目录 / 无策略 (dummy)。

  * 归一化: brax 的 ppo.train 返回的 params = (normalizer, policy, value)。
    **坑**: brax 会**无条件**更新 running stats (ppo/train.py:611-619), 不管
    normalize_observations 是 True 还是 False —— 所以 normalizer.count > 0
    **不能**作为"训练时开了归一化"的证据。唯一可靠来源是训练开关:
      - orbax checkpoint: ppo_network_config.json 的 normalize_observations
      - pkl: 没存 -> 默认 off (实测证据见 docs/go2-eval-harness.md §7 坑 4)
    要显式打开用 --normalize on。多 key 归一化时还必须把 obs 包成
    {"state": obs} 传进去, brax 才会用 normalizer_select 取对那一份统计量。
  """

  def __init__(self, path, action_size=12, normalize="auto", dist="auto",
               verbose=True):
    import jax
    import jax.numpy as jnp
    import pickle
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics

    self.name = path
    if os.path.isdir(path):
      from brax.training.agents.ppo import checkpoint as ppo_ckpt
      cfg_path = os.path.join(path, "ppo_network_config.json")
      cfg = json.load(open(cfg_path))
      kw = cfg["network_factory_kwargs"]
      self._ckpt_normalize = cfg.get("normalize_observations", None)
      params = ppo_ckpt.load(path)
      layers, ns, nv = infer_layers_and_dims(params)
      self.action_size = int(cfg["action_size"])
      dist = kw.get("distribution_type", "tanh_normal") if dist == "auto" else dist
      net = ppo_networks.make_ppo_networks(
          observation_size={"state": (ns,), "privileged_state": (nv,)},
          action_size=self.action_size,
          preprocess_observations_fn=self._preprocess(
              self._resolve_normalize(normalize, self._ckpt_normalize),
              running_statistics),
          policy_hidden_layer_sizes=tuple(kw["policy_hidden_layer_sizes"]),
          value_hidden_layer_sizes=tuple(kw["value_hidden_layer_sizes"]),
          policy_obs_key=kw["policy_obs_key"], value_obs_key=kw["value_obs_key"],
          distribution_type=dist, activation=jax.nn.tanh)
      self.source = "orbax:" + cfg_path
    else:
      with open(path, "rb") as f:
        params = pickle.load(f)
      self._ckpt_normalize = None
      layers, ns, nv = infer_layers_and_dims(params)
      pol = _policy_params_of(params)
      if dist == "auto":
        dist = "normal" if _has_std_param(pol) else "tanh_normal"
      self.action_size = action_size
      net = ppo_networks.make_ppo_networks(
          observation_size={"state": (ns,), "privileged_state": (nv,)},
          action_size=self.action_size,
          preprocess_observations_fn=self._preprocess(
              self._resolve_normalize(normalize, self._ckpt_normalize),
              running_statistics),
          policy_hidden_layer_sizes=layers,
          value_hidden_layer_sizes=layers,
          policy_obs_key="state", value_obs_key="privileged_state",
          distribution_type=dist, activation=jax.nn.tanh)
      self.source = "pkl:" + os.path.abspath(path)

    self.dist = dist
    self.ns = int(ns)
    self.nv = int(nv)
    self.layers = tuple(layers)
    self.normalizer, self.multi_key = _normalizer_info(
        params[0] if isinstance(params, (tuple, list)) and len(params) > 0 else None)
    self.use_normalize = self._resolve_normalize(normalize, self._ckpt_normalize)
    self._fn = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    self._jax = jax
    self._jnp = jnp
    self._jit = jax.jit(self._fn)
    if verbose:
      log(f"  [policy] {self.source}")
      log(f"  [policy] layers={self.layers} action={self.action_size} "
          f"obs(state={self.ns}, privileged={self.nv}) dist={self.dist}")
      log(f"  [policy] obs 归一化={'ON' if self.use_normalize else 'off'}"
          + (f"  (running stats 非空 count>0, 但训练开关未记录 -> 默认 off; "
             f"要打开用 --normalize on)" if (self.normalizer and not self.use_normalize)
             else "")
          + (f"  [多 key: 按 dict 传参]" if self.multi_key else ""))

  @staticmethod
  def _resolve_normalize(normalize, ckpt_normalize=None):
    """auto 的实际语义 (见类 docstring 的"坑")。

    * orbax: 用 checkpoint 里记录的训练开关 normalize_observations
    * pkl: 默认 off (brax 无条件更新 running stats, count>0 不足以判定)
    """
    if normalize == "on":
      return True
    if normalize == "off":
      return False
    return bool(ckpt_normalize) if ckpt_normalize is not None else False

  @staticmethod
  def _preprocess(use_norm, running_statistics):
    from brax.training import types as brax_types
    return (running_statistics.normalize if use_norm
            else brax_types.identity_observation_preprocessor)

  def __call__(self, obs):
    obs = np.asarray(obs, dtype=np.float32)
    if self.multi_key:
      out = self._jit({"state": self._jnp.asarray(obs)}, self._jax.random.PRNGKey(0))
    else:
      out = self._jit(self._jnp.asarray(obs), self._jax.random.PRNGKey(0))
    return np.asarray(out[0] if isinstance(out, (tuple, list)) else out,
                      dtype=np.float64)


class DummyPolicy:
  """无策略自检: 零动作 / 顶住 home / 随机 / 正弦。"""

  def __init__(self, kind, action_size=12, home_ctrl=None, seed=0):
    self.kind = kind
    self.action_size = action_size
    self.home = (np.zeros(action_size) if home_ctrl is None
                 else np.asarray(home_ctrl, dtype=np.float64))
    self.rng = np.random.default_rng(seed)
    self.t = 0
    self.name = "dummy:" + kind

  def __call__(self, obs):
    self.t += 1
    if self.kind == "zero":
      a = np.zeros(self.action_size)
    elif self.kind == "hold_home":
      a = self.home.copy()
    elif self.kind == "random":
      a = self.rng.uniform(-1.0, 1.0, self.action_size)
    elif self.kind == "sine":
      ph = 2 * np.pi * 0.5 * self.t * CTRL_DT
      a = self.home + 0.25 * np.sin(ph + np.arange(self.action_size))
    else:
      raise ValueError(f"未知 dummy 类型 {self.kind}")
    return a


# ---------------------------------------------------------------- 仿真器
class Go2Sim:
  """CPU MuJoCo 上的 Go2 闭环环境 (obs/动作/终止逐项对齐 envs/go2_walk.py)。"""

  def __init__(self, xml_path, args):
    global mujoco
    import mujoco as _mj
    mujoco = _mj
    self.xml_path = os.path.abspath(xml_path)
    self.args = args
    m = mujoco.MjModel.from_xml_path(self.xml_path)
    m.opt.timestep = SIM_DT
    if args.solver in SOLVER_PRESETS:
      m.opt.iterations, m.opt.ls_iterations = SOLVER_PRESETS[args.solver]
    self.model = m
    self.data = mujoco.MjData(m)

    # ---- 自由关节 (base) ----
    free = np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if free.size != 1:
      raise SystemExit(f"模型里应当恰好有 1 个 freejoint, 实际 {free.size} 个")
    self.base_jid = int(free[0])
    self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
    if self.nq != 19 or self.nv != 18 or self.nu != 12:
      log(f"  [warn] 非典型 Go2 维度: nq={self.nq} nv={self.nv} nu={self.nu} "
          f"(期望 19/18/12), 按模型实际维数继续")

    # ---- 软接触足端覆写 (训练 foot_soft_contact=True 时用) ----
    self.foot_geom_ids = []
    for name in FEET:
      gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
      if gid < 0:
        raise SystemExit(
            f"模型里找不到足端 geom '{name}'。Go2 训练模型应有名为 "
            f"FR/FL/RR/RL 的足端 geom (见 models/go1/go1_mjx_position.xml)")
      self.foot_geom_ids.append(gid)
    if args.foot_contact == "soft":
      for gid in self.foot_geom_ids:
        m.geom_solimp[gid] = FOOT_SOLIMP
        m.geom_condim[gid] = FOOT_CONDIM

    # ---- 足端 site (优先 site, 退化用 geom 位置) ----
    self.foot_site_ids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name) for name in FEET]
    self.use_sites = all(i >= 0 for i in self.foot_site_ids)
    if not self.use_sites:
      log("  [warn] 模型没有 FR/FL/RR/RL 足端 site, 用足端 geom 位置代替 "
          "(对接触/滑移统计无影响, 足高会有 ~2cm 偏差)")

    # ---- 地板 geom: 世界体上的所有 geom (plane / hfield / box) ----
    self.floor_geom_ids = set(np.where(m.geom_bodyid == 0)[0].tolist())

    # ---- 执行器: position 还是 motor ----
    self.act_jnt_ids = []
    for a in range(m.nu):
      jid = int(m.actuator_trnid[a, 0])
      self.act_jnt_ids.append(jid)
    self.is_position = np.array([
        m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE for a in range(m.nu)])
    self.act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                      or f"act{a}" for a in range(m.nu)]
    self.act_joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                            or f"jnt{j}" for j in self.act_jnt_ids]
    self.ctrl_range = np.array(m.actuator_ctrlrange, dtype=np.float64)
    self.jnt_range = np.array([m.jnt_range[j] for j in self.act_jnt_ids],
                              dtype=np.float64)

    # ---- dof 顺序: 训练侧 obs 用 qpos[7:] 原序, 部署侧用电机序 ----
    self.act_joint_posadr = np.array([m.jnt_qposadr[j] for j in self.act_jnt_ids])
    self.act_joint_dofadr = np.array([m.jnt_dofadr[j] for j in self.act_jnt_ids])
    dof_posadr_model = np.arange(self.nq - 12, self.nq)       # qpos[7:19]
    ordered = np.array_equal(self.act_joint_posadr, dof_posadr_model)
    self.dof_order_consistent = bool(ordered)
    if args.dof_order == "auto":
      self.dof_order = "model" if ordered else "actuator"
    else:
      self.dof_order = args.dof_order
    if not ordered:
      log("  [warn] 本模型的关节声明顺序 != 执行器顺序 "
          f"(actuator qposadr={self.act_joint_posadr.tolist()})。")
      log("         Go2 训练模型应与 models/go1/go1_mjx_position.xml 一样把"
          "两者搞成同序, 否则 obs 的 12:24 / 24:36 段与 ctrl 段会错位。")
      log(f"         --dof-order={self.dof_order} "
          f"({'按部署电机序 (qpos 置换)' if self.dof_order == 'actuator' else '按 qpos[7:] 原序'})")
    # dof 顺序 -> 执行器顺序 的置换
    if self.dof_order == "actuator":
      perm = np.argsort(np.argsort(self.act_joint_posadr))
      # dof 序列 (actuator 序) 第 i 项对应 model qpos 的哪一项
      self.dof_from_act = np.argsort(perm)
    else:
      self.dof_from_act = None

    # ---- keyframe home ----
    self.key_qpos, self.key_ctrl = self._read_home(args)
    self.home_ctrl = self.key_ctrl.copy()
    if args.home_pose == "unitree_rl_gym":
      self.key_qpos, self.key_ctrl = self._apply_unitree_home(m, self.key_qpos,
                                                              self.key_ctrl)
      self.home_ctrl = self.key_ctrl.copy()

    # ---- hfield (地形) ----
    self.has_hfield = bool(m.nhfield > 0)
    self._setup_hfield()

    # ---- 高度扫描网格 (与 env 的 _setup_height_sampler 同序: x 外层, y 内层) ----
    gx = np.linspace(SCAN["x_min"], SCAN["x_max"], SCAN["nx"])
    gy = np.linspace(SCAN["y_min"], SCAN["y_max"], SCAN["ny"])
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    self.scan_xy = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float64)
    self.scan_n = int(SCAN["nx"] * SCAN["ny"])

    # ---- 软关节范围 (joint_limit 奖励语义): position -> ctrlrange, motor -> 关节范围
    ref = np.where(self.is_position[:, None], self.ctrl_range, self.jnt_range)
    off = 0.5 * (1 - 0.9) * (ref[:, 1] - ref[:, 0])
    self.soft_lo = ref[:, 0] + off
    self.soft_hi = ref[:, 1] - off
    # 目标 clip 边界: position 执行器用 ctrlrange, motor 执行器用关节范围
    self.tgt_lo = np.where(self.is_position, self.ctrl_range[:, 0], self.jnt_range[:, 0])
    self.tgt_hi = np.where(self.is_position, self.ctrl_range[:, 1], self.jnt_range[:, 1])

    # ---- imu site ----
    self.imu_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    if self.imu_site < 0:
      log("  [warn] 模型无 'imu' site, 用 base 机体的 xmat 代替 projected_gravity")

    # ---- 接触传感器名 (仅用于对照日志) ----
    self.foot_contact_sensor = []
    for name in FEET:
      sid = mujoco.mj_name2id(
          m, mujoco.mjtObj.mjOBJ_SENSOR, f"{name}_floor_found")
      self.foot_contact_sensor.append(sid)

    log(f"  [model] solver iterations={m.opt.iterations} ls={m.opt.ls_iterations} "
        f"ccd={m.opt.ccd_iterations} timestep={m.opt.timestep}")
    log(f"  [model] {os.path.basename(self.xml_path)}: nq={self.nq} nv={self.nv} "
        f"nu={self.nu} actuators={'position' if self.is_position.all() else ('motor' if not self.is_position.any() else 'mixed')} "
        f"hfield={'yes' if self.has_hfield else 'no'} "
        f"footsites={'yes' if self.use_sites else 'geom-fallback'}")

  # -------------------------------------------------- 初始化辅助
  def _read_home(self, args):
    m = self.model
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
      raise SystemExit("模型里没有名为 'home' 的 keyframe (env 用 keyframe('home') 取初始姿态)")
    qpos = np.array(m.key_qpos[kid], dtype=np.float64).copy()
    ctrl = np.array(m.key_ctrl[kid], dtype=np.float64).copy()
    return qpos, ctrl

  def _apply_unitree_home(self, m, qpos, ctrl):
    """把 home 换成 unitree_rl_gym 惯例姿态 (部署/实机同款)。"""
    q = qpos.copy()
    c = ctrl.copy()
    matched = 0
    for a in range(m.nu):
      jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, self.act_jnt_ids[a])
      val = unitree_home_value(jname)
      if val is not None:
        c[a] = val
        q[m.jnt_qposadr[self.act_jnt_ids[a]]] = val
        matched += 1
    if matched != m.nu:
      log(f"  [warn] unitree_rl_gym home 只匹配到 {matched}/{m.nu} 个关节, "
          f"请核对关节命名: {self.act_joint_names}")
    return q, c

  def _setup_hfield(self):
    m = self.model
    self.hf_ok = False
    if not self.has_hfield:
      return
    hid = 0
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if gid < 0 or m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_HFIELD:
      log("  [warn] 模型有 hfield 但 geom 'floor' 不是 hfield 类型, 地形采样不可用")
      return
    self.hf_nrow = int(m.hfield_nrow[hid])
    self.hf_ncol = int(m.hfield_ncol[hid])
    self.hf_size = np.array(m.hfield_size[hid], dtype=np.float64)
    self.hf_gpos = np.array(m.geom_pos[gid], dtype=np.float64)
    self.hf_gquat = np.array(m.geom_quat[gid], dtype=np.float64)
    adr = int(m.hfield_adr[hid])
    n = self.hf_nrow * self.hf_ncol
    self.hf_table = np.asarray(m.hfield_data[adr:adr + n],
                                dtype=np.float64).reshape(self.hf_nrow, self.hf_ncol)
    self.hf_rotated = not (np.allclose(self.hf_gpos, 0.0, atol=1e-9)
                           and np.allclose(self.hf_gquat, [1, 0, 0, 0], atol=1e-9))
    self.hf_ok = True
    log(f"  [hfield] {self.hf_nrow}x{self.hf_ncol} size={self.hf_size} "
        f"h∈[{self.hf_table.min() * self.hf_size[2]:.3f}, "
        f"{self.hf_table.max() * self.hf_size[2]:.3f}] m"
        f"{' (floor 位姿非平凡, 走完整旋转路径)' if self.hf_rotated else ''}")

  # -------------------------------------------------- 核心 API
  def terrain_height(self, xy):
    """世界系 (...,2) -> 地面高度 (m)。逐点复刻 go2_walk.py:670-712。"""
    xy = np.asarray(xy, dtype=np.float64)
    if not self.hf_ok:
      return np.zeros(xy.shape[:-1], dtype=np.float64)
    sz = self.hf_size
    lead = xy.shape[:-1]
    p = xy.reshape(-1, 2)
    if not self.hf_rotated:
      local = p - self.hf_gpos[:2]
    else:  # pragma: no cover - 仅当有人挪动 floor 时
      w, x, y, z = self.hf_gquat
      R = np.array([
          [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
      dif = p - self.hf_gpos[:2]
      dif3 = np.concatenate([dif, np.zeros_like(dif[:, :1])], axis=-1)
      local = dif3 @ R
      local = local[:, :2]
    u = local[:, 0] / sz[0]
    v = local[:, 1] / sz[1]
    col = (u + 1.0) * 0.5 * (self.hf_ncol - 1)
    row = (v + 1.0) * 0.5 * (self.hf_nrow - 1)
    col = np.clip(col, 0.0, self.hf_ncol - 2.0)
    row = np.clip(row, 0.0, self.hf_nrow - 2.0)
    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(col).astype(np.int64)
    fr = row - r0
    fc = col - c0
    t = self.hf_table
    h = ((1 - fr) * ((1 - fc) * t[r0, c0] + fc * t[r0, c0 + 1])
         + fr * ((1 - fc) * t[r0 + 1, c0] + fc * t[r0 + 1, c0 + 1]))
    return (h * sz[2]).reshape(lead)

  def height_scan(self, data=None, rng=None, noise=0.0):
    """35 维高度扫描特征。复刻 go2_walk.py:714-745。"""
    d = self.data if data is None else data
    base_pos = d.qpos[:3]
    q = d.qpos[3:7] / max(np.linalg.norm(d.qpos[3:7]), 1e-8)
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    xy = self.scan_xy
    xy_world = np.stack([c * xy[:, 0] - s * xy[:, 1],
                          s * xy[:, 0] + c * xy[:, 1]], axis=-1) + base_pos[None, :2]
    h = self.terrain_height(xy_world)
    if rng is not None and noise > 0.0:
      h = h + rng.uniform(-noise, noise, size=h.shape)
    d_m = base_pos[2] - h - SCAN["base"]
    return np.clip(d_m, -SCAN["clip_m"], SCAN["clip_m"]) * SCAN["scale"]

  def base_rpy(self, data=None):
    d = self.data if data is None else data
    return rpy_from_quat(d.qpos[3:7])

  def body_linvel(self, data=None):
    """机体系线速度 R^T @ qvel[:3] (frame='body' 时 obs 用)。"""
    d = self.data if data is None else data
    R = self._rot(d)
    return R.T @ np.asarray(d.qvel[:3], dtype=np.float64)

  def _rot(self, d):
    if self.imu_site >= 0 and d.site_xmat.shape[0] > self.imu_site:
      return d.site_xmat[self.imu_site].reshape(3, 3)
    return d.xmat[1].reshape(3, 3)

  def projected_gravity(self, data=None):
    d = self.data if data is None else data
    return self._rot(d).T @ np.array([0.0, 0.0, -1.0])

  def feet_pos(self, data=None):
    d = self.data if data is None else data
    if self.use_sites:
      return np.array([d.site_xpos[i] for i in self.foot_site_ids], dtype=np.float64)
    return np.array([d.geom_xpos[i] for i in self.foot_geom_ids], dtype=np.float64)

  def feet_contact(self, data=None, tol=0.0):
    """足端-地板接触 (等价于训练侧的 {f}_floor_found 传感器: dist<=0 记为 found)。"""
    d = self.data if data is None else data
    out = [False] * 4
    for i in range(int(d.ncon)):
      con = d.contact[i]
      g1, g2 = int(con.geom1), int(con.geom2)
      if con.dist > tol:
        continue
      for f, gid in enumerate(self.foot_geom_ids):
        if (g1 == gid and g2 in self.floor_geom_ids) or (
            g2 == gid and g1 in self.floor_geom_ids):
          out[f] = True
    return np.array(out, dtype=bool)

  # -------------------------------------------------- reset / step
  def reset(self, seed, cmd, spawn):
    """复刻 env.reset: home keyframe + U(-0.1,0.1) 关节噪声, ctrl~N(home_ctrl,0.1)。"""
    m, d = self.model, self.data
    rng = np.random.default_rng(seed)
    qpos = self.key_qpos.copy()
    noise = float(self.args.reset_noise)
    qpos = qpos + rng.uniform(-noise, noise, size=self.nq)
    # env: ctrl = key_ctrl + reset_noise_scale * N(0,1)  (go2_walk.py:800-801)
    ctrl = self.key_ctrl + noise * rng.normal(size=self.nu)

    if spawn == "origin":
      x0, y0 = 0.0, 0.0
      h0 = float(self.terrain_height(np.zeros((1, 2)))[0]) if self.hf_ok else 0.0
      qpos[0], qpos[1] = x0, y0
      qpos[2] = h0 + self.args.spawn_z
      qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    elif spawn == "random":
      r = self.args.spawn_range
      x0, y0 = rng.uniform(-r, r, size=2)
      yaw = rng.uniform(-np.pi, np.pi)
      h0 = float(self.terrain_height(np.array([[x0, y0]]))[0]) if self.hf_ok else 0.0
      qpos[0], qpos[1] = x0, y0
      qpos[2] = h0 + self.args.spawn_z
      qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    # spawn == "home": 就用 keyframe 的 x/y/z + 噪声

    qvel = np.zeros(self.nv)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)

    return dict(cmd=np.asarray(cmd, dtype=np.float64),
                phase=TROT_PHASE0 + rng.uniform(0.0, 2 * np.pi),
                last_act=np.zeros(self.nu), rng=rng,
                x0=float(d.qpos[0]), y0=float(d.qpos[1]))

  def obs(self, st, use_scan, use_phase):
    """48 [+35] [+8] 维 actor obs, 与 go2_walk.py:1070-1162 逐项对齐。"""
    d = self.data
    if self.args.vel_frame == "body":
      lin = self.body_linvel(d) * SC_LIN
    else:
      lin = np.asarray(d.qvel[:3], dtype=np.float64) * SC_LIN
    ang = np.asarray(d.qvel[3:6], dtype=np.float64) * SC_ANG
    grav = self.projected_gravity(d)
    # dof 段 (顺序见 --dof-order)
    if self.dof_order == "actuator":
      qact = np.asarray([d.qpos[a] for a in self.act_joint_posadr], dtype=np.float64)
      dqact = np.asarray([d.qvel[a] for a in self.act_joint_dofadr], dtype=np.float64)
    else:
      qact = np.asarray(d.qpos[self.nq - 12:], dtype=np.float64)
      dqact = np.asarray(d.qvel[self.nv - 12:], dtype=np.float64)
    dofs_pos = qact - self.home_ctrl
    dofs_vel = dqact                      # **不乘 0.05** (见文件头"已知坑")
    base = np.concatenate([lin, ang, grav, st["cmd"] * SC_LIN,
                           dofs_pos, dofs_vel, st["last_act"]])
    parts = [base]
    if use_scan:
      parts.append(self.height_scan(d))
    if use_phase:
      ph = st["phase"]
      parts.append(np.concatenate([np.cos(ph), np.sin(ph)]))
    obs = np.concatenate(parts)
    return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

  def advance_phase(self, st):
    gf = GAIT_FREQ
    if GAIT_FREQ_SCALED:
      spd = float(np.linalg.norm(st["cmd"][:2]))
      gf = max(GAIT_FREQ * spd / 0.5, GAIT_FREQ_MIN)
    st["phase"] = (st["phase"] + 2 * np.pi * gf * CTRL_DT + np.pi) % (
        2 * np.pi) - np.pi

  def apply_action(self, action):
    """策略输出 -> ctrl。返回 (applied, n_clamped)。

    * position 执行器: ctrl = clip(a, 范围)  (env.step 原语义)
    * motor 执行器: tau = clip(kp*(q_des-q) - kd*qd, ctrlrange)  (部署侧 LowCmd 语义)
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)[:self.nu]
    mode = self.args.action_mode
    if mode == "tanh_soft":
      u = self.soft_lo + (np.clip(a, -1.0, 1.0) + 1.0) * 0.5 * (self.soft_hi - self.soft_lo)
    elif mode == "soft":
      u = np.clip(a, self.soft_lo, self.soft_hi)
    else:
      u = np.clip(a, self.tgt_lo, self.tgt_hi)
    # 目标顶在关节(位置)范围边界上的分量数 = "动作饱和"
    n_clamped = int(np.count_nonzero(
        (u <= self.tgt_lo + 1e-9) | (u >= self.tgt_hi - 1e-9)))
    if self.is_position.all():
      self.data.ctrl[:] = u
    else:
      q = np.array([self.data.qpos[j] for j in self.act_joint_posadr])
      dq = np.array([self.data.qvel[j] for j in self.act_joint_dofadr])
      tau = self.args.kp * (u - q) - self.args.kd * dq
      self.data.ctrl[:] = np.clip(tau, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
    return u, n_clamped


# ---------------------------------------------------------------- 一轮 episode
def fall_check(sim, args, h_under_base):
  """返回 (fell_env, fell_tilt, reason)。"""
  d = sim.data
  if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
    return True, True, "nonfinite"
  zmin, zmax = args.healthy_z_min, args.healthy_z_max
  z_rel = float(d.qpos[2]) - float(h_under_base)
  q = np.asarray(d.qpos[3:7], dtype=np.float64)
  q = q / max(np.linalg.norm(q), 1e-9)
  rad = np.deg2rad(args.healthy_qxy_deg)
  fell_env = (z_rel <= zmin) or (z_rel >= zmax) or \
      (abs(q[1]) >= rad) or (abs(q[2]) >= rad)
  roll, pitch, _ = rpy_from_quat(q)
  tilt = np.rad2deg(max(abs(roll), abs(pitch)))
  fell_tilt = (z_rel <= args.fall_z_min) or (z_rel >= zmax) or \
      (tilt >= args.fall_tilt_deg)
  reason = ""
  if fell_env:
    reason = ("z_rel=%.3f" % z_rel) if (z_rel <= zmin or z_rel >= zmax) \
        else ("quat_xy=%.3f/%.3f rad" % (abs(q[1]), abs(q[2])))
  return bool(fell_env), bool(fell_tilt), reason


def run_episode(sim, policy, args, seed, use_scan, use_phase, tag=""):
  st = sim.reset(seed, np.array(args.cmd, dtype=np.float64), args.spawn)
  n_steps = int(round(args.duration / CTRL_DT))
  rec = {k: [] for k in ("t", "qpos", "qvel", "ctrl", "action", "tau", "contact",
                         "feet_xy", "feet_z", "ground_h", "z_rel", "roll",
                         "pitch", "yaw", "v_body", "clamped", "q_act", "dq_act")}
  fall_env = fall_tilt = False
  fall_step = n_steps
  fall_reason = ""
  for i in range(n_steps):
    obs = sim.obs(st, use_scan, use_phase)
    action = np.asarray(policy(obs), dtype=np.float64).reshape(-1)[:sim.nu]
    applied, n_clamped = sim.apply_action(action)
    mujoco.mj_step(sim.model, sim.data, N_SUBSTEPS)
    sim.advance_phase(st)
    st["last_act"] = action.copy()

    d = sim.data
    feet = sim.feet_pos(d)
    ground = float(sim.terrain_height(d.qpos[None, :2])[0]) if sim.hf_ok else 0.0
    roll, pitch, yaw = rpy_from_quat(d.qpos[3:7])
    rec["t"].append((i + 1) * CTRL_DT)
    rec["qpos"].append(np.array(d.qpos, dtype=np.float64))
    rec["qvel"].append(np.array(d.qvel, dtype=np.float64))
    rec["ctrl"].append(applied.copy())
    rec["action"].append(action.copy())
    rec["tau"].append(np.array(d.actuator_force, dtype=np.float64))
    rec["contact"].append(sim.feet_contact(d))
    rec["feet_xy"].append(feet[:, :2].copy())
    rec["feet_z"].append(feet[:, 2].copy())
    rec["ground_h"].append(ground)
    rec["z_rel"].append(float(d.qpos[2]) - ground)
    rec["roll"].append(roll)
    rec["pitch"].append(pitch)
    rec["yaw"].append(yaw)
    rec["v_body"].append(sim.body_linvel(d))
    rec["clamped"].append(n_clamped)
    # 关节状态 (执行器序, 与 ctrl 对齐; 用于限位饱和统计)
    rec["q_act"].append(np.array(
        [d.qpos[a] for a in sim.act_joint_posadr], dtype=np.float64))
    rec["dq_act"].append(np.array(
        [d.qvel[a] for a in sim.act_joint_dofadr], dtype=np.float64))

    fe, ft, reason = fall_check(sim, args, ground)
    if fe or ft:
      fall_env, fall_tilt = fe, ft
      fall_step = i + 1
      fall_reason = reason
      break
  tr = {k: np.asarray(v) for k, v in rec.items()}
  tr["seed"] = seed
  tr["fell_env"] = fall_env
  tr["fell_tilt"] = fall_tilt
  tr["fall_step"] = fall_step
  tr["fall_reason"] = fall_reason
  tr["n_steps"] = len(tr["t"])
  tr["use_scan"] = use_scan
  tr["use_phase"] = use_phase
  tr["tag"] = tag
  tr["soft_lo"] = sim.soft_lo
  tr["soft_hi"] = sim.soft_hi
  tr["tgt_lo"] = sim.tgt_lo
  tr["tgt_hi"] = sim.tgt_hi
  tr["act_names"] = sim.act_names
  tr["joint_names"] = sim.act_joint_names
  return tr


# ---------------------------------------------------------------- 指标
def episode_metrics(tr, args):
  n = tr["n_steps"]
  if n == 0:
    return {}
  cmd = np.asarray(args.cmd, dtype=np.float64)
  v = tr["v_body"]                                     # (n,3) 机体系
  dt = CTRL_DT
  warm = int(round(args.warmup / dt))
  sl = slice(min(warm, n - 1), n)
  vx, vy, wz = v[:, 0], v[:, 1], v[:, 2]
  err = np.linalg.norm(v[:, :2] - cmd[None, :2], axis=-1)
  z = tr["z_rel"]
  roll, pitch = tr["roll"], tr["pitch"]
  contact = tr["contact"]                              # (n,4) bool
  duty = contact.mean(axis=0) if n else np.zeros(4)
  d1 = contact[:, 0] & contact[:, 3]                  # FR + RL
  d2 = contact[:, 1] & contact[:, 2]                  # FL + RR
  trot_frac = float(np.mean(d1 != d2))
  # 足端滑移 (接触时的水平速度)
  slip = []
  for i in range(1, n):
    for f in range(4):
      if contact[i, f]:
        slip.append(np.linalg.norm(tr["feet_xy"][i, f] - tr["feet_xy"][i - 1, f]) / dt)
  # 关节限位: 软范围越界比例 (joint_limit 奖励同语义) + 目标饱和比例
  qa = tr["q_act"]                       # (n,12) 执行器序
  lo, hi = tr["soft_lo"], tr["soft_hi"]
  tlo, thi = tr["tgt_lo"], tr["tgt_hi"]
  over = np.clip(lo - qa, 0.0, None) + np.clip(qa - hi, 0.0, None)
  joint_over_frac = float(np.mean(over > 1e-6))
  joint_over_max = float(over.max())
  tgt_sat = (tr["ctrl"] <= tlo + 1e-9) | (tr["ctrl"] >= thi - 1e-9)
  tgt_sat_frac = float(np.mean(tgt_sat))
  clamped_steps = float(np.mean(tr["clamped"] > 0)) if n else 0.0
  m = dict(
      seed=int(tr["seed"]),
      n_steps=int(n),
      duration_s=float(n * dt),
      fell_env=bool(tr["fell_env"]),
      fell_tilt=bool(tr["fell_tilt"]),
      fall_step=int(tr["fall_step"]),
      fall_time_s=float(tr["fall_step"] * dt),
      fall_reason=str(tr["fall_reason"]),
      fall_x=float(tr["qpos"][-1, 0]) if n else float("nan"),
      fall_y=float(tr["qpos"][-1, 1]) if n else float("nan"),
      fall_ground_h=float(tr["ground_h"][-1]) if n else float("nan"),
      # 速度
      vx_mean=float(vx.mean()), vx_std=float(vx.std()),
      vy_mean=float(vy.mean()), vy_std=float(vy.std()),
      wz_mean=float(wz.mean()), wz_std=float(wz.std()),
      vx_ss_mean=float(vx[sl].mean()), vx_ss_std=float(vx[sl].std()),
      vy_ss_mean=float(vy[sl].mean()), wz_ss_mean=float(wz[sl].mean()),
      track_err=float(err.mean()), track_err_ss=float(err[sl].mean()),
      # 姿态
      z_rel_mean=float(z.mean()), z_rel_min=float(z.min()), z_rel_max=float(z.max()),
      z_rel_ss_mean=float(z[sl].mean()),
      roll_mean_deg=float(np.rad2deg(roll.mean())),
      roll_max_deg=float(np.rad2deg(np.abs(roll).max())),
      pitch_mean_deg=float(np.rad2deg(pitch.mean())),
      pitch_max_deg=float(np.rad2deg(np.abs(pitch).max())),
      angvel_rms=float(np.sqrt((v[:, :] ** 2 * 0 + tr["qvel"][:, 3:6] ** 2).sum(axis=-1).mean())),
      # 步态
      duty=[float(x) for x in duty],
      trot_frac=trot_frac,
      slip_speed=float(np.mean(slip)) if slip else 0.0,
      n_contact_frames=int(contact.any(axis=1).sum()),
      # 关节
      tau_abs_mean=float(np.abs(tr["tau"]).mean()),
      tau_abs_max=float(np.abs(tr["tau"]).max()),
      clamp_frac=clamped_steps,
      joint_over_frac=joint_over_frac,
      joint_over_max=joint_over_max,
      tgt_sat_frac=tgt_sat_frac,
      # 位移
      dist_forward=float(tr["qpos"][-1, 0] - tr["qpos"][0, 0]) if n else 0.0,
      dist_path=float(np.linalg.norm(np.diff(tr["qpos"][:, :2], axis=0), axis=-1).sum())
      if n > 1 else 0.0,
      ground_h_max=float(tr["ground_h"].max()) if n else 0.0,
  )
  m["is_trot"] = bool(trot_frac > 0.5 and all(abs(d - 0.5) < 0.15 for d in m["duty"]))
  m["steady"] = bool((not tr["fell_env"]) and m["vx_ss_std"] < args.steady_std
                     and m["duration_s"] >= args.duration - 1e-6)
  return m


def aggregate(metrics, args):
  if not metrics:
    return {}
  def arr(k):
    return np.array([m[k] for m in metrics], dtype=np.float64)
  surv_env = 1.0 - arr("fell_env").mean()
  surv_tilt = 1.0 - arr("fell_tilt").mean()
  falls = arr("fall_step")
  m = dict(
      n_seeds=len(metrics),
      cmd=[float(x) for x in args.cmd],
      terrain=args.terrain,
      policy=os.path.basename(str(args.policy)),
      action_mode=args.action_mode,
      vel_frame=args.vel_frame,
      duration_s=float(args.duration),
      # 1. 速度跟踪
      vx_mean=float(arr("vx_mean").mean()),
      vx_std_across=float(arr("vx_mean").std()),
      vx_std_within=float(arr("vx_std").mean()),
      vy_mean=float(arr("vy_mean").mean()),
      wz_mean=float(arr("wz_mean").mean()),
      track_err=float(arr("track_err").mean()),
      track_err_ss=float(arr("track_err_ss").mean()),
      vx_ss_mean=float(arr("vx_ss_mean").mean()),
      vx_ss_std=float(arr("vx_ss_std").mean()),
      steady_rate=float(arr("steady").mean()),
      # 2. 存活
      survive_rate_env=float(surv_env),
      survive_rate_tilt=float(surv_tilt),
      fall_rate=float(arr("fell_env").mean()),
      fall_time_mean_s=float((falls * CTRL_DT).mean()),
      fall_time_p50_s=float(np.percentile(falls * CTRL_DT, 50)),
      # 3. 姿态
      z_rel_mean=float(arr("z_rel_mean").mean()),
      z_rel_min=float(arr("z_rel_min").min()),
      roll_mean_deg=float(arr("roll_mean_deg").mean()),
      roll_max_deg=float(arr("roll_max_deg").max()),
      pitch_mean_deg=float(arr("pitch_mean_deg").mean()),
      pitch_max_deg=float(arr("pitch_max_deg").max()),
      angvel_rms=float(arr("angvel_rms").mean()),
      # 4. 步态
      duty_mean=[float(x) for x in np.array([m["duty"] for m in metrics]).mean(axis=0)],
      trot_frac=float(arr("trot_frac").mean()),
      is_trot_frac=float(arr("is_trot").mean()),
      slip_speed=float(arr("slip_speed").mean()),
      tau_abs_mean=float(arr("tau_abs_mean").mean()),
      tau_abs_max=float(arr("tau_abs_max").max()),
      clamp_frac=float(arr("clamp_frac").mean()),
      joint_over_frac=float(arr("joint_over_frac").mean()),
      joint_over_max=float(arr("joint_over_max").max()),
      tgt_sat_frac=float(arr("tgt_sat_frac").mean()),
      # 5. 位移/地形
      dist_forward_mean=float(arr("dist_forward").mean()),
      dist_forward_std=float(arr("dist_forward").std()),
      dist_path_mean=float(arr("dist_path").mean()),
      ground_h_max_mean=float(arr("ground_h_max").mean()),
  )
  m["episodes"] = metrics
  return m


# ---------------------------------------------------------------- 地形分层
def terrain_report(all_tr, args):
  """按"躯干下方地形高度"分层统计通过率 + 摔倒位置分布。"""
  if not args.terrain_bins:
    edges = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
  else:
    edges = np.array([float(x) for x in args.terrain_bins.split(",")])
  rows = []
  for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i + 1]
    reached = 0
    survived = 0
    for tr in all_tr:
      h = tr["ground_h"]
      if h.size == 0:
        continue
      if h.max() >= lo:
        reached += 1
        # 在该档内是否摔倒: 摔倒时刻的 ground_h 落在本档
        if not tr["fell_env"]:
          survived += 1
        else:
          fh = float(tr["fall_ground_h"]) if "fall_ground_h" in tr else float(h[-1])
          if not (lo <= fh < hi):
            survived += 1
    rows.append(dict(bin_lo_m=float(lo), bin_hi_m=float(hi),
                     n_reached=int(reached),
                     n_survived=int(survived),
                     pass_rate=(float(survived) / reached) if reached else None))
  fall_xy = [[float(tr["qpos"][-1, 0]), float(tr["qpos"][-1, 1])]
             for tr in all_tr if tr["fell_env"]]
  fall_h = [float(tr["ground_h"][-1]) for tr in all_tr if tr["fell_env"]]
  return dict(bins=rows, fall_xy=fall_xy, fall_ground_h=fall_h,
              n_fell=int(len(fall_xy)))


# ---------------------------------------------------------------- 出图
def setup_cjk_font():
  """让图里的中文能正常显示 (本机有 Noto Sans CJK)。找不到就退回默认字体并提示。"""
  import matplotlib
  from matplotlib import font_manager
  for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC",
               "WenQuanYi Zen Hei", "Source Han Sans SC", "SimHei"):
    try:
      font_manager.findfont(name, fallback_to_default=False)
    except Exception:
      continue
    matplotlib.rcParams["font.sans-serif"] = [name] + list(
        matplotlib.rcParams["font.sans-serif"])
    matplotlib.rcParams["axes.unicode_minus"] = False
    return name
  return None


def make_plot(all_tr, agg, args, out_png):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  if setup_cjk_font() is None:
    log("  [warn] 没有可用的中文字体, PNG 里的中文会显示成方框 (数值看 json)")
  act_labels = list(all_tr[0].get("act_names", [f"act{i}" for i in range(12)])) \
      if all_tr else [f"act{i}" for i in range(12)]
  fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
  fig.suptitle(
      f"Go2 eval | {os.path.basename(str(args.policy))} | {args.terrain} | "
      f"cmd=({args.cmd[0]:.2f},{args.cmd[1]:.2f},{args.cmd[2]:.2f}) | "
      f"n={len(all_tr)} seeds x {args.duration:.0f}s", fontsize=11)

  tmax = max(tr["t"][-1] if tr["n_steps"] else 0.0 for tr in all_tr)
  grid = np.arange(0, tmax + 1e-9, CTRL_DT)

  def resample(tr, key):
    y = tr[key]
    if y.ndim == 1:
      return np.interp(grid, tr["t"], y)
    cols = [np.interp(grid, tr["t"], y[:, j]) for j in range(y.shape[1])]
    return np.stack(cols, axis=-1)

  # (a) 速度跟踪
  a = ax[0, 0]
  vx_all = np.stack([resample(tr, "v_body")[:, 0] for tr in all_tr]) if all_tr else None
  if vx_all is not None:
    a.plot(grid, vx_all.mean(axis=0), color="C0", label="vx (mean)")
    a.fill_between(grid, vx_all.min(axis=0), vx_all.max(axis=0), color="C0", alpha=0.15,
                   label="min~max")
  a.axhline(args.cmd[0], ls="--", color="k", label=f"cmd vx={args.cmd[0]:.2f}")
  a.set_xlabel("t (s)")
  a.set_ylabel("body vx (m/s)")
  a.set_title(f"(a) 速度跟踪  mean={agg.get('vx_mean', float('nan')):.3f} "
              f"err={agg.get('track_err', float('nan')):.3f} m/s")
  a.grid(alpha=0.3)
  a.legend(fontsize=8)

  # (b) 躯干高度 + 足下地形高度
  b = ax[0, 1]
  if all_tr:
    zs = np.stack([resample(tr, "z_rel") for tr in all_tr])
    gh = np.stack([resample(tr, "ground_h") for tr in all_tr])
    b.plot(grid, zs.mean(axis=0), color="C1", label="躯干相对地面高度 z_rel")
    b.fill_between(grid, zs.min(axis=0), zs.max(axis=0), color="C1", alpha=0.15)
    if gh.max() > 1e-6:
      b.plot(grid, gh.mean(axis=0), color="C2", ls=":", label="足下地形高度 h_ground")
    b.axhline(args.healthy_z_min, ls="--", color="r", lw=1,
              label=f"env z_min {args.healthy_z_min}")
  b.set_xlabel("t (s)")
  b.set_ylabel("z (m)")
  b.set_title(f"(b) 躯干高度  mean={agg.get('z_rel_mean', float('nan')):.3f} m "
              f"min={agg.get('z_rel_min', float('nan')):.3f}")
  b.grid(alpha=0.3)
  b.legend(fontsize=8)

  # (c) 每执行器平均 |力矩|
  c = ax[1, 0]
  tau = np.stack([np.abs(tr["tau"]).mean(axis=0) for tr in all_tr]) if all_tr else None
  if tau is not None:
    idx = np.arange(12)
    c.bar(idx, tau.mean(axis=0), yerr=tau.std(axis=0), color="C3", alpha=0.8)
    c.set_xticks(idx)
    c.set_xticklabels([n.replace("_joint", "") for n in act_labels], fontsize=6,
                      rotation=90)
    c.set_xlabel("执行器序号 (0..11)")
    c.set_ylabel("|tau| (N·m)")
  c.set_title(f"(c) 关节力矩  mean|tau|={agg.get('tau_abs_mean', float('nan')):.2f} "
              f"max={agg.get('tau_abs_max', float('nan')):.1f} N·m")
  c.grid(alpha=0.3, axis="y")

  # (d) 前 3 个执行器力矩时间曲线
  d = ax[1, 1]
  if all_tr:
    tt = np.stack([resample(tr, "tau") for tr in all_tr])
    for j in range(min(3, tt.shape[-1])):
      d.plot(grid, tt[:, :, j].mean(axis=0), label=f"act {j}")
      d.fill_between(grid, tt[:, :, j].min(axis=0), tt[:, :, j].max(axis=0), alpha=0.12)
  d.set_xlabel("t (s)")
  d.set_ylabel("tau (N·m)")
  d.set_title("(d) 力矩时间曲线 (执行器 0-2)")
  d.grid(alpha=0.3)
  d.legend(fontsize=8)

  fig.tight_layout(rect=[0, 0, 1, 0.96])
  fig.savefig(out_png, dpi=110)
  plt.close(fig)
  return out_png


# ---------------------------------------------------------------- 主流程
def pick_xml(args):
  if args.xml:
    p = os.path.expanduser(args.xml)
    if not os.path.exists(p):
      raise SystemExit(f"--xml 不存在: {p}")
    return p
  name = {"flat": "scene_mjx_position.xml",
          "terrain": "scene_mjx_terrain.xml"}[args.terrain]
  cands = [os.path.join(_ROOT, "models", "go2", name)]
  if args.terrain == "flat":
    cands.append(os.path.join(_DEPLOY_GO2, "scene_flat.xml"))
  for c in cands:
    if os.path.exists(c):
      return c
  raise SystemExit(
      f"找不到 {name}。候选: {cands}\n"
      f"  Go2 训练模型 (models/go2/{name}) 还没生成时, 平地上可用 "
      f"--xml {os.path.join(_DEPLOY_GO2, 'scene_flat.xml')} 先跑通链路。")


def build_policy(args, sim):
  if args.policy in (None, "", "none", "dummy"):
    log(f"  [policy] 无策略自检模式 --dummy {args.dummy}")
    return DummyPolicy(args.dummy, action_size=sim.nu, home_ctrl=sim.home_ctrl,
                       seed=args.seed0)
  if not os.path.exists(args.policy):
    raise SystemExit(f"--policy 不存在: {args.policy}")
  return Policy(args.policy, action_size=sim.nu, normalize=args.normalize,
                dist=args.dist)


def obs_plan(ns, args, sim):
  """由策略 obs 维度反推组件: 48 + 35*scan + 8*phase。"""
  if args.policy in (None, "", "none", "dummy"):
    scan = {"auto": False, "on": True, "off": False}[args.height_scan]
    phase = {"auto": False, "on": True, "off": False}[args.phase]
    if scan and not sim.hf_ok:
      raise SystemExit("--height-scan on 但模型没有 hfield 地形")
    return scan, phase, 48 + int(SCAN["nx"] * SCAN["ny"]) * int(scan) + 8 * int(phase)
  n_scan = int(SCAN["nx"] * SCAN["ny"])
  cand = []
  for s in (0, 1):
    for p in (0, 1):
      cand.append((bool(s), bool(p), 48 + n_scan * s + 8 * p))
  hits = [c for c in cand if c[2] == ns]
  if not hits:
    raise SystemExit(
        f"策略 obs 维 {ns} 与训练布局不匹配 "
        f"(48/{48 + n_scan}/{48 + 8}/{48 + n_scan + 8}, 扫描 {SCAN['nx']}x{SCAN['ny']})。"
        f"该策略可能来自别的 obs 配置, 请核对 envs/go2_walk.py。")
  scan, phase, _ = hits[0]
  if args.height_scan == "on":
    scan = True
  elif args.height_scan == "off":
    scan = False
  if args.phase == "on":
    phase = True
  elif args.phase == "off":
    phase = False
  if scan and not sim.hf_ok:
    log(f"  [warn] 策略需要 {n_scan} 维高度扫描, 但当前模型没有 hfield; "
        "按部署侧 obs.py 的约定喂 0 (平地上该特征本就≈0) —— "
        "要在地形上评测请用 models/go2/scene_mjx_terrain.xml")
  dim = 48 + n_scan * int(scan) + 8 * int(phase)
  if dim != ns:
    raise SystemExit(f"obs 组装维度 {dim} != 策略需要的 {ns}, 组件组合无法匹配")
  return scan, phase, dim


def build_parser():
  """命令行解析器 (单独抽出来, 方便测试/probe 直接构造同名参数)。"""
  ap = argparse.ArgumentParser(
      description="Go2 策略评测 (CPU MuJoCo 闭环)",
      formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--policy", default="none",
                  help="策略来源: pkl 文件 / orbax checkpoint 目录 / none(自检)")
  ap.add_argument("--dummy", default="hold_home",
                  choices=["zero", "hold_home", "random", "sine"],
                  help="--policy none 时的动作来源")
  ap.add_argument("--terrain", default="flat", choices=["flat", "terrain"],
                  help="平地 / hfield 地形 (决定用哪个 XML)")
  ap.add_argument("--xml", default=None, help="直接指定场景 XML (优先于 --terrain)")
  ap.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                  metavar=("VX", "VY", "WZ"), help="速度指令 (机体系, m/s, rad/s)")
  ap.add_argument("--seeds", type=int, default=20, help="episode 数 (每个 seed 一轮)")
  ap.add_argument("--duration", type=float, default=15.0, help="每轮时长 (s)")
  ap.add_argument("--out", default=os.path.join(_ROOT, "logs", "go2_eval"),
                  help="输出目录")
  ap.add_argument("--tag", default=None, help="输出文件名前缀 (默认自动)")
  ap.add_argument("--seed0", type=int, default=0, help="第一个 seed (种子集 = seed0..seed0+N-1)")
  ap.add_argument("--vel-frame", default="body", choices=["body", "world"],
                  help="obs 里 base 线速度的坐标系 (go2_walk 默认 body; go1_walk 是 world)")
  ap.add_argument("--action-mode", default="auto",
                  choices=["auto", "clip", "soft", "tanh_soft"],
                  help="策略输出 -> ctrl 目标: clip=clip(a,范围) (env 原语义); "
                       "soft=内缩百分之五后 clip; tanh_soft=[-1,1]->软范围线性映射")
  ap.add_argument("--dof-order", default="auto", choices=["auto", "model", "actuator"],
                  help="obs 关节段的顺序 (model=qpos[7:] 原序, 同 env; "
                       "actuator=按执行器序/部署电机序)")
  ap.add_argument("--home-pose", default="keyframe",
                  choices=["keyframe", "unitree_rl_gym"],
                  help="初始姿态: keyframe=模型 home 键帧; unitree_rl_gym=官方惯例")
  ap.add_argument("--spawn", default="auto", choices=["auto", "home", "origin", "random"],
                  help="出生点: origin=原点朝 +x (地形评测), home=键帧位置")
  ap.add_argument("--spawn-z", type=float, default=0.32,
                  help="origin/random 出生的躯干离地高度 (go2_walk.spawn_z_offset=0.32)")

  ap.add_argument("--spawn-range", type=float, default=3.0, help="random 出生的 xy 半宽")
  ap.add_argument("--reset-noise", type=float, default=0.1,
                  help="初始 qpos 均匀噪声半宽 (env reset_noise_scale=0.1)")
  ap.add_argument("--solver", default="cpu", choices=["cpu", "xml", "soft", "env"],
                  help="求解器档位: cpu=100/50 CPU 完整 Newton (默认); xml=模型自带; "
                       "env=20/10 (go2_walk 硬接触档); soft=2/5 (go2_walk MJX 软接触档)")
  ap.add_argument("--foot-contact", default="xml", choices=["xml", "soft"],
                  help="足端接触: xml=保持 XML 原样; soft=强制 MJX 软接触 "
                       "(solimp 0.9/0.95/0.023/0.5/2.0 + condim=3)")
  ap.add_argument("--height-scan", default="auto", choices=["auto", "on", "off"])
  ap.add_argument("--phase", default="auto", choices=["auto", "on", "off"])
  ap.add_argument("--scan_nx", type=int, default=None, help="覆盖高度扫描前向点数 (默认 7)")
  ap.add_argument("--scan_ny", type=int, default=None, help="覆盖高度扫描侧向点数 (默认 5)")
  ap.add_argument("--scan_x", type=float, nargs=2, default=None,
                  help="覆盖高度扫描前向范围 x_min x_max (默认 -0.3 0.9)")
  ap.add_argument("--scan_y", type=float, nargs=2, default=None,
                  help="覆盖高度扫描侧向范围 y_min y_max (默认 -0.4 0.4)")
  ap.add_argument("--normalize", default="auto", choices=["auto", "on", "off"],
                  help="brax obs 归一化: auto = orbax 用 checkpoint 记录的 "
                       "normalize_observations, pkl 默认 off (brax 无条件更新 "
                       "running stats, count>0 不足以判定; 见 docs §7 坑 4)")
  ap.add_argument("--dist", default="auto", choices=["auto", "normal", "tanh_normal"])
  ap.add_argument("--kp", type=float, default=20.0, help="motor 执行器的 PD 比例增益")
  ap.add_argument("--kd", type=float, default=1.0, help="motor 执行器的 PD 微分增益")
  ap.add_argument("--warmup", type=float, default=2.0, help="稳态统计的起始时间 (s)")
  ap.add_argument("--steady-std", type=float, default=0.15,
                  help="判稳态的 vx 标准差阈值 (m/s)")
  ap.add_argument("--healthy-z-min", type=float, default=HEALTHY_Z[0],
                  help="env 健康判据 z_rel 下界 (Go2 0.19, Go1 0.22)")
  ap.add_argument("--healthy-z-max", type=float, default=HEALTHY_Z[1],
                  help="env 健康判据 z_rel 上界 (0.65)")
  ap.add_argument("--healthy-qxy-deg", type=float, default=HEALTHY_QXY_DEG,
                  help="env 健康判据的四元数 x/y 分量阈值 (deg; Go2 训练 10, "
                       "eval_walk 的测试环境用 17.5 约等于真欧拉 35 度)")
  ap.add_argument("--fall-mode", default="env", choices=["env", "tilt"],
                  help="存活判据: env=复刻训练终止判据; tilt=真 roll/pitch 阈值")
  ap.add_argument("--fall-tilt-deg", type=float, default=30.0,
                  help="tilt 判据的倾角阈值 (deg)")
  ap.add_argument("--fall-z-min", type=float, default=0.12,
                  help="tilt 判据的 z_rel 下界 (m)")
  ap.add_argument("--terrain-bins", default=None,
                  help="地形分层边界 (逗号分隔, 米), 默认 0,0.05,0.1,0.15,0.2,0.3,0.5")
  ap.add_argument("--jax-platform", default="cpu", choices=["cpu", "gpu", "default"],
                  help="策略推理后端 (默认 cpu, 不占 GPU)")
  ap.add_argument("--dump-trace", action="store_true", help="额外保存 npz 轨迹")
  ap.add_argument("--no-plot", action="store_true")
  ap.add_argument("--quiet", action="store_true")
  return ap


def finalize_args(args):
  """把 auto 选项落成具体值 (供 main 与外部调用共用)。"""
  if args.jax_platform in ("cpu", "default"):
    os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
  elif args.jax_platform == "gpu":
    os.environ.pop("JAX_PLATFORM_NAME", None)
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

  if args.spawn == "auto":
    args.spawn = "origin" if args.terrain == "terrain" else "home"
  if args.action_mode == "auto":
    args.action_mode = "clip"     # env.step 原语义; 若是 tanh 策略会自动提示
  return args


def main():
  args = finalize_args(build_parser().parse_args())

  xml = pick_xml(args)
  os.makedirs(args.out, exist_ok=True)
  stamp = time.strftime("%Y%m%d-%H%M%S")
  tag = args.tag or (f"{os.path.splitext(os.path.basename(str(args.policy)))[0]}"
                     f"_{args.terrain}_cmd{args.cmd[0]:+.2f}_{stamp}")
  json_path = os.path.join(args.out, tag + ".json")
  png_path = os.path.join(args.out, tag + ".png")

  log("=" * 78)
  log(f"Go2 评测: policy={args.policy}  terrain={args.terrain}  xml={xml}")
  log(f"  cmd={args.cmd}  seeds={args.seeds}  duration={args.duration}s  "
      f"spawn={args.spawn}  vel_frame={args.vel_frame}")
  log("=" * 78)

  t0 = time.time()
  # v5: 高度扫描几何覆盖 (必须在 Go2Sim 构造前, 它用模块级 SCAN 建网格)
  if args.scan_nx is not None:
    SCAN["nx"] = args.scan_nx
  if args.scan_ny is not None:
    SCAN["ny"] = args.scan_ny
  if args.scan_x is not None:
    SCAN["x_min"], SCAN["x_max"] = args.scan_x
  if args.scan_y is not None:
    SCAN["y_min"], SCAN["y_max"] = args.scan_y
  if any(a is not None for a in (args.scan_nx, args.scan_ny, args.scan_x, args.scan_y)):
    print(f"[scan] 覆盖高度扫描: {SCAN['nx']}x{SCAN['ny']}={SCAN['nx']*SCAN['ny']} 点, "
          f"x[{SCAN['x_min']},{SCAN['x_max']}] y[{SCAN['y_min']},{SCAN['y_max']}]")
  sim = Go2Sim(xml, args)
  policy = build_policy(args, sim)
  if isinstance(policy, Policy) and getattr(policy, "dist", "") == "tanh_normal" \
      and args.action_mode == "clip":
    log("  [warn] 策略是 tanh_normal (输出 [-1,1]), 但 --action-mode=clip 会把 "
        "[-1,1] 当成绝对位置目标喂给关节; 这**正是** env.step 的行为, "
        "但若本意是线性映射请用 --action-mode tanh_soft")
  ns = getattr(policy, "ns", None)
  scan, phase, dim = obs_plan(ns if ns else 48, args, sim)
  log(f"  [obs] scan={'on' if scan else 'off'} phase={'on' if phase else 'off'} "
      f"dim={dim}")

  metrics, traces = [], []
  for i in range(args.seeds):
    seed = args.seed0 + i
    t1 = time.time()
    tr = run_episode(sim, policy, args, seed, scan, phase, tag=tag)
    m = episode_metrics(tr, args)
    metrics.append(m)
    traces.append(tr)
    if not args.quiet:
      log(f"  [{i + 1:>3}/{args.seeds}] seed={seed:<4} "
          f"steps={m['n_steps']:>4} fell={int(m['fell_env'])} "
          f"vx={m['vx_mean']:+.3f} z={m['z_rel_mean']:.3f} "
          f"tilt_max={max(m['pitch_max_deg'], m['roll_max_deg']):5.1f}° "
          f"duty={np.round(m['duty'], 2).tolist()} "
          f"({time.time() - t1:.1f}s)")
  agg = aggregate(metrics, args)
  agg["terrain_report"] = terrain_report(traces, args)
  agg["config"] = dict(
      xml=xml, spawn=args.spawn, reset_noise=args.reset_noise,
      foot_contact=args.foot_contact, home_pose=args.home_pose,
      action_mode=args.action_mode, dof_order=sim.dof_order,
      dof_order_consistent=sim.dof_order_consistent,
      kp=args.kp, kd=args.kd, simulate_s=time.time() - t0,
      scan=scan, phase=phase, obs_dim=dim,
      normalizer=getattr(policy, "normalizer", None),
      dist=getattr(policy, "dist", "dummy"))

  if not args.no_plot and traces:
    make_plot(traces, agg, args, png_path)
    log(f"  [plot] {png_path}")
  if args.dump_trace:
    npz = os.path.join(args.out, tag + ".trace.npz")
    np.savez_compressed(npz, **{f"seed{tr['seed']}_{k}": v for tr in traces
                                for k, v in tr.items()
                                if isinstance(v, np.ndarray)})
    log(f"  [trace] {npz}")
  with open(json_path, "w") as f:
    json.dump(agg, f, ensure_ascii=False, indent=2, default=str)
  log(f"  [json] {json_path}")

  # ---------------- 打印报告 ----------------
  log("")
  log("-" * 78)
  log(f"结果: {tag}")
  log("-" * 78)
  log(f"  [1] 速度跟踪  v_cmd={agg['cmd']}")
  log(f"      实际 vx = {agg['vx_mean']:+.3f} ± {agg['vx_std_across']:.3f} m/s "
      f"(轮内 std {agg['vx_std_within']:.3f})")
  log(f"      vy = {agg['vy_mean']:+.3f}   wz = {agg['wz_mean']:+.3f} rad/s")
  log(f"      跟踪误差 = {agg['track_err']:.3f} m/s (稳态段 {agg['track_err_ss']:.3f})")
  log(f"      稳态轮占比 = {100 * agg['steady_rate']:.0f}%")
  log(f"  [2] 存活      存活率(env判据) = {100 * agg['survive_rate_env']:.1f}%  "
      f"(tilt判据 {100 * agg['survive_rate_tilt']:.1f}%)")
  log(f"      平均倒下跌落时间 = {agg['fall_time_mean_s']:.2f}s "
      f"(中位 {agg['fall_time_p50_s']:.2f}s)")
  log(f"  [3] 姿态      躯干相对地面高度 = {agg['z_rel_mean']:.3f} m "
      f"(最低 {agg['z_rel_min']:.3f})")
  log(f"      roll {agg['roll_mean_deg']:+.2f}° (max {agg['roll_max_deg']:.1f}°)  "
      f"pitch {agg['pitch_mean_deg']:+.2f}° (max {agg['pitch_max_deg']:.1f}°)")
  log(f"      角速度 RMS = {agg['angvel_rms']:.2f} rad/s")
  log(f"  [4] 步态      每足占空比 = {np.round(agg['duty_mean'], 3).tolist()} "
      f"(FR,FL,RR,RL)")
  log(f"      trot 占比 = {100 * agg['trot_frac']:.0f}%  "
      f"判为 trot 的轮占比 = {100 * agg['is_trot_frac']:.0f}%")
  log(f"      足端滑移 = {agg['slip_speed']:.3f} m/s   "
  f"|tau| 均值 = {agg['tau_abs_mean']:.2f} N·m (peak {agg['tau_abs_max']:.1f})")
  log(f"      目标被限幅的步占比 = {100 * agg['clamp_frac']:.1f}%  "
      f"目标顶到范围边界的比例 = {100 * agg['tgt_sat_frac']:.1f}%")
  log(f"      关节超出软限位比例 = {100 * agg['joint_over_frac']:.1f}% "
      f"(最大越界 {agg['joint_over_max']:.3f} rad, 软限位 = 范围内缩 5%)")
  log(f"  [5] 地形/位移 前进距离 = {agg['dist_forward_mean']:.2f} ± "
      f"{agg['dist_forward_std']:.2f} m  路程 = {agg['dist_path_mean']:.2f} m")
  tr5 = agg["terrain_report"]
  if any(r["n_reached"] for r in tr5["bins"]):
    log("      地形分层通过率 (按躯干下地形高度):")
    for r in tr5["bins"]:
      pr = "  n/a" if r["pass_rate"] is None else f"{100 * r['pass_rate']:5.1f}%"
      log(f"        h∈[{r['bin_lo_m']:.2f},{r['bin_hi_m']:.2f}) m: "
          f"到达 {r['n_reached']:>3} 轮, 通过 {pr}")
  else:
    log("      地形分层: 平地 (全程 h_ground=0), 无分层数据")
  log(f"      摔倒位置: {tr5['fall_xy'] if tr5['fall_xy'] else '无'}")
  log(f"  [6] 图: {png_path if not args.no_plot else '(已关闭)'}")
  log(f"  总耗时 {time.time() - t0:.1f}s")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
