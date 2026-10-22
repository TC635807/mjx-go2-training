#!/usr/bin/env python
"""把 128x128 的 brax PPO 参数**精确扩展**成 512-256 的大网络, 生成可 restore 的
orbax checkpoint —— 用于 R5 大网络**热启动**(而不是从零训练 2-3 小时)。

映射 (数学上等价, 因为 tanh(0)=0, 零块不激活):
  policy: hidden_0 (243,128)->(243,512) 左块保留; hidden_1 (128,128)->(512,256)
          放在 [:128,:128]; 输出 Dense_0 (128,12)->(256,12) 上块保留;
          std_param (12,) 不变。
  value : hidden_0 (314,128)->(314,512); hidden_1 ->(512,256) 块对角;
          hidden_2 (128,1)->(256,1) 上块保留。
于是大网络在初始时刻与旧网络**输出完全相同**, 再继续训练以利用新增容量。

用法:
  python sim/warmstart_bignet.py --pkl policies/go2_walk_v13.pkl \
      --ckpt_config models/go2_v13_hyb2_logs/checkpoints/000020316160/ppo_network_config.json \
      --out models/go2_v17_warm --layers 512,256
输出: <out>/checkpoints/000000000000/ (ppo_network_config.json + orbax 参数)
"""
import argparse
import json
import os
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)


def pad_policy(pol, new_layers, extra_in_rows=0):
  """pol: {'params': {'MLP_0': {'hidden_i': {kernel,bias}}, 'Dense_0': ...}, 'std_param'}。

  extra_in_rows: v27 观测历史 —— 在输入层**末尾**追加 extra 行 (全 0), 使新网络
  对"只有前 243 维、历史维为 0"的输入与原网络逐位等价。
  """
  import jax.numpy as jnp
  mlp_old = pol["params"]["MLP_0"]
  old_h = [k for k in mlp_old if k.startswith("hidden_")]
  n_old = len(old_h)
  assert len(new_layers) == n_old, (len(new_layers), n_old)
  new_mlp = {}
  for i, width in enumerate(new_layers):
    name = "hidden_%d" % i
    old_w = mlp_old[name]["kernel"]
    old_b = mlp_old[name]["bias"]
    old_out = old_w.shape[1]
    in_dim = (int(old_w.shape[0]) + extra_in_rows if i == 0
              else int(new_layers[i - 1]))
    new_w = jnp.zeros((in_dim, width), dtype=old_w.dtype)
    new_b = jnp.zeros((width,), dtype=old_b.dtype)
    if i == 0:
      new_w = new_w.at[:old_w.shape[0], :old_out].set(old_w)
      new_b = new_b.at[:old_out].set(old_b)
    else:
      prev_out = mlp_old["hidden_%d" % (i - 1)]["kernel"].shape[1]
      new_w = new_w.at[:prev_out, :old_out].set(old_w)
      new_b = new_b.at[:old_out].set(old_b)
    new_mlp[name] = {"kernel": new_w, "bias": new_b}
  old_dense = pol["params"]["Dense_0"]
  w, b = old_dense["kernel"], old_dense["bias"]
  prev_out = mlp_old["hidden_%d" % (n_old - 1)]["kernel"].shape[1]
  new_w = jnp.zeros((new_layers[-1], w.shape[1]), dtype=w.dtype)
  new_w = new_w.at[:prev_out].set(w)
  return {"params": {"MLP_0": new_mlp,
                     "Dense_0": {"kernel": new_w, "bias": b},
                     "std_param": pol["params"]["std_param"]}}


def pad_normalizer(norm, extra_rows):
  """v27: 观测历史使 state obs 变长 -> 运行统计 (mean/std/summed_variance) 也要补齐。

  新维度初始 mean=0 / std=1 / summed_variance=0; 训练过程中 brax 会用真实数据更新。
  (本项目 normalize_observations=False, 统计量不影响前向, 但 brax 训练步会校验形状)
  """
  import jax.numpy as jnp
  if extra_rows <= 0:
    return norm
  def _pad(d, fill):
    out = dict(d)
    s = out["state"]
    out["state"] = jnp.concatenate([s, jnp.full((extra_rows,), fill, s.dtype)])
    return out
  import dataclasses
  return dataclasses.replace(
      norm,
      mean=_pad(norm.mean, 0.0),
      std=_pad(norm.std, 1.0),
      summed_variance=_pad(norm.summed_variance, 0.0))


def pad_value(val, new_layers):
  import jax.numpy as jnp
  src = val["params"]
  out = {"params": {}}
  for i, width in enumerate(new_layers):
    name = "hidden_%d" % i
    w, b = src[name]["kernel"], src[name]["bias"]
    old_out = w.shape[1]
    in_dim = int(w.shape[0]) if i == 0 else int(new_layers[i - 1])
    new_w = jnp.zeros((in_dim, width), dtype=w.dtype)
    new_b = jnp.zeros((width,), dtype=b.dtype)
    if i == 0:
      new_w = new_w.at[:, :old_out].set(w)
      new_b = new_b.at[:old_out].set(b)
    else:
      prev_out = src["hidden_%d" % (i - 1)]["kernel"].shape[1]
      new_w = new_w.at[:prev_out, :old_out].set(w)
      new_b = new_b.at[:old_out].set(b)
    out["params"][name] = {"kernel": new_w, "bias": new_b}
  # 最后一层 (输出 1 维)
  w, b = src["hidden_%d" % len(new_layers)]["kernel"], src["hidden_%d" % len(new_layers)]["bias"]
  prev_out = src["hidden_%d" % (len(new_layers) - 1)]["kernel"].shape[1]
  new_w = jnp.zeros((new_layers[-1], w.shape[1]), dtype=w.dtype)
  new_w = new_w.at[:prev_out].set(w)
  out["params"]["hidden_%d" % len(new_layers)] = {"kernel": new_w, "bias": b}
  return out


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--pkl", default="policies/go2_walk_v13.pkl")
  ap.add_argument("--ckpt_config",
                  default="models/go2_v13_hyb2_logs/checkpoints/000020316160/ppo_network_config.json")
  ap.add_argument("--out", default="models/go2_v17_warm")
  ap.add_argument("--layers", default="512,256")
  ap.add_argument("--obs_history_frames", type=int, default=0,
                  help="v27: actor 历史帧数; 输入层追加 (N-1)*56 个 0 行")
  ap.add_argument("--step", type=int, default=0)
  args = ap.parse_args(argv)
  layers = tuple(int(x) for x in args.layers.split(","))
  with open(args.pkl, "rb") as fh:
    params = pickle.load(fh)
  norm, pol, val = params
  extra_in = max(0, int(args.obs_history_frames) - 1) * 56
  norm = pad_normalizer(norm, extra_in)
  big_pol = pad_policy(pol, layers, extra_in_rows=extra_in)
  big_val = pad_value(val, layers)
  big = [norm, big_pol, big_val]

  # ---- 等价性自检: 旧网络 vs 扩展网络在同一输入上的输出 ----
  from brax.training.agents.ppo import networks as ppo_networks
  obs_state = jp.zeros((5, 243 + extra_in))
  obs_state_old = jp.zeros((5, 243))
  obs_priv = jp.zeros((5, 314))
  _mlp = pol["params"]["MLP_0"]
  old_layers = tuple(_mlp["hidden_%d" % i]["kernel"].shape[1]
                     for i in range(len([k for k in _mlp if k.startswith("hidden_")])))

  def make_net(layer_sizes):
    return ppo_networks.make_ppo_networks(
        observation_size={"state": (243,), "privileged_state": (314,)},
        action_size=12, policy_hidden_layer_sizes=layer_sizes,
        value_hidden_layer_sizes=layer_sizes, policy_obs_key="state",
        value_obs_key="privileged_state", distribution_type="normal",
        activation=jax.nn.tanh)

  net_old = make_net(old_layers)
  net_new = make_net(layers)
  def _first(x):
    return x[0] if isinstance(x, (tuple, list)) else x

  m_old = _first(net_old.policy_network.apply(norm, pol, obs_state_old))
  m_new = _first(net_new.policy_network.apply(norm, big_pol, obs_state))
  d_pol = float(jp.max(jp.abs(m_old - m_new)))
  v_old = _first(net_old.value_network.apply(norm, val, obs_priv))
  v_new = _first(net_new.value_network.apply(norm, big_val, obs_priv))
  d_val = float(jp.max(jp.abs(v_old - v_new)))
  print("[warmstart] policy 等价性 max|d| = %.3e ; value max|d| = %.3e" % (d_pol, d_val))
  if not (d_pol < 1e-4 and d_val < 1e-4):
    raise SystemExit("扩展网络与旧网络输出不一致, 终止 (检查映射)")

  # ---- 写 brax 兼容 checkpoint ----
  from brax.training.agents.ppo import checkpoint as ppo_ckpt
  from ml_collections import config_dict as cd
  cfg_dict = json.load(open(args.ckpt_config))
  nf = cfg_dict["network_factory_kwargs"]
  nf["policy_hidden_layer_sizes"] = list(layers)
  nf["value_hidden_layer_sizes"] = list(layers)
  if extra_in:
    cfg_dict["observation_size"]["state"]["shape"] = [243 + extra_in]
  # 从 json 读回的 kernel init 是字符串, brax save 需要 callable 或 None -> 置 None
  for k in list(nf):
    if k.endswith("kernel_init_fn") and isinstance(nf[k], str):
      nf[k] = None
  cfg = cd.ConfigDict(cfg_dict)
  out = os.path.abspath(args.out)
  ppo_ckpt.save(out, args.step, big, cfg)
  ck = os.path.join(out, "%012d" % args.step)
  print("[warmstart] 写出 checkpoint:", ck)
  print("  文件:", sorted(os.listdir(ck))[:6])
  return 0


if __name__ == "__main__":
  sys.exit(main())
