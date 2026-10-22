#!/usr/bin/env python
"""把 brax PPO 的 orbax 检查点 (checkpoints/best 或 <step>/) 转成 .pkl 参数文件。

动机: 部署侧 scripts/export_go1_policy.py 只吃 .pkl (params 元组)。训练结束时保存的
pkl 是**最终参数**, 而 v5/v6 的实测反复表明 best 检查点常优于 final 参数
(v5: final mix 75.0% vs best 81.2%)。所以部署导出应优先用 best。

用法:
  python sim/ckpt_to_pkl.py --ckpt models/go2_v6a_logs/checkpoints/best \
      --out policies/go2_walk_v6_best.pkl
"""
import argparse
import os
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # 只读参数, 不需要 GPU

import numpy as np  # noqa: E402


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True, help="orbax 检查点目录 (含 ppo_network_config.json)")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  ckpt = os.path.abspath(args.ckpt)
  if not os.path.isdir(ckpt):
    raise SystemExit(f"检查点不存在: {ckpt}")
  cfg_path = os.path.join(ckpt, "ppo_network_config.json")
  if not os.path.isfile(cfg_path):
    raise SystemExit(f"不是 brax 检查点 (缺 ppo_network_config.json): {ckpt}")

  from brax.training.agents.ppo import checkpoint as ppo_ckpt
  params = ppo_ckpt.load(ckpt)

  def _leaves(x):
    if isinstance(x, dict):
      for v in x.values():
        yield from _leaves(v)
    elif isinstance(x, (tuple, list)):
      for v in x:
        yield from _leaves(v)
    else:
      try:
        yield np.asarray(x)
      except Exception:
        return

  leaves = list(_leaves(params))
  n = sum(int(a.size) for a in leaves if a.dtype != object)
  out = os.path.abspath(args.out)
  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, "wb") as f:
    pickle.dump(params, f)
  print(f"检查点: {ckpt}")
  print(f"参数叶子数={len(leaves)}  总参数={n:,}")
  print(f"已保存: {out} ({os.path.getsize(out)/1e6:.2f} MB)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
