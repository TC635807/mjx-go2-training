#!/usr/bin/env python
"""把 brax/orbax 检查点目录导出成 policies/*.pkl (与 train_go2.py 末尾一致)。

为什么要这个: 训练崩溃/被中断时不会走到 train 脚本的保存那一步, 但检查点已经落盘,
里面就是可用的策略参数。用它把检查点捞成 pkl, 同时把 brax 的 network 配置一起记下来。

用法 (**务必在 CPU 上跑, 避免和训练抢显存**):
  JAX_PLATFORMS=cpu python sim/export_go2_policy.py \
      --ckpt models/go2_v1_logs/checkpoints/000033030144 \
      --out policies/go2_walk_policy_v1_soft33M.pkl
"""
import argparse
import json
import os
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True, help="检查点目录 (绝对或相对)")
  ap.add_argument("--out", required=True, help="输出 pkl 路径")
  ap.add_argument("--note", default="", help="写进旁注文件里的说明")
  args = ap.parse_args()

  ckpt = os.path.abspath(args.ckpt)
  out = os.path.abspath(args.out)
  if not os.path.isdir(ckpt):
    raise SystemExit(f"检查点目录不存在: {ckpt}")

  from brax.training.agents.ppo import checkpoint as ppo_ckpt
  params = ppo_ckpt.load(ckpt)

  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, "wb") as f:
    pickle.dump(params, f)
  print(f"已写出: {out} ({os.path.getsize(out)} bytes)")

  cfg_path = os.path.join(ckpt, "ppo_network_config.json")
  meta = {"source_ckpt": ckpt, "pkl": out, "note": args.note}
  if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
      meta["ppo_network_config"] = json.load(f)
  with open(out + ".meta.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
  print(f"已写出旁注: {out}.meta.json")
  return 0


if __name__ == "__main__":
  sys.exit(main())
