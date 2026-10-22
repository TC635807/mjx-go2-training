#!/usr/bin/env bash
# Go2 地形行走训练 —— **这是实际跑出交付模型的那一条命令** (Go2 v1)。
#
# 配方来源: docs/experiment-log.md §24.8 (v22) / §28.3 (v23) —— Go1 上唯一
# "地形 + 高度扫描 + 扰动 + 观测噪声"全开且稳定跑通的配置。
# Go2 的四处差异 (全部有实测依据):
#   1. 不续训 (Go1 检查点不能跨机器人), 所以不传 --lr 1e-4, 用默认 3e-4。
#   2. 指令范围扩大到 vx∈[0,1.0], vy∈±0.4, wz∈±1.5 → 必须 --body_frame。
#   3. --sb3_envs 4096: Go1 的 768 档实测 ~3400 fps, 4096 档 ~4700 fps (显存够)。
#   4. envs/go2_walk.py 的 naconmax 从 Go1 的 8*8192 提到 32*4096 —— Go2 在地形上
#      需要的接触槽比 Go1 多 (实测 4096 环境报 overflow 到 86518 = 21.1/world),
#      不修会静默丢接触、物理退化。
#
# 实测: 40M 步 ≈ 2.5 h @ fps 4300-4700 (RTX 5060 Laptop 8GB)。
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-40000000}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55

exec /home/tc63/mujoco/.venv/bin/python -u -m train.train_go2 \
  --num_timesteps "$STEPS" --num_evals 16 \
  --sb3_full --sb3_envs 4096 \
  --sample_cmd --fwd_turn --body_frame \
  --cmd_low "0,-0.4,-1.5" --cmd_high "1.0,0.4,1.5" \
  --tilt_deg 17.5 --natural_gait --dangle_penalty --layers 128,128 \
  --terrain --height_scan --push --obs_noise \
  --climb_reward --soft_landing --gait_phase --body_contact \
  --ascent --ascent_w 4.0 --ascent_gate 0.2 \
  --symmetry --symmetry_w -0.4 \
  --logdir models/go2_v1_logs \
  --save_name go2_walk_policy.pkl
