#!/usr/bin/env bash
# Go2 v3 训练 —— 与 v2 (交付模型 policies/go2_walk_policy_hard.pkl, 37.26M 步)
# 的**唯一**差异: obs 0:3 (基座线速度) 从"真值 + 小小白噪声"改成
# "带真实估计误差的速度" (常值偏置 + OU 慢漂 + 观测延迟), 并适当加大白噪声。
#
# 依据 (实测, 见 docs/go2-v3-obs-vel-error.md):
#   sim 内现有 LegOdometry 相对真值: 偏置 ~0.07 m/s, 即时 std ~0.23 m/s,
#   1s 平滑后 std ~0.06 m/s (慢漂), 且有若干步滞后。
#   训练注入: bias U(-0.12,0.12) m/s (逐轴), OU std 0.08 m/s / tau 2s,
#   延迟 0~2 个控制步 (0~40ms), 白噪声 0.05 -> 0.15 m/s。
#   真机 (编码器噪声/IMU 零偏) 只会更差, 所以取比 sim 实测略保守。
#
# 单变量原则: 地形、奖励、接触参数与 v2 完全一致; 地形升级留给 v4。
#
# 实测: 40M 步 ≈ 2.5 h @ RTX 5060 Laptop 8GB (fps 4200-4700)。
# 显存: 用 3072 env + mem fraction 0.45 (v2 在 4096 env 于 33M 步处 OOM 过一次)。
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-40000000}"
LOGDIR="${2:-models/go2_v3_logs}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

exec /home/tc63/mujoco/.venv/bin/python -u -m train.train_go2 \
  --num_timesteps "$STEPS" --num_evals 16 \
  --sb3_full --sb3_envs 3072 \
  --sample_cmd --fwd_turn --body_frame \
  --cmd_low "0,-0.4,-1.5" --cmd_high "1.0,0.4,1.5" \
  --tilt_deg 17.5 --natural_gait --dangle_penalty --layers 128,128 \
  --terrain --height_scan --push --obs_noise \
  --obs_noise_linvel 0.15 \
  --obs_vel_bias 0.12 --obs_vel_drift 0.08 --obs_vel_tau 2.0 --obs_vel_delay 2 \
  --climb_reward --soft_landing --gait_phase --body_contact \
  --ascent --ascent_w 4.0 --ascent_gate 0.2 \
  --symmetry --symmetry_w -0.4 \
  --logdir "$LOGDIR" \
  --save_name go2_walk_v3.pkl
