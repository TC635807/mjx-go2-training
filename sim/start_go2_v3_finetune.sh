#!/usr/bin/env bash
# Go2 v3 修正版: 从 v2 的**硬接触**最终检查点 (37.26M 步) 续训 10M 步,
# 只改一件事: obs 0:3 换成"带真实估计误差的速度" (bias + OU 慢漂 + 延迟).
#
# 为什么是"续训"而不是"从零 40M":
#   * v2 已经是硬接触、已收敛的交付基线 (37.26M 步, 平地 100%/部署 93.8%);
#   * v3 相对 v2 的唯一语义变化就是速度观测分布, 从收敛策略微调最省时 (~40min vs 2.5h);
#   * 代价: 比 v2 多 10M 步 (lr 2e-4), A/B 对比时要把这点写清楚。
#
# ⚠ 事故记录: 第一次 v3 (models/go2_v3_logs) 漏了 --hard_contact, 跑的是
#   **软接触**物理 (= v1 的 5.2x sim2sim gap 那套), 在 33.03M 步 OOM 崩掉且
#   不该用于交付。该目录保留作证据, 不再续训。
#
# 显存: 8GB 卡与主机桌面共享 (桌面占 ~2.8GB), 故 envs 3072 -> 2560, mem 0.42.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-10000000}"
LOGDIR="${2:-models/go2_v3ft_logs}"
CKPT="$(readlink -f models/go2_v2c_logs/checkpoints/000007372800)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42

if [ ! -f "$CKPT/ppo_network_config.json" ]; then
  echo "缺少 v2 检查点: $CKPT" >&2; exit 2
fi
mkdir -p "$LOGDIR"; LOGDIR="$(readlink -f "$LOGDIR")"
echo "v3 微调: ckpt=$CKPT (v2 硬接触 37.26M) + 本段 $STEPS 步, lr=2e-4, logdir=$LOGDIR"
echo "唯一差异: obs 0:3 = bias±0.12 + OU(0.08,2s) + delay 0..2 步 + 白噪声 0.15"

exec /home/tc63/mujoco/.venv/bin/python -u -m train.train_go2 \
  --num_timesteps "$STEPS" --num_evals 6 \
  --sb3_full --sb3_envs 2560 --hard_contact \
  --sample_cmd --fwd_turn --body_frame \
  --cmd_low "0,-0.4,-1.5" --cmd_high "1.0,0.4,1.5" \
  --tilt_deg 17.5 --natural_gait --dangle_penalty --layers 128,128 \
  --terrain --height_scan --push --obs_noise \
  --obs_noise_linvel 0.15 \
  --obs_vel_bias 0.12 --obs_vel_drift 0.08 --obs_vel_tau 2.0 --obs_vel_delay 2 \
  --climb_reward --soft_landing --gait_phase --body_contact \
  --ascent --ascent_w 4.0 --ascent_gate 0.2 \
  --symmetry --symmetry_w -0.4 \
  --restore "$CKPT" --step_offset 37257216 --lr 2e-4 \
  --logdir "$LOGDIR" \
  --save_name go2_walk_v3.pkl
