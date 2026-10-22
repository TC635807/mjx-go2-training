#!/usr/bin/env bash
# v8b: 从 v8-best 在 mix 地形上短续训 (3M, 低 lr), 用于评估"平衡地形 -> 原 mix 回补"
# 是否能同时保住 boxes 收益与 mix/stairs/slope 表现。
# 用法: bash sim/run_go2_v8b_mix.sh [起点ckpt]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
LOG=logs/go2_v8b_train.log
mkdir -p logs models/go2_v8b_logs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
LIDAR="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
BASE="--sb3_full --sample_cmd --fwd_turn --body_frame \
  --cmd_low 0,-0.4,-1.5 --cmd_high 1.0,0.4,1.5 \
  --tilt_deg 17.5 --natural_gait --dangle_penalty --layers 128,128 \
  --terrain --height_scan --push --obs_noise \
  --climb_reward --soft_landing --gait_phase --body_contact \
  --ascent --ascent_w 4.0 --ascent_gate 0.2 \
  --symmetry --symmetry_w -0.4"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42
SRC=${1:-models/go2_v8_logs/checkpoints/best}
CKPT=$(readlink -f "$SRC" 2>/dev/null || true)
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then echo "!! 起点不存在: $SRC"; exit 2; fi
ST=$(basename "$CKPT")
say "===== v8b: 从 $CKPT 在 mix 地形短续训 3M (lr 5e-5) ====="
$PY -u -m train.train_go2 --num_timesteps 3000000 --num_evals 3 \
  $BASE $SCAN $LIDAR --terrain_xml models/go2/scene_mjx_terrain_mix.xml \
  --sb3_envs 2560 --hard_contact --healthy_z_ref feet \
  --restore "$CKPT" --step_offset 0 --lr 5e-5 \
  --logdir models/go2_v8b_logs --save_name go2_walk_v8b.pkl 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! v8b 未正常完成"; exit 1; fi
say "v8b 完成"; ls -d models/go2_v8b_logs/checkpoints/*/ 2>/dev/null | tee -a "$LOG"
