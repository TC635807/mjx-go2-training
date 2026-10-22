#!/usr/bin/env bash
# v8: 平衡地形续训 (2026-10-19) —— v7 boxes 专项的正向收益 (+7pp boxes)
# 但 mix/stairs/slope 掉 2~6pp (遗忘), v8 用"boxes 3/9 + 保留其它地形 + 难度 1.1"重训。
#
# 起点: v6a@7.37M (v7 的负迁移不继承)
# 单变量 = 地形 (v8: boxes 3/9, stairs 2/9, rough/slope/wave/flat 各 1/9; 硬度 1.1)
#
# 用法: bash sim/run_go2_v8_mix.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
LOG=logs/go2_v8_train.log
mkdir -p logs models/go2_v8_logs
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

CKPT=$(readlink -f models/go2_v6a_logs/checkpoints/000007372800)
if [ ! -d "$CKPT" ]; then echo "!! 找不到起点 $CKPT"; exit 2; fi
say "===== v8: 从 v6a@7.37M 续训 10M, 平衡地形 (boxes 3/9, 硬度 1.1) ====="
say "起点: $CKPT"
say "地形: models/go2/scene_mjx_terrain_v8.xml"

$PY -u -m train.train_go2 --num_timesteps 10000000 --num_evals 5 \
  $BASE $SCAN $LIDAR --terrain_xml models/go2/scene_mjx_terrain_v8.xml \
  --sb3_envs 2560 --hard_contact --healthy_z_ref feet \
  --restore "$CKPT" --step_offset 7372800 --lr 1e-4 \
  --logdir models/go2_v8_logs --save_name go2_walk_v8.pkl 2>&1 | tee -a "$LOG"

if ! grep -q '训练完成' "$LOG"; then say "!! v8 未正常完成"; exit 1; fi
say "训练完成"
echo "hfield_overflow=$(grep -c 'height field collision overflow' "$LOG" || true)" | tee -a "$LOG"
echo "nan=$(grep -c 'loss=nan' "$LOG" || true)" | tee -a "$LOG"
say "检查点:"; ls -d models/go2_v8_logs/checkpoints/*/ 2>/dev/null | tee -a "$LOG"
