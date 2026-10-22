#!/usr/bin/env bash
# v9/R2: 摆动抬脚高度扫描 (0.15 -> SWING), 修 boxes "抬脚净空不足越箱棱"。
# 单变量 = swing_height (联动 max_foot_height)。
# **地形保持 mix (= v6a 的训练地形)**: v7/v8 实测"换地形续训"会导致 mix/stairs/slope
# 回退 1~5pp (遗忘), 所以本轮不换地形, 只改抬脚高度。
# 起点: v6a@7.37M (交付模型), 可 SRC 覆盖。
# 用法: [SWING=0.20 SRC=... TERRAIN=models/go2/scene_mjx_terrain_mix.xml] bash sim/run_go2_v9_swing.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
SWING=${SWING:-0.20}
SRC=${SRC:-models/go2_v6a_logs/checkpoints/000007372800}
TERRAIN=${TERRAIN:-models/go2/scene_mjx_terrain_mix.xml}
CKPT=$(readlink -f "$SRC" 2>/dev/null || true)
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then echo "!! 起点不存在: $SRC"; exit 2; fi
TAG="v9_swing${SWING}"
LOG="logs/go2_${TAG}_train.log"
mkdir -p logs "models/go2_${TAG}_logs"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42
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
say "===== $TAG: 从 $CKPT 续训 10M, swing_height=$SWING (默认 0.15), 地形 $TERRAIN ====="
$PY -u -m train.train_go2 --num_timesteps 10000000 --num_evals 5 \
  $BASE $SCAN $LIDAR --terrain_xml "$TERRAIN" \
  --swing_height "$SWING" \
  --sb3_envs 2560 --hard_contact --healthy_z_ref feet \
  --restore "$CKPT" --step_offset 0 --lr 1e-4 \
  --logdir "models/go2_${TAG}_logs" --save_name "go2_walk_${TAG}.pkl" 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! $TAG 未正常完成"; exit 1; fi
say "$TAG 完成"; ls -d "models/go2_${TAG}_logs/checkpoints"/*/ 2>/dev/null | tee -a "$LOG"
