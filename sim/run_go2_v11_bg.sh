#!/usr/bin/env bash
# v11/R1: 真竖面离散方块地形 (box geom) —— 修 hfield 无竖面 (0.094m 格把 0.11m
# 箱棱抹成 ~50° 坡)。地形由 sim/gen_terrain_boxgeom.py 生成:
#   bg200: 176 个 box geom (0.4m 方格, 高<=0.13m), 底面 hfield 全 0;
#   env: terrain_boxgrid_json 解析查表 + contact_mode=clearance (box geom 不触发
#        foot-floor 传感器)。
# 单变量 = 地形几何 (起点/奖励/遮挡/超参同 v6a)。
# 用法: [STEPS=12000000 SRC=...] bash sim/run_go2_v11_bg.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
STEPS=${STEPS:-12000000}
SRC=${SRC:-models/go2_v6a_logs/checkpoints/000007372800}
SCENE=${SCENE:-models/go2/scene_mjx_terrain_bg200.xml}
BOXGRID=${BOXGRID:-models/go2/assets/bg200.json}
CKPT=$(readlink -f "$SRC" 2>/dev/null || true)
[ -z "$CKPT" ] || [ ! -d "$CKPT" ] && { echo "!! 起点不存在: $SRC"; exit 2; }
[ -f "$BOXGRID" ] || { echo "!! 高度表不存在: $BOXGRID"; exit 2; }
TAG=${TAG:-v11_bg200}
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
say "===== $TAG: 从 $CKPT 续训 $STEPS, 真竖面 box geom 地形 ($SCENE) ====="
say "boxgrid: $BOXGRID | contact_mode=clearance"
$PY -u -m train.train_go2 --num_timesteps "$STEPS" --num_evals 5 \
  $BASE $SCAN $LIDAR --terrain_xml "$SCENE" \
  --boxgrid_json "$BOXGRID" --contact_mode clearance --spawn_ring \
  --sb3_envs 2560 --hard_contact --healthy_z_ref feet \
  --restore "$CKPT" --step_offset 0 --lr 1e-4 \
  --logdir "models/go2_${TAG}_logs" --save_name "go2_walk_${TAG}.pkl" 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! $TAG 未正常完成"; exit 1; fi
say "$TAG 完成"; ls -d "models/go2_${TAG}_logs/checkpoints"/*/ 2>/dev/null | tee -a "$LOG"
