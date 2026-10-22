#!/usr/bin/env bash
# v12/R1b: **混合几何**地形续训 —— 离散障碍用 box geom 真竖面, 平滑地形用 hfield。
# 地形 hyb1 (3x3 patch: boxes 0.06/0.09/0.13 + stairs_x/stairs_y 竖面 + rough/slope hfield,
# 场 ±4.2m, 外圈平地), 由 sim/gen_terrain_hybrid.py 生成。
# 动机: bg200 (纯 boxes 竖面) 让 v6a 从 26.8% 起步, 纯障碍场信号太负; 混合地形保留
# 平地/斜坡/起伏的"正样本", 更容易学且更接近交付场景。
# 用法: [SRC=... STEPS=15000000] bash sim/run_go2_v12_hyb.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
STEPS=${STEPS:-15000000}
LR=${LR:-1e-4}
NUM_EVALS=${NUM_EVALS:-5}
SRC=${SRC:-models/go2_v6a_logs/checkpoints/000007372800}
SCENE=${SCENE:-models/go2/scene_mjx_terrain_hyb1.xml}
BOXGRID=${BOXGRID:-models/go2/assets/hyb1.json}
TAG=${TAG:-v12_hyb1}
CKPT=$(readlink -f "$SRC" 2>/dev/null || true)
[ -z "$CKPT" ] || [ ! -d "$CKPT" ] && { echo "!! 起点不存在: $SRC"; exit 2; }
[ -f "$BOXGRID" ] || { echo "!! 高度表不存在: $BOXGRID"; exit 2; }
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
say "===== $TAG: 从 $CKPT 续训 $STEPS, 混合几何地形 ($SCENE) ====="
$PY -u -m train.train_go2 --num_timesteps "$STEPS" --num_evals "$NUM_EVALS" \
  $BASE $SCAN $LIDAR --terrain_xml "$SCENE" \
  --boxgrid_json "$BOXGRID" --contact_mode clearance --spawn_ring \
  --sb3_envs 2560 --hard_contact --healthy_z_ref feet \
  --restore "$CKPT" --step_offset 0 --lr "$LR" \
  --logdir "models/go2_${TAG}_logs" --save_name "go2_walk_${TAG}.pkl" 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! $TAG 未正常完成"; exit 1; fi
say "$TAG 完成"; ls -d "models/go2_${TAG}_logs/checkpoints"/*/ 2>/dev/null | tee -a "$LOG"
