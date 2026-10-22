#!/usr/bin/env bash
# v14/R5: 大网络 + 长训练 (调研建议: 开源 rough 配方 512-256-128 + 147M 样本;
# 本项目一直用 128x128 + 7-17M)。**从零训练** (网络形状不同, 不能续训),
# 地形 hyb2 (竖面+hfield 混合), 40M 步。
# 用法: [STEPS=40000000 LAYERS=512,256,128] bash sim/run_go2_v14_bignet.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
STEPS=${STEPS:-40000000}
LAYERS=${LAYERS:-512,256,128}
TAG=${TAG:-v14_big}
SCENE=${SCENE:-models/go2/scene_mjx_terrain_hyb2.xml}
BOXGRID=${BOXGRID:-models/go2/assets/hyb2.json}
LOG="logs/go2_${TAG}_train.log"
mkdir -p logs "models/go2_${TAG}_logs"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
LIDAR="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
BASE="--sb3_full --sample_cmd --fwd_turn --body_frame \
  --cmd_low 0,-0.4,-1.5 --cmd_high 1.0,0.4,1.5 \
  --tilt_deg 17.5 --natural_gait --dangle_penalty --layers $LAYERS \
  --terrain --height_scan --push --obs_noise \
  --climb_reward --soft_landing --gait_phase --body_contact \
  --ascent --ascent_w 4.0 --ascent_gate 0.2 \
  --symmetry --symmetry_w -0.4"
say "===== $TAG: 从零训练 $STEPS 步, layers=$LAYERS, 地形 hyb2 ====="
$PY -u -m train.train_go2 --num_timesteps "$STEPS" --num_evals 8 \
  $BASE $SCAN $LIDAR --terrain_xml "$SCENE" \
  --boxgrid_json "$BOXGRID" --contact_mode clearance --spawn_ring \
  --sb3_envs 2560 --hard_contact --healthy_z_ref feet --lr 1e-4 \
  --logdir "models/go2_${TAG}_logs" --save_name "go2_walk_${TAG}.pkl" 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! $TAG 未正常完成"; exit 1; fi
say "$TAG 完成"; ls -d "models/go2_${TAG}_logs/checkpoints"/*/ 2>/dev/null | tee -a "$LOG"
