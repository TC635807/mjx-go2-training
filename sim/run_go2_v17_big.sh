#!/usr/bin/env bash
# v17/R5: **大网络** (默认 512-256-128, 调研建议) 分阶段训练。
#   阶段 A: 在 mix hfield 地形从零 (或 SRC) 训练, 先学会走;
#   阶段 B: 在 hyb2 混合几何地形续训, 学竖面。
# 用法:
#   TAG=v17a_big_mix SCENE=models/go2/scene_mjx_terrain_mix.xml STEPS=12000000 \
#     bash sim/run_go2_v17_big.sh
#   TAG=v17b_big_hyb2 SRC=models/go2_v17a_big_mix_logs/checkpoints/best \
#     SCENE=models/go2/scene_mjx_terrain_hyb2.xml BOXGRID=models/go2/assets/hyb2.json \
#     CONTACT=clearance RING=1 STEPS=24000000 bash sim/run_go2_v17_big.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
STEPS=${STEPS:-12000000}
LR=${LR:-1e-4}
HIST=${HIST:-0}
RANGE=${RANGE:-0}
NOISE=${NOISE:-0}
NOISE_ARG=""
[ "$NOISE" != "0" ] && NOISE_ARG="--obs_noise --obs_noise_linvel 0.05 --obs_vel_bias 0.05 --obs_vel_drift 0.04 --obs_vel_delay 1"
RANGE_ARG=""
[ "$RANGE" != "0" ] && RANGE_ARG="--random_init_range $RANGE"
NUM_EVALS=${NUM_EVALS:-8}
LAYERS=${LAYERS:-512,256,128}
TAG=${TAG:-v17_big}
SCENE=${SCENE:-models/go2/scene_mjx_terrain_mix.xml}
BOXGRID=${BOXGRID:-}
CONTACT=${CONTACT:-sensor}
RING=${RING:-0}
SRC=${SRC:-}
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
EXTRA=()
[ -n "$BOXGRID" ] && EXTRA+=(--boxgrid_json "$BOXGRID" --contact_mode "$CONTACT")
[ "$RING" = "1" ] && EXTRA+=(--spawn_ring)
REST=()
if [ -n "$SRC" ]; then
  CK=$(readlink -f "$SRC" 2>/dev/null || true)
  [ -z "$CK" ] || [ ! -d "$CK" ] && { echo "!! 起点不存在: $SRC"; exit 2; }
  REST+=(--restore "$CK" --step_offset 0)
  say "起点: $CK"
else
  say "从零训练"
fi
say "===== $TAG: $STEPS 步, layers=$LAYERS, 地形 $SCENE ====="
$PY -u -m train.train_go2 --num_timesteps "$STEPS" --num_evals "$NUM_EVALS" \
  $BASE $SCAN $LIDAR --terrain_xml "$SCENE" \
  --sb3_envs 2560 --hard_contact --healthy_z_ref feet --lr "$LR" \
  --obs_history_frames "$HIST" $RANGE_ARG $NOISE_ARG \
  "${EXTRA[@]}" "${REST[@]}" \
  --logdir "models/go2_${TAG}_logs" --save_name "go2_walk_${TAG}.pkl" 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! $TAG 未正常完成"; exit 1; fi
say "$TAG 完成"; ls -d "models/go2_${TAG}_logs/checkpoints"/*/ 2>/dev/null | tee -a "$LOG"
