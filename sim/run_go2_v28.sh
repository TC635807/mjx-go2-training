#!/usr/bin/env bash
# v28: **台阶专项 + 步态自然性/机体稳定性** (2026-10-21, 回应用户三条反馈)
#
# 动机 (实测):
#  1) 上台成功率低: 训练侧 warp 评测 stg1 70.9% / bg200 73.6% 是最弱轴; 台阶在
#     hyb2 只占少量 patch, 随机出生遇到台阶的样本比例低 -> 台阶梯度被稀释。
#     v28 阶段 A 用 sim/gen_terrain_stairs_box.py 生成的 **只有台阶** 真竖面
#     box 地形 (stairs1/stairs2), 台阶占可走区域 1/3 以上, 外圈平地环形出生。
#  2) 四髋摆动极大/走路晃: 实测 hip_q mean 29-33°/p95 42-47°, hip 动作 p95
#     1.2-1.5rad 顶到 ±1.047 限位, v20/v24/v25a/v25b 一致且平地同样存在
#     -> 奖励面缺项。v28 新增 --hip_abduction_w (四髋偏离 home 平方和) +
#     提高 --action_rate_w (动作平滑) + 可选 --xy_angvel_w (压制横滚/俯仰角速度)。
#  3) 脚卡在瓦缝/台阶立面: 部署侧碰撞几何问题 (T1/T2 诊断), 不是策略; 阶段 B
#     在 hyb2 巩固时用修复后的部署评测。
#
# 训练配方 (相对 v25b 的最小改动, 其余 robustness/obs/lidar/命令采样全部一致):
#   v25b = sim/run_go2_v17_big.sh 的 BASE + SCAN + LIDAR + sb3_envs 2560 +
#          hard_contact + healthy_z_ref feet + lr 2e-5 (+ v25a 速度观测随机化)
#   v28 新增: HIPW (默认 -0.5), ACTW (默认 -0.02), SWING (默认 0.18)
#
# 用法:
#   # 阶段 A: 从 v25b 热启动, stairs1 专项 3M (快速实验, 保持可部署性)
#   TAG=v28a_stairs SRC=models/go2_v25b_cons_logs/checkpoints/000002048000 \
#     SCENE=models/go2/scene_mjx_terrain_stairs1.xml BOXGRID=models/go2/assets/stairs1.json \
#     CONTACT=clearance RING=1 STEPS=3000000 LAYERS=1024,512 LR=2e-5 \
#     HIPW=-0.5 ACTW=-0.02 SWING=0.18 bash sim/run_go2_v28.sh
#   # 阶段 B: hyb2 巩固 2M (与 v25b 同款, 保可部署性)
#   TAG=v28c_cons SRC=models/go2_v28a_stairs_logs/checkpoints/best \
#     SCENE=models/go2/scene_mjx_terrain_hyb2.xml BOXGRID=models/go2/assets/hyb2.json \
#     CONTACT=clearance RING=1 STEPS=2000000 LR=2e-5 HIPW=-0.5 ACTW=-0.02 \
#     bash sim/run_go2_v28.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
STEPS=${STEPS:-3000000}
LR=${LR:-2e-5}
HIST=${HIST:-0}
RANGE=${RANGE:-0}
NOISE=${NOISE:-1}
NUM_EVALS=${NUM_EVALS:-4}
LAYERS=${LAYERS:-1024,512}
VLAYERS=${VLAYERS:-}
TAG=${TAG:-v28_stairs}
SCENE=${SCENE:-models/go2/scene_mjx_terrain_stairs1.xml}
BOXGRID=${BOXGRID:-models/go2/assets/stairs1.json}
CONTACT=${CONTACT:-clearance}
RING=${RING:-1}
SRC=${SRC:-models/go2_v25b_cons_logs/checkpoints/000002048000}
ENVS=${ENVS:-2560}
# --- v28 步态/稳定性权重 ---
HIPW=${HIPW:--0.5}
HIPACTW=${HIPACTW:-}
ENT=${ENT:-}
ACTW=${ACTW:--0.02}
XYW=${XYW:-}
ORIW=${ORIW:-}
SWING=${SWING:-0.18}
SYMM=${SYMM:--0.4}
EDGE=${EDGE:-}
EDGE_THRESH=${EDGE_THRESH:-}
FEET_EDGE_BG=${FEET_EDGE_BG:-0}
ASCENT=${ASCENT:-4.0}
CMD_LOW=${CMD_LOW:-0,-0.4,-1.5}
CMD_HIGH=${CMD_HIGH:-1.0,0.4,1.5}
SPAWN_STAIRS=${SPAWN_STAIRS:-0}
EP_LEN=${EP_LEN:-750}
PROGW=${PROGW:-}
CLEARW=${CLEARW:-}
CLEART=${CLEART:-}
LAZYW=${LAZYW:-}
TERMW=${TERMW:-}
RFLOOR=${RFLOOR:-}
PHASEW=${PHASEW:-}
FULL_RESET=${FULL_RESET:-0}
NACON=${NACON:-}
NJMAX=${NJMAX:-}
XLAFRAC=${XLAFRAC:-0.42}
RESETS_PER_EVAL=${RESETS_PER_EVAL:-0}
CURRICULUM=${CURRICULUM:-0}
FPROB=${FPROB:-}
CMD_F_LOW=${CMD_F_LOW:-}
CMD_F_HIGH=${CMD_F_HIGH:-}
CURINIT=${CURINIT:-2}
CURSUCC=${CURSUCC:-3.0}
CURMIN=${CURMIN:-1.0}
TBPEN=${TBPEN:-}
ACCW=${ACCW:-}
LOG="logs/go2_${TAG}_train.log"
mkdir -p logs "models/go2_${TAG}_logs"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=$XLAFRAC
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
LIDAR="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
BASE="--sb3_full --sample_cmd --fwd_turn --body_frame \
  --cmd_low $CMD_LOW --cmd_high $CMD_HIGH \
  --tilt_deg 17.5 --natural_gait --dangle_penalty --layers $LAYERS \
  --terrain --height_scan --push --obs_noise \
  --climb_reward --soft_landing --gait_phase --body_contact \
  --ascent --ascent_w $ASCENT --ascent_gate 0.2 \
  --symmetry --symmetry_w $SYMM \
  --hip_abduction_w $HIPW --action_rate_w $ACTW --swing_height $SWING"
V28=()
[ -n "$VLAYERS" ] && V28+=(--value_layers "$VLAYERS")
[ -n "$HIPACTW" ] && V28+=(--hip_action_w "$HIPACTW")
[ -n "$ENT" ] && V28+=(--entropy "$ENT")
[ -n "$TBPEN" ] && V28+=(--terminate_body_pen "$TBPEN")
[ -n "$PROGW" ] && V28+=(--progress_w "$PROGW")
[ -n "$CLEARW" ] && V28+=(--clearance_w "$CLEARW")
[ -n "$FPROB" ] && V28+=(--cmd_forward_prob "$FPROB")
[ -n "$CMD_F_LOW" ] && V28+=(--cmd_low_forward "$CMD_F_LOW")
[ -n "$CMD_F_HIGH" ] && V28+=(--cmd_high_forward "$CMD_F_HIGH")
[ -n "$CLEART" ] && V28+=(--clearance_target "$CLEART")
[ -n "$LAZYW" ] && V28+=(--lazy_stop_w "$LAZYW")
[ -n "$TERMW" ] && V28+=(--termination_w "$TERMW")
[ -n "$RFLOOR" ] && V28+=(--reward_clip_min "$RFLOOR")
[ -n "$NACON" ] && V28+=(--naconmax "$NACON")
[ -n "$NJMAX" ] && V28+=(--njmax "$NJMAX")
[ -n "$PHASEW" ] && V28+=(--gait_w "$PHASEW")
RESET_ARGS=()
[ "$FULL_RESET" = "1" ] && RESET_ARGS+=(--full_reset)
[ "$RESETS_PER_EVAL" != "0" ] && RESET_ARGS+=(--num_resets_per_eval "$RESETS_PER_EVAL")
[ "$CURRICULUM" = "1" ] && RESET_ARGS+=(--curriculum --curriculum_init_level "$CURINIT" --curriculum_success_dist "$CURSUCC" --curriculum_min_dist "$CURMIN")
[ -n "$ACCW" ] && V28+=(--action_acc_w "$ACCW")
[ -n "$XYW" ] && V28+=(--xy_angvel_w "$XYW")
[ -n "$ORIW" ] && V28+=(--orientation_w "$ORIW")
[ -n "$EDGE" ] && V28+=(--feet_edge_w "$EDGE")
[ -n "$EDGE_THRESH" ] && V28+=(--feet_edge_thresh "$EDGE_THRESH")
[ "$FEET_EDGE_BG" = "1" ] && V28+=(--feet_edge_bg)
NOISE_ARG=""
[ "$NOISE" != "0" ] && NOISE_ARG="--obs_noise --obs_noise_linvel 0.05 --obs_vel_bias 0.05 --obs_vel_drift 0.04 --obs_vel_delay 1"
RANGE_ARG=""
[ "$RANGE" != "0" ] && RANGE_ARG="--random_init_range $RANGE"
EXTRA=()
[ -n "$BOXGRID" ] && EXTRA+=(--boxgrid_json "$BOXGRID" --contact_mode "$CONTACT")
[ "$RING" = "1" ] && EXTRA+=(--spawn_ring)
[ "$SPAWN_STAIRS" != "0" ] && EXTRA+=(--spawn_stairs_prob "$SPAWN_STAIRS")
REST=()
if [ -n "$SRC" ]; then
  CK=$(readlink -f "$SRC" 2>/dev/null || true)
  if [ -z "$CK" ] || [ ! -d "$CK" ]; then echo "!! 起点不存在: $SRC"; exit 2; fi
  REST+=(--restore "$CK" --step_offset 0)
  say "起点: $CK"
else
  say "从零训练"
fi
say "===== $TAG: $STEPS 步, layers=$LAYERS vly=$VLAYERS, 地形 $SCENE ====="
say "v28 权重: hip=$HIPW action_rate=$ACTW xy_angvel=$XYW orientation=$ORIW swing=$SWING edge=$EDGE"
say "v28 其它: ascent=$ASCENT cmd=$CMD_LOW~$CMD_HIGH spawn_stairs=$SPAWN_STAIRS"
$PY -u -m train.train_go2 --num_timesteps "$STEPS" --num_evals "$NUM_EVALS" \
  --episode_length "$EP_LEN" "${RESET_ARGS[@]}" \
  $BASE $SCAN $LIDAR --terrain_xml "$SCENE" \
  --sb3_envs "$ENVS" --hard_contact --healthy_z_ref feet --lr "$LR" \
  --obs_history_frames "$HIST" $RANGE_ARG $NOISE_ARG \
  "${V28[@]}" "${EXTRA[@]}" "${REST[@]}" \
  --logdir "models/go2_${TAG}_logs" --save_name "go2_walk_${TAG}.pkl" 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! $TAG 未正常完成"; exit 1; fi
say "$TAG 完成"
ls -d "models/go2_${TAG}_logs/checkpoints"/*/ 2>/dev/null | tee -a "$LOG"
