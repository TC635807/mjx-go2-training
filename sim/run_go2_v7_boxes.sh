#!/usr/bin/env bash
# v7: boxes/stairs 专项续训 (2026-10-19) —— 修完 v24 终止判据后的第一轮地形补短
#
# 动机 (摔倒诊断 sim/diag_falls.sh, v6a@7.37M, 384 env):
#   * 旧判据 (躯干下方地形做 z 基准) 在 boxes 上 50% 的终止是假摔 (躯干跨箱顶);
#     已修为 v24 "四足下方地形最小值" 基准 (envs/go2_walk.py: healthy_z_ref)。
#   * 修完后 boxes 仍有 ~34% 真摔, 终止原因几乎全是 tilt -> 踩方块棱被绊倒。
#   * 训练地形 mix 里 boxes 只占 1/9、最高 0.085 m; 评测 boxes_only 最高 0.113 m。
# 本实验 (单变量 = 地形): 只换 terrain_xml 为 v7 (boxes 4/9 + stairs 2/9 +
# rough/slope/flat 各 1/9, 硬度 1.25 -> boxes 0.089/0.115/0.15 m, 台阶 0.074/0.10 m),
# 其余 (奖励/网络/遮挡模型/超参) 与 v6a 完全相同。
#
# 用法: bash sim/run_go2_v7_boxes.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
LOG=logs/go2_v7_train.log
mkdir -p logs models/go2_v7_logs
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
if [ ! -d "$CKPT" ]; then echo "!! 找不到 v6a 起点 $CKPT"; exit 2; fi
say "===== v7: 从 v6a@7.37M 续训 8M, 地形 v7 (boxes 4/9), 修后终止判据 feet ====="
say "起点: $CKPT"
say "地形: models/go2/scene_mjx_terrain_v7.xml (preset v7, 128x128, 0.094m/格)"
say "遮挡: $LIDAR"

$PY -u -m train.train_go2 --num_timesteps 8000000 --num_evals 5 \
  $BASE $SCAN $LIDAR --terrain_xml models/go2/scene_mjx_terrain_v7.xml \
  --sb3_envs 2560 --hard_contact --healthy_z_ref feet \
  --restore "$CKPT" --step_offset 7372800 --lr 1e-4 \
  --logdir models/go2_v7_logs --save_name go2_walk_v7.pkl 2>&1 | tee -a "$LOG"

if ! grep -q '训练完成' "$LOG"; then say "!! v7 未正常完成"; exit 1; fi
say "训练完成"
echo "hfield_overflow=$(grep -c 'height field collision overflow' "$LOG" || true)" | tee -a "$LOG"
echo "broadphase=$(grep -c 'broadphase overflow' "$LOG" || true)" | tee -a "$LOG"
echo "nan=$(grep -c 'loss=nan' "$LOG" || true)" | tee -a "$LOG"
say "v7 训练结束; 检查点目录:"; ls -d models/go2_v7_logs/checkpoints/*/ 2>/dev/null | tee -a "$LOG"
