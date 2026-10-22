#!/usr/bin/env bash
# v6a: 从 v5a-best (20.07M) 续训 12M —— 首次**雷达遮挡感知**训练 (真机 L1 几何)
#
# 遮挡模型 (envs/go2_walk.py: height_scan.lidar):
#   * 物理可见性: 机头前装传感器 mount=(0.289,0,-0.047), 垂直 FOV -60°..+30°
#     (Unitree L1 360°x90°, 装配下倾 15°), 机器人碰撞 geom 精确射线求交
#     (box slab / capsule Ericson / sphere), 地形自遮挡 4 点采样;
#   * 高程图记忆 occ_recall 0.5 (物理不可见格点仍有一半概率是"记得的地形", 否则填 0);
#   * 随机丢点 5% + 成片扇区缺失 p=0.25 (扇区宽度 U(0.6,2.4) rad);
#   * 安装抖动 mount ±2cm / vfov ±0.15rad。
# 单变量 = 感知通道: 地形/奖励/物理/env 数/网络结构全部与 v5a 相同。
#
# 用法: bash sim/run_go2_v6a_lidar.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
LOG=logs/go2_v6a_lidar.log
mkdir -p logs
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

CKPT=$(readlink -f models/go2_v5a_logs/checkpoints/000020070400)
say "===== v6a: 从 v5a 20.07M 续训 12M, 雷达遮挡感知 (ckpt=$CKPT) ====="
say "遮挡配置: $LIDAR"
$PY -u -m train.train_go2 --num_timesteps 12000000 --num_evals 6 \
  $BASE $SCAN $LIDAR --terrain_xml models/go2/scene_mjx_terrain_mix.xml \
  --sb3_envs 2560 --hard_contact \
  --restore "$CKPT" --step_offset 20070400 --lr 1e-4 \
  --logdir models/go2_v6a_logs --save_name go2_walk_v6a.pkl 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! v6a 未正常完成, 中止评测"; exit 1; fi
say "训练完成; 开始评测 (warp 遮挡矩阵 + 平地)"

CK6=$(readlink -f models/go2_v6a_logs/checkpoints/best)
say "v6a best=$CK6"

$PY sim/status_go2_train.py 2>/dev/null | head -5 | tee -a "$LOG" || true

say "===== 遮挡矩阵: v6a-best vs v5a-best (warp, 128 env x 6s) ====="
bash sim/eval_occlusion_matrix.sh logs/go2_v6a_occlusion.tsv \
  v6a_best=$CK6 v5a_best=models/go2_v5a_logs/checkpoints/000020070400 2>&1 | tee -a "$LOG"

say "===== 平地 (CPU, 16x15s) ====="
JAX_PLATFORMS=cpu timeout 1200 $PY sim/eval_go2.py --policy "$CK6" \
  --terrain flat --cmd 0.5 0 0 --seeds 16 --duration 15 --vel-frame body $SCAN \
  2>&1 | tee -a "$LOG" | grep -E '存活率|实际 vx|跟踪误差|躯干相对地面高度|占空比|trot 占比|足端滑移' || true

say "===== 训练溢出/NaN 自检 ====="
echo "hfield_overflow=$(grep -c 'height field collision overflow' "$LOG" || true)" | tee -a "$LOG"
echo "broadphase=$(grep -c 'broadphase overflow' "$LOG" || true)" | tee -a "$LOG"
echo "nan_in_log=$(grep -c 'loss=nan' "$LOG" || true)" | tee -a "$LOG"
say "v6a 全部完成"
