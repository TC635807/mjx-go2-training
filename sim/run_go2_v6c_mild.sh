#!/usr/bin/env bash
# v6c: v5a -> 轻量遮挡微调 6M (mild occlusion fine-tune)
#
# 动机 (§4.5/§4.6): v6a 证明了"遮挡训练能显著改善盲扫鲁棒性"(blind mix +7pp,
# boxes +9.4pp), 但重遮挡 (occ_recall .5 / sector .25) 可能损伤 clean 能力,
# 且"训得更久"本身会让随机出生存活率漂移下降。所以本段:
#   * 起点 = v5a-final (当前地形最强, mix perfect 83.6)
#   * 遮挡**减弱**: occ_recall .8 / dropout .03 / sector_prob .10 / ray_samples 4
#   * 只训 6M, lr 5e-5 (低步数+低 lr, 抑制漂移)
#   * 地形 mix / 其余配方与 v6a 完全一致 (单变量 = 遮挡强度与步数)
#
# 用法: bash sim/run_go2_v6c_mild.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
LOG=logs/go2_v6c_mild.log
mkdir -p logs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
LIDAR="--lidar --lidar_occ_recall 0.8 --lidar_dropout 0.03 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.10 --lidar_sector_prob 0.10"
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
say "===== v6c: 轻遮挡微调 6M (ckpt=$CKPT) ====="
say "遮挡: $LIDAR"
$PY -u -m train.train_go2 --num_timesteps 6000000 --num_evals 6 \
  $BASE $SCAN $LIDAR --terrain_xml models/go2/scene_mjx_terrain_mix.xml \
  --sb3_envs 2560 --hard_contact \
  --restore "$CKPT" --step_offset 20070400 --lr 5e-5 \
  --logdir models/go2_v6c_logs --save_name go2_walk_v6c.pkl 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! v6c 未正常完成"; exit 1; fi
say "训练完成; 检查点 = models/go2_v6c_logs/checkpoints/"
ls -d models/go2_v6c_logs/checkpoints/*/ | tee -a "$LOG"
say "v6c 完成 (评测由下一段单独跑, 避免与本脚本的 GPU 时间冲突)"
