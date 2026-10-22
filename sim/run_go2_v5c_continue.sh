#!/usr/bin/env bash
# v5c: 把 v5a (大扫描 187 点 + mix, 20M) 再训 20M —— 检验"20M 是否欠训".
# 单变量 = 训练步数 (扫描/地形/奖励全同 v5a). 之后跑 128-env per-type + 平地评测.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
LOG=logs/go2_v5c_continue.log
mkdir -p logs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
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
say "===== v5c: 从 v5a 20M 续训 +20M (地形 mix, 大扫描) ckpt=$CKPT ====="
$PY -u -m train.train_go2 --num_timesteps 20000000 --num_evals 8 \
  $BASE $SCAN --terrain_xml models/go2/scene_mjx_terrain_mix.xml \
  --sb3_envs 2560 --hard_contact \
  --restore "$CKPT" --step_offset 20070400 --lr 2e-4 \
  --logdir models/go2_v5c_logs --save_name go2_walk_v5c.pkl 2>&1 | tee -a "$LOG"
grep -q '训练完成' "$LOG" || { say "!! v5c 未完成, 退出"; exit 1; }
CK=$(readlink -f models/go2_v5c_logs/checkpoints/best)
say "v5c 训练完成; best=$CK"

echo "===== 评测 (128 env, warp) =====" | tee -a "$LOG"
probe() {
  local tag="$1" pol="$2" scene="$3"
  local out
  out=$(timeout 900 $PY sim/probe_go2_warp_eval.py --policy "$pol" --envs 128 --steps 300 \
        --spawn random --cmd 0.5 0 0 --foot-contact hard --terrain_xml "$scene" $SCAN 2>&1 \
        | grep -E '存活率' | tr '\n' ' ')
  echo "$tag | $out" | tee -a "$LOG"
}
probe v5c_mix     "$CK" models/go2/scene_mjx_terrain_mix.xml
probe v5c_stairs  "$CK" models/go2/scene_mjx_terrain_stairs_only.xml
probe v5c_slope   "$CK" models/go2/scene_mjx_terrain_slope_only.xml
probe v5c_boxes   "$CK" models/go2/scene_mjx_terrain_boxes_only.xml
probe v5c_hardmap "$CK" models/go2/scene_mjx_terrain_v5.xml
echo "===== 平地 (CPU 15s x 16) =====" | tee -a "$LOG"
JAX_PLATFORMS=cpu timeout 900 $PY sim/eval_go2.py --policy "$CK" --terrain flat \
  --cmd 0.5 0 0 --seeds 16 --duration 15 --vel-frame body $SCAN 2>&1 | tee -a "$LOG" \
  | grep -E '存活率|实际 vx|跟踪误差|占空比|trot 占比' || true
say "v5c 全部完成"
