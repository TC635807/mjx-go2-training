#!/usr/bin/env bash
# Go2 v5: 地形鲁棒性课程 (两段式) —— 依据开源做法 (legged_gym/mjlab/IsaacLab):
#   * 混合地形 + 按比例加权 (台阶/斜坡为主), 而不是只训单一地形;
#   * 难度分档 + 课程 (本仓库没有 per-env 地形, 用"两段式宏观课程"近似);
#   * 高度扫描放大到官方 Go2 配置量级: mjlab/unitree Go2 = 1.6m x 1.0m @ 0.1m
#     (17x11=187 点, ray_alignment=yaw); legged_gym = 1.6x1.0m 17x11。我们原来只有
#     7x5=35 点 / 0.2m 间距, 是 4.6 倍稀疏 -> 这次换成 17x11 (obs 243)。
#
# Stage A (20M): 普通混合地形 mix (n=128) —— 重建带新扫描的基础行走能力
# Stage B (20M): v5 台阶/斜坡加权 + 硬度 1.5 的地形 (n=128) —— 专项强化
# Stage B 之后: per-terrain-type 评测 (warp) + 平地 CPU 评测
#
# 用法: bash sim/run_go2_v5_curriculum.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
LOG=logs/go2_v5_curriculum.log
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

say "===== Stage A: 20M, 地形 mix, 扫描 17x11 ====="
$PY -u -m train.train_go2 --num_timesteps 20000000 --num_evals 8 \
  $BASE $SCAN --terrain_xml models/go2/scene_mjx_terrain_mix.xml \
  --sb3_envs 2560 --hard_contact \
  --logdir models/go2_v5a_logs --save_name go2_walk_v5a.pkl 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! Stage A 未正常完成, 中止"; exit 1; fi
CKPT="$(ls -d models/go2_v5a_logs/checkpoints/*/ | grep -E '/[0-9]{9,12}/' | sort | tail -1)"
CKPT="$(readlink -f "$CKPT")"
say "Stage A 完成; ckpt=$CKPT"

say "===== Stage B: 20M, 地形 v5 (台阶/斜坡加权, 硬度1.5), 续训 ====="
$PY -u -m train.train_go2 --num_timesteps 20000000 --num_evals 8 \
  $BASE $SCAN --terrain_xml models/go2/scene_mjx_terrain_v5.xml \
  --sb3_envs 2560 --hard_contact \
  --restore "$CKPT" --step_offset 20000000 --lr 2e-4 \
  --logdir models/go2_v5b_logs --save_name go2_walk_v5.pkl 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! Stage B 未正常完成, 中止"; exit 1; fi
say "Stage B 完成; 开始评测"

{
  echo "########## v5 平地 (训练 XML, 16x15s) ##########"
  JAX_PLATFORMS=cpu timeout 900 $PY sim/eval_go2.py --policy policies/go2_walk_v5.pkl \
      --terrain flat --cmd 0.5 0 0 --seeds 16 --duration 15 --vel-frame body 2>&1 | tail -26
  echo; echo "########## v5 混合地形 mix (CPU, 16x6s) ##########"
  JAX_PLATFORMS=cpu timeout 900 $PY sim/eval_go2.py --policy policies/go2_walk_v5.pkl \
      --xml models/go2/scene_mjx_terrain_mix.xml --cmd 0.5 0 0 --seeds 16 --duration 6 --vel-frame body 2>&1 | tail -24
} > logs/go2_v5_eval_cpu.txt 2>&1

run_probe() {
  local tag="$1" pol="$2" scene="$3"
  local out
  out=$(timeout 900 $PY sim/probe_go2_warp_eval.py --policy "$pol" --envs 32 --steps 300 \
        --spawn random --cmd 0.5 0 0 --foot-contact hard --terrain_xml "$scene" 2>&1 \
        | grep -E '存活率|存活期机体系 vx' | tr '\n' ' ')
  echo "$tag | $out"
}
{
  echo "===== v5 per-type (warp, random spawn, 32 env x 6s) ====="
  run_probe v5_mix     policies/go2_walk_v5.pkl models/go2/scene_mjx_terrain_mix.xml
  run_probe v5_hardmap policies/go2_walk_v5.pkl models/go2/scene_mjx_terrain_v5.xml
  for t in stairs slope rough boxes; do
    run_probe "v5_$t" policies/go2_walk_v5.pkl "$(printf 'models/go2/scene_mjx_terrain_%s_only.xml' "$t")"
  done
  echo
  echo "===== v4 对照 (同图同口径) ====="
  run_probe v4_mix     policies/go2_walk_v4.pkl models/go2/scene_mjx_terrain_mix.xml
  run_probe v4_hardmap policies/go2_walk_v4.pkl models/go2/scene_mjx_terrain_v5.xml
  for t in stairs slope rough boxes; do
    run_probe "v4_$t" policies/go2_walk_v4.pkl "$(printf 'models/go2/scene_mjx_terrain_%s_only.xml' "$t")"
  done
} > logs/go2_v5_eval_warp.txt 2>&1

say "评测完成: logs/go2_v5_eval_cpu.txt / logs/go2_v5_eval_warp.txt"
grep -E 'v5_|v4_' logs/go2_v5_eval_warp.txt | tee -a "$LOG"
