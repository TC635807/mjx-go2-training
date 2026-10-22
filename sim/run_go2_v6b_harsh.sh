#!/usr/bin/env bash
# v6b: 从 v6a 冠军检查点 (@7.37M) 继续做**更狠的遮挡课程** 8M —— 目标: 把
# harsh/blind 鲁棒性再抬 3~5pp, 同时不伤 clean 地形能力。
#
# 单变量 = 遮挡强度 (地形/奖励/物理/网络全部不动):
#   occ_recall 0.5 -> 0.25 / dropout .05 -> .10 / sector .25 -> .40 (宽 1.0~3.0rad)
#   地形自遮挡采样 4 -> 5 点 / mount 抖动 .02 -> .03 / vfov 抖动 .15 -> .20
#
# ⚠️ 本脚本占 GPU; 必须通过守门员串行启动, 不要与其他训练/评测并发:
#   bash sim/gpu_guard.sh run -- bash sim/run_go2_v6b_harsh.sh
#
# 评测: 训练后按**地形存活率**选检查点 (eval_reward 不可信, 见 v6 实验记录 §4.5):
#   2M/4M/6M/8M 检查点 x (mix, stairs, boxes) x 384 env x nominal 遮挡。
#
# 用法: bash sim/gpu_guard.sh run -- bash sim/run_go2_v6b_harsh.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
LOG=logs/go2_v6b_harsh.log
mkdir -p logs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
LIDAR="--lidar --lidar_occ_recall 0.25 --lidar_dropout 0.10 --lidar_mount_jitter 0.03 --lidar_vfov_jitter 0.20 --lidar_sector_prob 0.40 --lidar_sector_width 1.0 3.0 --lidar_ray_samples 5"
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
say "===== v6b: 硬遮挡课程续训 8M (起点 v6a@7.37M, ckpt=$CKPT) ====="
say "遮挡: $LIDAR"
$PY -u -m train.train_go2 --num_timesteps 8000000 --num_evals 4 \
  $BASE $SCAN $LIDAR --terrain_xml models/go2/scene_mjx_terrain_mix.xml \
  --sb3_envs 2560 --hard_contact \
  --restore "$CKPT" --step_offset 27443200 --lr 4e-5 \
  --logdir models/go2_v6b_logs --save_name go2_walk_v6b.pkl 2>&1 | tee -a "$LOG"
if ! grep -q '训练完成' "$LOG"; then say "!! v6b 未正常完成"; exit 1; fi
say "训练完成; 按**地形存活率**选检查点 (384 env, nominal 遮挡)"

OUT=logs/go2_v6b_ckpt_sweep.tsv
printf "ckpt\tmix\tstairs\tboxes\tsum\n" > "$OUT"
BEST="" ; BEST_SUM=-1 ; BEST_CK=""
for step in 2048000 4096000 6144000 8000000; do
  ck=$(readlink -f models/go2_v6b_logs/checkpoints/$(printf '%012d' "$step") 2>/dev/null || true)
  [ -d "$ck" ] || { say "!! 缺检查点 $step"; continue; }
  mix=0; st=0; bx=0
  for spec in "mix:models/go2/scene_mjx_terrain_mix.xml" \
              "stairs:models/go2/scene_mjx_terrain_stairs_only.xml" \
              "boxes:models/go2/scene_mjx_terrain_boxes_only.xml"; do
    t=${spec%%:*}; scene=${spec#*:}
    out=$($PY sim/probe_go2_warp_eval.py --policy "$ck" --envs 384 --steps 300 \
          --spawn random --cmd 0.5 0 0 --foot-contact hard --terrain_xml "$scene" \
          $SCAN $LIDAR 2>&1 | grep -vE 'Module |load on device')
    surv=$(echo "$out" | sed -n 's/.*存活率.*= *\([0-9.]*\)%.*/\1/p')
    echo "  $step $t surv=${surv:-NA}%"
    case "$t" in
      mix) mix=${surv:-0} ;;
      stairs) st=${surv:-0} ;;
      boxes) bx=${surv:-0} ;;
    esac
  done
  sum=$($PY -c "print(round($mix+$st+$bx,1))")
  printf "%s\t%s\t%s\t%s\t%s\n" "$step" "$mix" "$st" "$bx" "$sum" >> "$OUT"
  say "  @$step: mix=$mix stairs=$st boxes=$bx sum=$sum"
  if $PY -c "import sys; sys.exit(0 if $sum > $BEST_SUM else 1)"; then
    BEST="$step"; BEST_SUM="$sum"; BEST_CK="$ck"
  fi
done
say "v6b 最佳检查点 = $BEST (sum=$BEST_SUM), ckpt=$BEST_CK"
say "sweep 表: $OUT"
say "v6b 完成 (与 v6a/v5a 的最终矩阵对照另行单跑)"
