#!/usr/bin/env bash
# v6b 收尾: (1) 用**实际存在的检查点**按地形存活率选优; (2) 与 v6a/v5a 做 384 env
# 全遮挡矩阵; (3) 平地 CPU 评测。
#
# 为什么单独一个脚本: v6b 训练脚本里预写的检查点步号 (2M/4M/6M/8M) 与 brax 实际
# 落盘点 (eval 间隔 2.70336M) 不一致, 且改运行中的脚本不安全 -> 训练结束后用这个
# 脚本按目录名动态扫描。
#
# ⚠️ 占 GPU; 必须串行:
#   bash sim/gpu_guard.sh run -- bash sim/sweep_v6b_and_compare.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
LOG=logs/go2_v6b_sweep_compare.log
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
LIDAR="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- (1) 实际检查点列表 (12 位数字目录, 按步数排序; 排除 best 副本避免重复) ----
mapfile -t CKS < <(ls -d models/go2_v6b_logs/checkpoints/[0-9]*/ 2>/dev/null \
                   | sed 's#/$##' | sort -t/ -k5 -n)
[ "${#CKS[@]}" -gt 0 ] || { say "!! 没有 v6b 检查点"; exit 1; }
say "v6b 检查点: ${CKS[*]}"

OUT=logs/go2_v6b_ckpt_sweep.tsv
printf "ckpt\tmix\tstairs\tboxes\tsum\n" > "$OUT"
BEST_CK="" ; BEST_SUM=-1 ; BEST_STEP=""
for ck in "${CKS[@]}"; do
  step=$(basename "$ck")
  mix=0; st=0; bx=0
  for spec in "mix:models/go2/scene_mjx_terrain_mix.xml" \
              "stairs:models/go2/scene_mjx_terrain_stairs_only.xml" \
              "boxes:models/go2/scene_mjx_terrain_boxes_only.xml"; do
    t=${spec%%:*}; scene=${spec#*:}
    out=$($PY sim/probe_go2_warp_eval.py --policy "$ck" --envs 384 --steps 300 \
          --spawn random --cmd 0.5 0 0 --foot-contact hard --terrain_xml "$scene" \
          $SCAN $LIDAR 2>&1 | grep -vE 'Module |load on device')
    surv=$(echo "$out" | sed -n 's/.*存活率.*= *\([0-9.]*\)%.*/\1/p')
    say "  sweep $step $t surv=${surv:-NA}%"
    case "$t" in
      mix) mix=${surv:-0} ;;
      stairs) st=${surv:-0} ;;
      boxes) bx=${surv:-0} ;;
    esac
  done
  sum=$($PY -c "print(round($mix+$st+$bx,1))")
  printf "%s\t%s\t%s\t%s\t%s\n" "$step" "$mix" "$st" "$bx" "$sum" >> "$OUT"
  say "  => @$step mix=$mix stairs=$st boxes=$bx sum=$sum"
  if $PY -c "import sys; sys.exit(0 if $sum > $BEST_SUM else 1)"; then
    BEST_CK="$ck"; BEST_STEP="$step"; BEST_SUM="$sum"
  fi
done
say "v6b 最佳检查点 = $BEST_STEP (sum=$BEST_SUM) ckpt=$BEST_CK"
echo "$BEST_CK" > /tmp/go2_v6b_best_ck.txt

# ---- (2) 全遮挡矩阵: v6b_best vs v6a_7.4M vs v5a ----
say "===== 384 env x 5 地形 x 4 遮挡档: v6b vs v6a vs v5a ====="
ENVS=384 bash sim/eval_occlusion_matrix.sh logs/go2_v6b_final_matrix.tsv \
  "v6b_best=$BEST_CK" \
  "v6a_7.4M=models/go2_v6a_logs/checkpoints/000007372800" \
  "v5a_final=models/go2_v5a_logs/checkpoints/000020070400" 2>&1 | tee -a "$LOG"
$PY sim/analyze_occlusion_matrix.py logs/go2_v6b_final_matrix.tsv | tee -a "$LOG"
$PY sim/plot_occlusion_matrix.py logs/go2_v6b_final_matrix.tsv docs/img/go2-v6b-occlusion.png

# ---- (3) 平地 ----
say "===== v6b best 平地 (CPU 16x15s) ====="
JAX_PLATFORMS=cpu timeout 1200 $PY sim/eval_go2.py --policy "$BEST_CK" \
  --terrain flat --cmd 0.5 0 0 --seeds 16 --duration 15 --vel-frame body $SCAN 2>&1 \
  | tee -a "$LOG" | grep -E '存活率|实际 vx|跟踪误差|躯干相对|占空比|trot|滑移|roll ' || true
say "v6b 收尾全部完成"
