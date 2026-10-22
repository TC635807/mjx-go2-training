#!/usr/bin/env bash
# v24 最终评测矩阵: v7-best vs v6a@7.37M (+可选 v5a), 384 env x 6s,
# 5 地形 x 4 遮挡档 (perfect/nominal/harsh/blind), 使用 v24 修正健康终止判据
# (healthy_z_ref=feet, 四足下方地形最小值)。旧 v6 矩阵数字是 body 判据, 不可直接比。
#
# 用法: bash sim/final_matrix_v24.sh            # 2 策略
#       WITH_V5A=1 bash sim/final_matrix_v24.sh # 加 v5a 对照 (3 策略, 约 +10min)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=${OUT:-logs/go2_v24_matrix.tsv}
CK7=${CK7:-models/go2_v7_logs/checkpoints/best}
CK6=models/go2_v6a_logs/checkpoints/000007372800
CK5=models/go2_v5a_logs/checkpoints/000020070400
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"

# 1) 地形 x 遮挡档矩阵 (warp 训练物理)
SPECS=("v6a_7.4M=$CK6" "v7_best=$CK7")
if [ "${WITH_V5A:-0}" = "1" ]; then SPECS+=("v5a=$CK5"); fi
ENVS=${ENVS:-384} bash sim/eval_occlusion_matrix.sh "$OUT" "${SPECS[@]}"

# 2) 分析表 + 柱状图
echo
"$PY" sim/analyze_occlusion_matrix.py "$OUT" | tee logs/go2_v24_matrix_analysis.md
"$PY" sim/plot_occlusion_matrix.py "$OUT" docs/img/go2-v24-occlusion.png

# 3) 平地 CPU 闭环 (16 seeds x 15 s)
echo
echo "===== 平地 (CPU, 16x15s) =====" | tee logs/go2_v24_flat.txt
for spec in "v6a_7.4M=$CK6" "v7_best=$CK7"; do
  tag=${spec%%:*}; ck=$(readlink -f "${spec#*:}")
  echo "--- $tag ---" | tee -a logs/go2_v24_flat.txt
  JAX_PLATFORMS=cpu timeout 1200 "$PY" sim/eval_go2.py --policy "$ck" \
    --terrain flat --cmd 0.5 0 0 --seeds 16 --duration 15 --vel-frame body $SCAN \
    2>&1 | grep -E '存活率|实际 vx|跟踪误差|躯干相对地面高度|占空比|trot 占比|足端滑移' \
    | tee -a logs/go2_v24_flat.txt
done
echo "DONE -> $OUT"
