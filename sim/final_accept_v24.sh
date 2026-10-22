#!/usr/bin/env bash
# v24 最终验收: 一条命令跑完
#   1) 遮挡矩阵 (5 地形 + v7/v8 训练地形) x 4 遮挡档, 384 env 同进程
#   2) 有效出生 (spawn_max_slope=0.25) 补充表
#   3) 平地 CPU 16x15s
# 用法:
#   POLS="v6a_7.4M=...,v8_best=..." bash sim/final_accept_v24.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUTDIR=${OUTDIR:-logs/go2_final_v24}
mkdir -p "$OUTDIR"
POLS=${POLS:-"v6a_7.4M=models/go2_v6a_logs/checkpoints/000007372800,v8_best=models/go2_v8_logs/checkpoints/best"}
export POLS

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "===== 1/3 遮挡矩阵 (384 env, 同进程多策略) ====="
OUT="$OUTDIR/matrix.tsv" bash sim/matrix_fast_v24.sh 2>&1 | tee "$OUTDIR/matrix.log" | tail -3
cp -f logs/go2_v24_matrix.tsv "$OUTDIR/matrix.tsv" 2>/dev/null || true
cp -f logs/go2_v24_matrix_analysis.md "$OUTDIR/matrix_analysis.md" 2>/dev/null || true

say "===== 2/3 有效出生 (spawn_max_slope=0.25) ====="
bash sim/eval_spawn_valid.sh "$OUTDIR/validspawn.tsv" 2>&1 | tee "$OUTDIR/validspawn.log" | tail -20

say "===== 3/3 平地 (CPU, 16x15s) ====="
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
for spec in $POLS; do
  tag="${spec%%=*}"; path="${spec#*=}"
  ck="$(readlink -f "$path" 2>/dev/null || true)"
  [ -z "$ck" ] && continue
  echo "--- $tag ---" | tee -a "$OUTDIR/flat.txt"
  JAX_PLATFORMS=cpu timeout 1200 "$PY" sim/eval_go2.py --policy "$ck" \
    --terrain flat --cmd 0.5 0 0 --seeds 16 --duration 15 --vel-frame body $SCAN \
    2>&1 | grep -E '存活率|实际 vx|跟踪误差|躯干相对地面高度|占空比|trot 占比|足端滑移' \
    | tee -a "$OUTDIR/flat.txt"
done
say "全部完成 -> $OUTDIR"
