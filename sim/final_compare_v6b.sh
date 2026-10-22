#!/usr/bin/env bash
# v6b 最终对照: v6b_best vs v6a@7.37M vs v5a, 384 env x 5 地形 x 4 遮挡档。
#
# ⚠️ 占 GPU; 必须串行:
#   bash sim/gpu_guard.sh run -- bash sim/final_compare_v6b.sh <v6b_ckpt>
#
# 输出:
#   logs/go2_v6b_final_matrix.tsv / _analysis.md / docs/img/go2-v6b-occlusion.png
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
CK6B="${1:-}"
if [ -z "$CK6B" ] || [ ! -d "$CK6B" ]; then
  # 从 v6b 日志里解析最佳检查点
  CK6B=$(sed -n 's/.*ckpt=\(.*\)$/\1/p' logs/go2_v6b_harsh.log | tail -1)
fi
[ -d "$CK6B" ] || { echo "找不到 v6b 检查点 (参数或日志解析失败): $CK6B"; exit 1; }
CK6B=$(readlink -f "$CK6B")
echo "[compare] v6b = $CK6B"

ENVS=384 bash sim/eval_occlusion_matrix.sh logs/go2_v6b_final_matrix.tsv \
  "v6b_best=$CK6B" \
  "v6a_7.4M=models/go2_v6a_logs/checkpoints/000007372800" \
  "v5a_final=models/go2_v5a_logs/checkpoints/000020070400"

$PY sim/analyze_occlusion_matrix.py logs/go2_v6b_final_matrix.tsv
$PY sim/plot_occlusion_matrix.py logs/go2_v6b_final_matrix.tsv docs/img/go2-v6b-occlusion.png

echo "===== v6b 平地 (CPU 16x15s) ====="
JAX_PLATFORMS=cpu timeout 1200 $PY sim/eval_go2.py --policy "$CK6B" \
  --terrain flat --cmd 0.5 0 0 --seeds 16 --duration 15 --vel-frame body \
  --scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5 2>&1 \
  | grep -E '存活率|实际 vx|跟踪误差|躯干相对|占空比|trot|滑移|roll ' || true
echo "[compare] 完成"
