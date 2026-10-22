#!/usr/bin/env bash
# 诊断: (1) v5a 完整遮挡矩阵 (补齐之前 probe 崩溃的行); (2) v6a 中间检查点在完美扫描下
# 的地形存活率 -> 判断"继续训练是否本身就在退化"(与遮挡无关的漂移)。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
OUT=logs/go2_diag_ckpt_sweep.txt
: > "$OUT"

echo "===== (1) v5a 完整遮挡矩阵 (4 档 x 5 地形, 128 env) =====" | tee -a "$OUT"
bash sim/eval_occlusion_matrix.sh logs/go2_v5a_occlusion_full.tsv \
  v5a_best=models/go2_v5a_logs/checkpoints/000020070400 2>&1 | tee -a "$OUT"

echo "===== (2) v6a 检查点扫描 (完美扫描, mix + stairs, 128 env) =====" | tee -a "$OUT"
probe_perfect() {
  local tag="$1" ck="$2" scene="$3"
  local out
  out=$(timeout 600 "$PY" sim/probe_go2_warp_eval.py --policy "$ck" \
        --envs 128 --steps 300 --spawn random --cmd 0.5 0 0 --foot-contact hard \
        --terrain_xml "$scene" $SCAN 2>&1 | grep -vE 'Module |load on device')
  echo "$tag | $(echo "$out" | grep -E '存活率|存活曲线' | tr '\n' ' ')" | tee -a "$OUT"
}
for spec in "v5a:models/go2_v5a_logs/checkpoints/000020070400" \
            "v6a@2.5M:models/go2_v6a_logs/checkpoints/000002457600" \
            "v6a@4.9M:models/go2_v6a_logs/checkpoints/000004915200" \
            "v6a@7.4M:models/go2_v6a_logs/checkpoints/000007372800" \
            "v6a@9.8M:models/go2_v6a_logs/checkpoints/000009830400" \
            "v6a@12.3M:models/go2_v6a_logs/checkpoints/000012288000" \
            "v6a_best:models/go2_v6a_logs/checkpoints/best"; do
  tag=${spec%%:*}; ck=$(readlink -f "${spec#*:}")
  probe_perfect "${tag}_mix" "$ck" models/go2/scene_mjx_terrain_mix.xml
  probe_perfect "${tag}_stairs" "$ck" models/go2/scene_mjx_terrain_stairs_only.xml
done
echo "DONE" >> "$OUT"
