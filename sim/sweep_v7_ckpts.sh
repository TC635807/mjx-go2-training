#!/usr/bin/env bash
# v7 检查点扫描: 每个 ckpt 在 v7 训练地形 + boxes_only + stairs_only 上各跑
# 128 env x 300 步 (6s, 训练口径 nominal 遮挡), 输出 TSV 供选点。
# 注意: v24 起健康终止判据改为"四足下方地形最小值"(envs/go2_walk.py healthy_z_ref),
# 与旧日志数字不可直接比。
# 用法: bash sim/sweep_v7_ckpts.sh [out.tsv]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=${1:-logs/go2_v7_ckpt_sweep.tsv}
ENVS=${ENVS:-128}
STEPS=${STEPS:-300}
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
LIDAR="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
printf "ckpt\tterrain\tsurv\tvx\tz\tcurve\n" > "$OUT"
run_one() {
  local tag="$1" ck="$2" scene="$3"
  local out
  out=$($PY -u sim/probe_go2_warp_eval.py --policy "$ck" --envs "$ENVS" \
        --steps "$STEPS" --spawn random --cmd 0.5 0 0 --foot-contact hard \
        --terrain_xml "$scene" $SCAN $LIDAR 2>&1 \
        | grep -vE 'Module |load on device|height field collision|decrease the number')
  local surv vx z curve
  surv=$(echo "$out" | sed -n 's/.*存活率.*= *\([0-9.]*\)%.*/\1/p')
  vx=$(echo "$out" | sed -n 's/.*机体系 vx = \([+-][0-9.]*\).*/\1/p')
  z=$(echo "$out" | sed -n 's/.*躯干 z = \([0-9.]*\).*/\1/p')
  curve=$(echo "$out" | sed -n 's/.*存活曲线: //p' | tr -d ' ')
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$(basename "$scene" .xml | sed 's/scene_mjx_terrain_//')" \
    "${surv:-NA}" "${vx:-NA}" "${z:-NA}" "${curve:-NA}" >> "$OUT"
  echo "[$(date +%H:%M:%S)] $tag $(basename "$scene") surv=${surv:-NA}%"
}
declare -a SPECS=(
  "v6a@7.37M:models/go2_v6a_logs/checkpoints/000007372800"
)
for d in models/go2_v7_logs/checkpoints/*/; do
  [ -d "$d" ] || continue
  st=$(basename "$d")
  [ "$st" = "best" ] && continue
  SPECS+=("v7@$st:$d")
done
for spec in "${SPECS[@]}"; do
  tag=${spec%%:*}; path=${spec#*:}
  CK=$(readlink -f "$path" 2>/dev/null || true)
  if [ -z "$CK" ] || [ ! -d "$CK" ]; then echo "!! skip $tag ($path)"; continue; fi
  run_one "$tag" "$CK" models/go2/scene_mjx_terrain_v7.xml
  run_one "$tag" "$CK" models/go2/scene_mjx_terrain_boxes_only.xml
  run_one "$tag" "$CK" models/go2/scene_mjx_terrain_stairs_only.xml
done
echo "DONE -> $OUT"
