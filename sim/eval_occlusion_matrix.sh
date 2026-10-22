#!/usr/bin/env bash
# 遮挡鲁棒性矩阵评测: 策略 x 地形 x 遮挡档 (warp 训练物理, ENVS x 300 步 = 6s)
# 用法: [ENVS=384] bash sim/eval_occlusion_matrix.sh [out.tsv] [policy=path ...]
#   默认 ENVS=128 (快速筛查); 结论性对比务必 ENVS=384 (单点噪声从 ±4~6pp 降到 ±2~3pp)
# 遮挡档:
#   perfect = 完美解析扫描 (v1-v5 训练口径) —— 上界
#   nominal = v6 训练口径 (occ_recall .5 / dropout .05 / 扇区 .25 / 安装抖动)
#   harsh   = 最坏情况 (occ_recall 0 = 只用当前可见格点 / dropout .20 / 扇区 .5)
#   blind   = 全 0 扫描 (dropout 1.0 -> 所有格点填 fill=0) —— 下界, "瞎走"
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=${1:-logs/go2_occlusion_matrix.tsv}
ENVS=${ENVS:-128}
shift || true
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
# v24: 额外 probe 参数 (例如 SPAWN_EXTRA="--spawn_max_slope 0.25" 做"有效出生"口径)
SPAWN_EXTRA=${SPAWN_EXTRA:-}
declare -A CFG=(
  [perfect]=""
  [nominal]="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
  [harsh]="--lidar --lidar_occ_recall 0.0 --lidar_dropout 0.20 --lidar_sector_prob 0.5"
  [blind]="--lidar --lidar_occ_recall 0.0 --lidar_dropout 1.0"
)
if [ "$#" -eq 0 ]; then
  set -- v5a_best=models/go2_v5a_logs/checkpoints/best \
         v5c_best=models/go2_v5c_logs/checkpoints/best
fi
printf "policy\tterrain\tcfg\tsurv\tvx\tz\tscan_vis\tscan_info\n" > "$OUT"
for spec in "$@"; do
  p=${spec%%=*}; path=${spec#*=}
  CK=$(readlink -f "$path" 2>/dev/null || true)
  if [ -z "$CK" ] || [ ! -d "$CK" ]; then echo "!! missing $p ($path)"; continue; fi
  for t in mix stairs slope rough boxes; do
    scene=models/go2/scene_mjx_terrain_${t}_only.xml
    [ "$t" = mix ] && scene=models/go2/scene_mjx_terrain_mix.xml
    for c in perfect nominal harsh blind; do
      out=$($PY sim/probe_go2_warp_eval.py --policy "$CK" --envs "$ENVS" --steps 300 \
            --spawn random --cmd 0.5 0 0 --foot-contact hard --terrain_xml "$scene" \
            $SCAN ${CFG[$c]} $SPAWN_EXTRA 2>&1 | grep -v 'Module .* load on device')
      surv=$(echo "$out" | sed -n 's/.*存活率.*= *\([0-9.]*\)%.*/\1/p')
      vx=$(echo "$out" | sed -n 's/.*机体系 vx = \([+-][0-9.]*\).*/\1/p')
      z=$(echo "$out" | sed -n 's/.*躯干 z = \([0-9.]*\).*/\1/p')
      sv=$(echo "$out" | sed -n 's/.*物理可见率 = \([0-9.]*\).*/\1/p')
      si=$(echo "$out" | sed -n 's/.*信息可用率 = \([0-9.]*\).*/\1/p')
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$p" "$t" "$c" \
        "${surv:-NA}" "${vx:-NA}" "${z:-NA}" "${sv:-NA}" "${si:-NA}" >> "$OUT"
      echo "[$(date +%H:%M:%S)] $p $t $c surv=${surv:-NA}% vx=${vx:-NA} sv=${sv:-NA} si=${si:-NA}"
    done
  done
done
echo "DONE -> $OUT"
