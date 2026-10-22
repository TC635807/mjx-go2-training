#!/usr/bin/env bash
# 最终高功效遮挡矩阵: v6a@7.4M (冠军候选) vs v5a-final, 5 地形 x 4 遮挡档 x 384 env
# 输出: logs/go2_v6_final_matrix.tsv
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=logs/go2_v6_final_matrix.tsv
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
ENVS=${ENVS:-384}
declare -A CFG=(
  [perfect]=""
  [nominal]="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
  [harsh]="--lidar --lidar_occ_recall 0.0 --lidar_dropout 0.20 --lidar_sector_prob 0.5"
  [blind]="--lidar --lidar_occ_recall 0.0 --lidar_dropout 1.0"
)
printf "policy\tterrain\tcfg\tsurv\tvx\tz\tscan_vis\tscan_info\tcurve\n" > "$OUT"
POLS=( "v6a_7.4M:models/go2_v6a_logs/checkpoints/000007372800"
       "v5a_final:models/go2_v5a_logs/checkpoints/000020070400" )
for spec in "${POLS[@]}"; do
  p=${spec%%:*}; ck=$(readlink -f "${spec#*:}")
  for terr in mix stairs slope rough boxes; do
    scene=models/go2/scene_mjx_terrain_${terr}_only.xml
    [ "$terr" = mix ] && scene=models/go2/scene_mjx_terrain_mix.xml
    for c in perfect nominal harsh blind; do
      out=$($PY sim/probe_go2_warp_eval.py --policy "$ck" --envs "$ENVS" --steps 300 \
            --spawn random --cmd 0.5 0 0 --foot-contact hard --terrain_xml "$scene" \
            $SCAN ${CFG[$c]} 2>&1 | grep -vE 'Module |load on device')
      surv=$(echo "$out" | sed -n 's/.*存活率.*= *\([0-9.]*\)%.*/\1/p')
      vx=$(echo "$out" | sed -n 's/.*机体系 vx = \([+-][0-9.]*\).*/\1/p')
      z=$(echo "$out" | sed -n 's/.*躯干 z = \([0-9.]*\).*/\1/p')
      sv=$(echo "$out" | sed -n 's/.*物理可见率 = \([0-9.]*\).*/\1/p')
      si=$(echo "$out" | sed -n 's/.*信息可用率 = \([0-9.]*\).*/\1/p')
      curve=$(echo "$out" | sed -n 's/.*存活曲线: //p')
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$p" "$terr" "$c" \
        "${surv:-NA}" "${vx:-NA}" "${z:-NA}" "${sv:-NA}" "${si:-NA}" "$curve" >> "$OUT"
      echo "[$(date +%H:%M:%S)] $p $terr $c surv=${surv:-NA}% vx=${vx:-NA} si=${si:-NA}"
      echo "    curve: ${curve:-NA}"
    done
  done
done
echo "DONE -> $OUT"
