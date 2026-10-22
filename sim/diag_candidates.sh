#!/usr/bin/env bash
# 高统计功效的候选检查点选择题: v5a vs v6a 各中间检查点, 在**部署口径遮挡 (nominal)** 下
# 跑 mix/stairs/boxes x 384 env (单次 128 env 的抽样噪声 ±4~6pp, 384 env 降到 ±2~3pp)。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=logs/go2_v6_candidates.tsv
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
LIDAR="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
ENVS=${ENVS:-384}
printf "policy\tterrain\tsurv\tvx\tz\tscan_vis\tscan_info\tcurve\n" > "$OUT"

POLS=(
  "v5a_final:models/go2_v5a_logs/checkpoints/000020070400"
  "v6a_4.9M:models/go2_v6a_logs/checkpoints/000004915200"
  "v6a_7.4M:models/go2_v6a_logs/checkpoints/000007372800"
  "v6a_9.8M:models/go2_v6a_logs/checkpoints/000009830400"
  "v6a_best:models/go2_v6a_logs/checkpoints/best"
)
for spec in "${POLS[@]}"; do
  p=${spec%%:*}; ck=$(readlink -f "${spec#*:}")
  for spec_t in "mix:models/go2/scene_mjx_terrain_mix.xml" \
                "stairs:models/go2/scene_mjx_terrain_stairs_only.xml" \
                "boxes:models/go2/scene_mjx_terrain_boxes_only.xml"; do
    t=${spec_t%%:*}; scene=${spec_t#*:}
    out=$($PY sim/probe_go2_warp_eval.py --policy "$ck" --envs "$ENVS" --steps 300 \
          --spawn random --cmd 0.5 0 0 --foot-contact hard --terrain_xml "$scene" \
          $SCAN $LIDAR 2>&1 | grep -vE 'Module |load on device')
    surv=$(echo "$out" | sed -n 's/.*存活率.*= *\([0-9.]*\)%.*/\1/p')
    vx=$(echo "$out" | sed -n 's/.*机体系 vx = \([+-][0-9.]*\).*/\1/p')
    z=$(echo "$out" | sed -n 's/.*躯干 z = \([0-9.]*\).*/\1/p')
    sv=$(echo "$out" | sed -n 's/.*物理可见率 = \([0-9.]*\).*/\1/p')
    si=$(echo "$out" | sed -n 's/.*信息可用率 = \([0-9.]*\).*/\1/p')
    curve=$(echo "$out" | sed -n 's/.*存活曲线: //p')
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$p" "$t" "${surv:-NA}" "${vx:-NA}" "${z:-NA}" "${sv:-NA}" "${si:-NA}" "$curve" >> "$OUT"
    echo "[$(date +%H:%M:%S)] $p $t surv=${surv:-NA}% vx=${vx:-NA} sv=${sv:-NA} si=${si:-NA}"
    echo "    curve: ${curve:-NA}"
  done
done
echo "DONE -> $OUT"
