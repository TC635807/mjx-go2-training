#!/usr/bin/env bash
# 碎石/瓦砾地形 (真竖面小 box 密集场) 评测 —— 目标要求里的"凹凸不平(含碎石)"口径。
# 地形: sim/gen_terrain_boxgeom.py --grid 0.18 --height 0.07 --keep 0.55 --ext 5.0
#       --span 3.2 --seed 7 --out_prefix rubble
# 出生: 外圈平地环形出生 (hfield 半径 5.0m; box 场半径 3.2m; 自动环 3.55-4.5m 落在平地上)。
# 用法:
#   POLS="v20=models/go2_v20_wide_logs/checkpoints/000011468800" bash sim/eval_rubble.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=${OUT:-logs/go2_rubble_eval.tsv}
ENVS=${ENVS:-512}
POLS=${POLS:-"v20=models/go2_v20_wide_logs/checkpoints/000011468800"}
$PY -u sim/probe_sweep_params.py \
  --ckpts "$POLS" \
  --terrains "rubble=models/go2/scene_mjx_terrain_rubble.xml" \
  --boxgrid_json models/go2/assets/rubble.json \
  --envs "$ENVS" --steps 300 --spawn random --spawn_ring --cmd 0.5 0 0 \
  --foot-contact hard --healthy_z_ref feet --contact_mode clearance \
  --lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 \
  --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25 \
  --out "$OUT" 2>&1 | grep -E '^\[sweep\]|surv=|DONE|Traceback|Error'
echo "结果: $OUT"
