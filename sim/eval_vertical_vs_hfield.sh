#!/usr/bin/env bash
# 双套口径评测: (A) 真竖面地形 (box geom: bg200 boxes / stg1 stairs / hyb1 混合)
#             (B) 原 hfield 地形 (mix/stairs/slope/rough/boxes)
# 用法:
#   POLS="v6a=...,v12_best=..." bash sim/eval_vertical_vs_hfield.sh
# 注意: 两套地形必须分两个进程跑 (boxgrid_json 对不同场景高度表不同)。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUTDIR=${OUTDIR:-logs/go2_vert_eval}
mkdir -p "$OUTDIR"
POLS=${POLS:-"v6a=models/go2_v6a_logs/checkpoints/000007372800"}
ENVS=${ENVS:-384}
HIST=${HIST:-0}
HIST_ARG=""
[ "$HIST" != "0" ] && HIST_ARG="--obs_history_frames $HIST"
LIDAR="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "===== A) 真竖面地形 (box geom, clearance 接触, 环形出生) ====="
$PY -u sim/probe_sweep_params.py \
  --ckpts "$POLS" \
  --terrains "bg200=models/go2/scene_mjx_terrain_bg200.xml,stg1=models/go2/scene_mjx_terrain_stg1.xml,hyb1=models/go2/scene_mjx_terrain_hyb1.xml,hyb2=models/go2/scene_mjx_terrain_hyb2.xml,hyb3=models/go2/scene_mjx_terrain_hyb3.xml" \
  --boxgrid_map "bg200=models/go2/assets/bg200.json,stg1=models/go2/assets/stg1.json,hyb1=models/go2/assets/hyb1.json,hyb2=models/go2/assets/hyb2.json,hyb3=models/go2/assets/hyb3.json" \
  --envs "$ENVS" --steps 300 --spawn random --spawn_ring --cmd 0.5 0 0 \
  --foot-contact hard --healthy_z_ref feet --contact_mode clearance $LIDAR $HIST_ARG \
  --out "$OUTDIR/vertical.tsv" 2>&1 | grep -E '^\\[sweep\\]|surv=|DONE|Traceback|Error'

say "===== B) 原 hfield 地形 (与历史数字可比) ====="
$PY -u sim/probe_sweep_params.py \
  --ckpts "$POLS" \
  --terrains "mix=models/go2/scene_mjx_terrain_mix.xml,stairs=models/go2/scene_mjx_terrain_stairs_only.xml,slope=models/go2/scene_mjx_terrain_slope_only.xml,rough=models/go2/scene_mjx_terrain_rough_only.xml,boxes=models/go2/scene_mjx_terrain_boxes_only.xml" \
  --envs "$ENVS" --steps 300 --spawn random --cmd 0.5 0 0 \
  --foot-contact hard --healthy_z_ref feet $LIDAR $HIST_ARG \
  --out "$OUTDIR/hfield.tsv" 2>&1 | grep -E '^\\[sweep\\]|surv=|DONE|Traceback|Error'

say "结果: $OUTDIR/vertical.tsv (真竖面) 与 $OUTDIR/hfield.tsv (hfield)"
