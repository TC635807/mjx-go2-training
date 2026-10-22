#!/usr/bin/env bash
# R1 前置验证: (1) 真竖面 box geom 地形的物理吞吐 (perf) 与 (2) v6a 在真竖面 vs
# hfield 斜边 boxes 上的存活率对比 (同进程)。
# 用法: bash sim/probe_boxgeom_terrain.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=logs/go2_boxgeom_probe.tsv
echo "===== perf: box geom 地形 128 env x 300 steps (v6a, nominal 遮挡) ====="
/usr/bin/time -f "wall=%e s maxrss=%M KB" \
  $PY -u sim/probe_go2_warp_eval.py --policy models/go2_v6a_logs/checkpoints/000007372800 \
  --envs 128 --steps 300 --spawn random --spawn_ring --cmd 0.5 0 0 --foot-contact hard \
  --terrain_xml models/go2/scene_mjx_terrain_bg200.xml \
  --boxgrid_json models/go2/assets/bg200.json --contact_mode clearance \
  --scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5 \
  --lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 \
  --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25 2>&1 \
  | grep -E 'v26 R1|存活率|存活曲线|wall=|机体系 vx|躯干 z|overflow' | tee logs/go2_boxgeom_perf.txt
echo
echo "===== 同进程对比: bg200(竖面) vs boxes_only(hfield 斜边) ====="
$PY -u sim/probe_sweep_params.py \
  --ckpts "v6a=models/go2_v6a_logs/checkpoints/000007372800" \
  --terrains "bg200=models/go2/scene_mjx_terrain_bg200.xml,boxes_hf=models/go2/scene_mjx_terrain_boxes_only.xml" \
  --envs 384 --steps 300 --spawn random --cmd 0.5 0 0 --foot-contact hard \
  --healthy_z_ref feet --lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 \
  --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25 \
  --boxgrid_json models/go2/assets/bg200.json --contact_mode clearance --spawn_ring \
  --out "$OUT" 2>&1 | grep -E '^\\[sweep\\]|surv=|DONE|Traceback|Error'
echo "DONE -> $OUT"
