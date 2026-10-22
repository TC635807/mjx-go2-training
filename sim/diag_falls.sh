#!/usr/bin/env bash
# v7 诊断: 某策略在多地形上的摔倒位置/地形原因 (warp 物理, ENVS x STEPS)
# 用法: [ENVS=384 STEPS=300] bash sim/diag_falls.sh [ckpt] [terrain...]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
CK=$1
if [ ! -d "$CK" ]; then CK=models/go2_v6a_logs/checkpoints/000007372800; fi
shift || true
ENVS=${ENVS:-384}
STEPS=${STEPS:-300}
SCAN="--scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 --scan_y -0.5 0.5"
TERRAINS="$*"
if [ -z "$TERRAINS" ]; then TERRAINS="boxes stairs mix rough"; fi
for t in $TERRAINS; do
  scene=models/go2/scene_mjx_terrain_${t}_only.xml
  [ "$t" = mix ] && scene=models/go2/scene_mjx_terrain_mix.xml
  # v7 等自定义地形没有 _only 后缀: 退回到 scene_mjx_terrain_<t>.xml
  if [ ! -f "$scene" ]; then scene=models/go2/scene_mjx_terrain_${t}.xml; fi
  echo "===== $t (envs=$ENVS steps=$STEPS) ====="
  $PY -u sim/probe_go2_warp_eval.py --policy "$CK" --envs "$ENVS" --steps "$STEPS" \
    --spawn random --cmd 0.5 0 0 --foot-contact hard --terrain_xml "$scene" \
    $SCAN --diag 2>&1 | grep -v 'Module .* load on device'
done
