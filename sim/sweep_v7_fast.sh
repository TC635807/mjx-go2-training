#!/usr/bin/env bash
# v7 检查点扫描 (快版): 单进程多策略 x 多地形, 每个地形只编译一次。
# 评测口径: warp 训练物理, 128 env x 300 步 (6s), 随机出生, nominal 遮挡。
# 用法: bash sim/sweep_v7_fast.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=${OUT:-logs/go2_v7_sweep_fast.tsv}
ENVS=${ENVS:-128}
STEPS=${STEPS:-300}

# 检查点列表: v6a 参照 + v7 全部检查点 (按目录名排序)
CKPTS="v6a@7.37M=models/go2_v6a_logs/checkpoints/000007372800"
for d in models/go2_v7_logs/checkpoints/*/; do
  [ -d "$d" ] || continue
  st=$(basename "$d")
  [ "$st" = "best" ] && continue
  CKPTS="$CKPTS,v7@$st=$d"
done
CKPTS="$CKPTS,v7_best=models/go2_v7_logs/checkpoints/best"
echo "ckpts: $CKPTS"

"$PY" sim/probe_sweep_params.py \
  --ckpts "$CKPTS" \
  --terrains "v7=models/go2/scene_mjx_terrain_v7.xml,boxes=models/go2/scene_mjx_terrain_boxes_only.xml,stairs=models/go2/scene_mjx_terrain_stairs_only.xml" \
  --envs "$ENVS" --steps "$STEPS" --spawn random --cmd 0.5 0 0 \
  --foot-contact hard --healthy_z_ref feet \
  --lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 \
  --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25 \
  --out "$OUT" 2>&1 | grep -vE 'Module |load on device|height field collision|decrease the number'
echo "DONE -> $OUT"
