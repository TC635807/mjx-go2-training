#!/usr/bin/env bash
# 导出交付策略: orbax ckpt -> pkl (训练仓) -> npz (部署仓), 并打印 sha256。
# 用法: bash sim/export_deploy_policy.sh <ckpt_dir> <tag>   # tag 例如 v8
set -uo pipefail
cd "$(dirname "$0")/.."
HERE=$(pwd)
ROOT=/home/tc63/mujoco/mjx-go2-training
DEPLOY=/home/tc63/mujoco/mjx-go2-mujoco2ros2
PY=/home/tc63/mujoco/.venv/bin/python
CK=$(readlink -f "$1" 2>/dev/null || true)
TAG=${2:-v8}
if [ -z "$CK" ] || [ ! -d "$CK" ]; then echo "!! 检查点不存在: $1"; exit 2; fi
PKL="$ROOT/policies/go2_walk_${TAG}.pkl"
NPZ="$DEPLOY/ros2_controller/policies/go2_walk_${TAG}.npz"
echo "[export] ckpt = $CK"
JAX_PLATFORMS=cpu "$PY" "$ROOT/sim/ckpt_to_pkl.py" --ckpt "$CK" --out "$PKL"
JAX_PLATFORMS=cpu "$PY" "$DEPLOY/scripts/export_policy.py" --pkl "$PKL" \
  --out "$NPZ" --name "go2_walk_$TAG"
echo "[export] pkl = $PKL"
echo "[export] npz = $NPZ"
sha256sum "$PKL" "$NPZ"
ls -l "$PKL" "$NPZ"
