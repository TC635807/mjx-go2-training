#!/usr/bin/env bash
# v4 (混合地形) vs v2 (旧地形) 的 CPU 评测套件 —— 训练物理 + 混合地形 + 部署模型.
# 产物: logs/go2_v4_eval.txt
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=logs/go2_v4_eval.txt
{
echo "########## v4: flat (训练 XML, position actuator) ##########"
JAX_PLATFORMS=cpu timeout 900 "$PY" sim/eval_go2.py --policy policies/go2_walk_v4.pkl \
    --terrain flat --cmd 0.5 0 0 --seeds 16 --duration 15 --vel-frame body 2>&1 | tail -28

echo; echo "########## v4: 混合地形 (n=128, 6s x 16) ##########"
JAX_PLATFORMS=cpu timeout 900 "$PY" sim/eval_go2.py --policy policies/go2_walk_v4.pkl \
    --xml models/go2/scene_mjx_terrain_mix.xml --cmd 0.5 0 0 --seeds 16 --duration 6 --vel-frame body 2>&1 | tail -28

echo; echo "########## v2: 混合地形 (同图, 6s x 16, 对照) ##########"
JAX_PLATFORMS=cpu timeout 900 "$PY" sim/eval_go2.py --policy policies/go2_walk_policy_hard.pkl \
    --xml models/go2/scene_mjx_terrain_mix.xml --cmd 0.5 0 0 --seeds 16 --duration 6 --vel-frame body 2>&1 | tail -28

echo; echo "########## v4: 旧地形 (6s x 16) ##########"
JAX_PLATFORMS=cpu timeout 900 "$PY" sim/eval_go2.py --policy policies/go2_walk_v4.pkl \
    --terrain terrain --cmd 0.5 0 0 --seeds 16 --duration 6 --vel-frame body 2>&1 | tail -28

echo; echo "########## v4: flat (部署 XML, motor+PD, 10s x 8) ##########"
JAX_PLATFORMS=cpu timeout 900 "$PY" sim/eval_go2.py --policy policies/go2_walk_v4.pkl \
    --xml /home/tc63/mujoco/mjx-go2-mujoco2ros2/sim_env/models/go2/scene_flat.xml \
    --cmd 0.5 0 0 --seeds 8 --duration 10 --vel-frame body 2>&1 | tail -28
} > "$OUT" 2>&1
echo "done -> $OUT"
grep -E '^#####|存活率|实际 vx|跟踪误差|躯干相对地面高度' "$OUT"
