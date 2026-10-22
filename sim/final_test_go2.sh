#!/usr/bin/env bash
# Go2 交付模型的**完整自测**。全部 CPU (JAX_PLATFORMS=cpu), 不占 GPU。
#
# 用法:
#   bash sim/final_test_go2.sh <策略: pkl 或 orbax 检查点目录> [输出前缀]
#
# 六组测试:
#   1 平地, cmd 0.5             —— 基本行走能力 + 速度跟踪
#   2 地形(原点出生), cmd 0.5    —— 一定复杂地形上的通过性 (主指标)
#   3 地形(随机出生 ±5m)         —— 训练同分布, 排除出生点偏差
#   4 多指令扫描                 —— 前进/后退?/侧移/转向的速度跟踪泛化
#   5 部署模型 (motor + PD 20/1) —— 策略在**部署物理**上能不能跑
#   6 纯站立鲁棒性 (cmd 0)       —— 不动时是否稳
set -uo pipefail
cd "$(dirname "$0")/.."

POL="${1:?用法: bash sim/final_test_go2.sh <policy> [tag]}"
TAG="${2:-final}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/tc63/mujoco/.venv/bin/python
# 注意: 必须用**带地面的场景**, 不能用 models/go2/go2.xml (那只是机器人本体,
# 没有 floor) —— 直接喂 go2.xml 会让机器人自由落体, 测出来"占空比 0/前进 0"。
DEPLOY_XML=/home/tc63/mujoco/mjx-go2-mujoco2ros2/sim_env/models/go2/scene_flat.xml
OUT=logs/go2_eval

run () {  # run <标题> <参数...>
  local title="$1"; shift
  echo ""
  echo "################ $title"
  "$PY" sim/eval_go2.py --policy "$POL" --tag "${TAG}_${title}" "$@" 2>&1 \
    | sed -n '/\[1\]/,/\[6\]/p'
}

run flat_050        --terrain flat    --cmd 0.5 0.0 0.0 --seeds 16 --duration 15
run terr_050        --terrain terrain --cmd 0.5 0.0 0.0 --seeds 16 --duration 15
run terr_rand       --terrain terrain --cmd 0.5 0.0 0.0 --seeds 16 --duration 15 \
                    --spawn random --spawn-range 5
for c in "0.3 0.0 0.0" "0.8 0.0 0.0" "0.5 0.3 0.0" "0.5 0.0 0.8"; do
  run "cmd_${c// /_}" --terrain terrain --cmd $c --seeds 8 --duration 15
done
run stand_000       --terrain terrain --cmd 0.0 0.0 0.0 --seeds 8 --duration 15

echo ""
echo "################ 部署模型 (motor 力矩执行器 + 外部 PD kp20/kd1)"
"$PY" sim/eval_go2.py --policy "$POL" --tag "${TAG}_deploy" \
  --xml "$DEPLOY_XML" --dof-order actuator --kp 20 --kd 1 --home-pose unitree_rl_gym \
  --cmd 0.5 0.0 0.0 --seeds 16 --duration 15 2>&1 \
  | sed -n '/\[1\]/,/\[6\]/p'

echo ""
echo "全部结果已写入 $OUT/ ($TAG_*.json / $TAG_*.png)"
