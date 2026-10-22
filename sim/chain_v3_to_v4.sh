#!/usr/bin/env bash
# v3 -> v4 自动接力:
#   1) 等 v3 (policies/go2_walk_v3.pkl + 日志"训练完成") 正常结束;
#   2) 用 CPU 跑 v3 的两组离线评测 (不占 GPU), 结果写 logs/go2_v3_eval.txt;
#   3) 在混合地形上启动 v4 (detach, 日志 logs/go2_v4_train.log)。
# v3 若异常退出 (OOM/NaN), **不**启动 v4, 直接报错退出 (避免用坏基线污染 v4 归因)。
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/chain_v3_v4.log
mkdir -p logs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "接力已武装: 等 v3 退出 (pgrep train_go2 go2_v3_logs)"
while pgrep -f 'train.train_go2.*go2_v3_logs' >/dev/null 2>&1; do sleep 60; done

if grep -q '训练完成' logs/go2_v3_train.log && [ -f policies/go2_walk_v3.pkl ]; then
  say "v3 正常完成: $(grep -E '^[0-9]+: eval_reward' logs/go2_v3_train.log | tail -1)"
else
  say "!! v3 未正常完成 (无 训练完成/pkl), 不启动 v4"
  tail -25 logs/go2_v3_train.log | tee -a "$LOG"
  exit 1
fi

say "v3 CPU 评测开始 (flat 训练物理 + 部署模型 + terrain 训练物理)"
{
  echo "===== v3 flat (训练 XML, position actuator) ====="
  JAX_PLATFORMS=cpu timeout 900 /home/tc63/mujoco/.venv/bin/python sim/eval_go2.py \
      --policy policies/go2_walk_v3.pkl --terrain flat --cmd 0.5 0 0 \
      --seeds 16 --duration 15 --vel-frame body 2>&1 | tail -30
  echo; echo "===== v3 flat (部署 XML, motor+PD) ====="
  JAX_PLATFORMS=cpu timeout 900 /home/tc63/mujoco/.venv/bin/python sim/eval_go2.py \
      --policy policies/go2_walk_v3.pkl \
      --xml /home/tc63/mujoco/mjx-go2-mujoco2ros2/sim_env/models/go2/scene_flat.xml \
      --cmd 0.5 0 0 --seeds 8 --duration 10 --vel-frame body 2>&1 | tail -30
  echo; echo "===== v3 terrain (旧地形, 训练物理) ====="
  JAX_PLATFORMS=cpu timeout 900 /home/tc63/mujoco/.venv/bin/python sim/eval_go2.py \
      --policy policies/go2_walk_v3.pkl --terrain terrain --cmd 0.5 0 0 \
      --seeds 16 --duration 15 --vel-frame body 2>&1 | tail -30
} > logs/go2_v3_eval.txt 2>&1
say "v3 评测完成 -> logs/go2_v3_eval.txt"
grep -E '存活率|实际 vx|跟踪误差|躯干相对地面高度' logs/go2_v3_eval.txt | tee -a "$LOG"

say "启动 v4 (混合地形): logs/go2_v4_train.log"
setsid nohup bash sim/start_go2_v4_train.sh > logs/go2_v4_train.log 2>&1 &
say "v4 已 detach 启动 pid=$! ; 等待首个 eval (约 10 分钟)"
