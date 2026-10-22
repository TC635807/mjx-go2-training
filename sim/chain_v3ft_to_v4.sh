#!/usr/bin/env bash
# v3 微调 (硬接触, 从 v2 37.26M 续 10M) -> v3 CPU 评测 -> v4 (混合地形) 自动接力。
# 触发条件: 进程退出 + 日志含"训练完成" + policies/go2_walk_v3.pkl 存在;
# 失败则只报错, 不启动 v4 (避免坏基线污染 v4 归因)。
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/chain_v3ft_v4.log
mkdir -p logs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "接力已武装: 等 v3 微调退出 (pattern go2_v3ft_logs)"
while pgrep -f 'train.train_go2.*go2_v3ft_logs' >/dev/null 2>&1; do sleep 60; done

if grep -q '训练完成' logs/go2_v3ft_train.log && [ -f policies/go2_walk_v3.pkl ]; then
  say "v3 微调正常完成: $(grep -E '^[0-9]+: eval_reward' logs/go2_v3ft_train.log | tail -1)"
else
  say "!! v3 微调未正常完成, 不启动 v4"
  tail -20 logs/go2_v3ft_train.log | tee -a "$LOG"
  exit 1
fi

say "v3 CPU 评测开始 (flat 训练物理 / flat 部署模型 / 旧地形)"
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

say "启动 v4 (混合地形 + 硬接触): logs/go2_v4_train.log"
setsid nohup bash sim/start_go2_v4_train.sh > logs/go2_v4_train.log 2>&1 &
say "v4 已 detach 启动 pid=$!"
