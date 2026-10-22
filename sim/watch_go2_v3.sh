#!/usr/bin/env bash
# v3 训练进度监视 (只读, 不占 GPU): 每 5 分钟一行。
# v3 = v2 + obs 0:3 的"真实速度估计误差" (bias/慢漂/延迟), 见 docs/go2-v3-obs-vel-error.md
cd "$(dirname "$0")/.."
for i in $(seq 1 2000); do
  sleep 300
  ck=$(ls -d models/go2_v3_logs/checkpoints/*/ 2>/dev/null | grep -E '/[0-9]{12}/' | sort | tail -1)
  line=$(grep -E 'eval_reward' logs/go2_v3_train.log | tail -1)
  ovf=$(grep -c 'broadphase overflow' logs/go2_v3_train.log)
  nan=$(grep -c 'NaN' logs/go2_v3_train.log)
  echo "[$(date +%H:%M:%S)] ovf=${ovf} nan=${nan} ${ck:-none} | ${line}"
done
