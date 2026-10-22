#!/usr/bin/env bash
# 主训练进度监视 (只读, 不占 GPU): 每 5 分钟打一行进度。
cd "$(dirname "$0")/.."
for i in $(seq 1 1000); do
  sleep 300
  ck=$(ls -d models/go2_v1_logs/checkpoints/*/ 2>/dev/null | grep -E '/[0-9]{12}/' | sort | tail -1)
  line=$(grep -E 'eval_reward' models/train_go2_v1.log | tail -1)
  echo "[$(date +%H:%M:%S)] ckpt=${ck:-none} | ${line}"
done
