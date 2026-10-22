#!/usr/bin/env bash
# 只读进度监视 (不占 GPU): 每 5 分钟一行。v2b = 硬接触, 从 2.75M 续到 40M。
cd "$(dirname "$0")/.."
for i in $(seq 1 2000); do
  sleep 300
  ck=$(ls -d models/go2_v2b_logs/checkpoints/*/ 2>/dev/null | grep -E '/[0-9]{12}/' | sort | tail -1)
  line=$(grep -E 'eval_reward' models/train_go2_v2b.log | tail -1)
  ovf=$(grep -c 'broadphase overflow' models/train_go2_v2b.log)
  echo "[$(date +%H:%M:%S)] ovf=${ovf} ${ck:-none} | ${line}"
done
