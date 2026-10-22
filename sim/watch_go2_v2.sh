#!/usr/bin/env bash
# 只读进度监视: 每 5 分钟一行。v2 = 硬接触 40M。
cd "$(dirname "$0")/.."
for i in $(seq 1 1000); do
  sleep 300
  ck=$(ls -d models/go2_v2_hard_logs/checkpoints/*/ 2>/dev/null | grep -E '/[0-9]{12}/' | sort | tail -1)
  line=$(grep -E 'eval_reward' models/train_go2_v2.log | tail -1)
  ovf=$(grep -c 'broadphase overflow' models/train_go2_v2.log)
  echo "[$(date +%H:%M:%S)] ovf=${ovf} ckpt=${ck:-none} | ${line}"
done
