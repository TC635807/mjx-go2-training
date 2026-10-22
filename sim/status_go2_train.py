#!/usr/bin/env python3
"""Go2 训练状态一览 (v3 / v4 / 接力链 / GPU), 只读。

  /home/tc63/mujoco/.venv/bin/python sim/status_go2_train.py
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TOTAL = 40_000_000


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              cwd=ROOT).stdout.strip()
    except Exception as exc:
        return "ERR " + str(exc)


def find_pid(pattern: str) -> str:
    out = sh("pgrep -af '%s'" % pattern)
    for line in out.splitlines():
        pid = line.split()[0]
        # 排除 pgrep 自身/本脚本
        if "status_go2_train" in line or "pgrep -af" in line:
            continue
        return pid
    return ""


def show(tag: str, log: str, logdir: str, total: int = TOTAL) -> None:
    print("==== %s ====" % tag)
    pid = find_pid("train.train_go2.*--logdir %s" % logdir)
    print("状态:", ("RUNNING (pid %s)" % pid) if pid else "未运行")
    if pid:
        print("已跑:", sh("ps -o etime= -p %s" % pid).strip())
    last = ""
    if os.path.exists(os.path.join(ROOT, log)):
        with open(os.path.join(ROOT, log), encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if re.match(r"^\d+.*: eval_reward", line):
                    last = line.strip()
    print("最近 eval:", last or "n/a")
    ovf = sh("grep -c 'broadphase overflow' %s 2>/dev/null || true" % log)
    nan = sh("grep -c 'NaN' %s 2>/dev/null || true" % log)
    print("溢出/NaN: overflow=%s nan=%s" % (ovf or 0, nan or 0))
    ck = sorted(glob.glob(os.path.join(ROOT, logdir, "checkpoints", "[0-9]" * 12)))
    if len(ck) >= 2 and pid:
        n2 = int(os.path.basename(ck[-1]))
        n1 = int(os.path.basename(ck[-2]))
        t2 = os.path.getmtime(ck[-1])
        t1 = os.path.getmtime(ck[-2])
        rate = (n2 - n1) / max(t2 - t1, 1.0)
        rem = max(total - n2, 0)
        eta = time.time() + rem / rate if rate > 0 else 0
        print("进度: %.2fM / %.0fM, 速率 %.0f 步/s, ETA %s (剩 %.0f 分钟)"
              % (n2 / 1e6, total / 1e6, rate,
                 time.strftime("%H:%M", time.localtime(eta)) if eta else "n/a",
                 rem / rate / 60 if rate > 0 else -1))
    print()


def main() -> int:
    show("v7 (boxes/stairs 专项; 从 v6a@7.37M 续训 8M; v24 终止判据)",
         "logs/go2_v7_train.log", "models/go2_v7_logs", total=8_000_000)
    show("v6a (雷达遮挡感知; 从 v5a-best 续训 12M)",
         "logs/go2_v6a_lidar.log", "models/go2_v6a_logs", total=12_000_000)
    show("v5a (187 点扫描 + mix, 20M; v6 的起点)",
         "logs/go2_v5_curriculum.log", "models/go2_v5a_logs", total=20_000_000)
    show("v4 (混合地形 n=128 + v2 配方; 当前部署默认)",
         "logs/go2_v4_train.log", "models/go2_v4_logs", total=40_000_000)
    print()
    print("==== GPU ====")
    print(sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
