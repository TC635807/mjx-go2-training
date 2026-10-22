#!/usr/bin/env python
"""遮挡矩阵 TSV -> 论文/报告用柱状图 (地形 x 遮挡档 x 策略)。

  python sim/plot_occlusion_matrix.py logs/go2_v6a_occlusion.tsv docs/img/go2-v6-occlusion.png

左图 = 基线策略 (v5a), 右图 = 遮挡感知策略 (v6a): 每地形 4 根柱 (perfect/nominal/
harsh/blind 存活率 %)。图里同时标出 4 档数字, 用来直观看出
"完美扫描 -> 真感知 -> 最坏遮挡 -> 全盲" 的退化曲线。
"""
import collections
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TERRAINS = ("mix", "stairs", "slope", "rough", "boxes")
CFGS = ("perfect", "nominal", "harsh", "blind")
CFG_COLORS = ("#1f77b4", "#2ca02c", "#ff7f0e", "#d62728")
CFG_LABELS = ("perfect scan", "lidar nominal (v6)", "lidar harsh", "blind (all fill)")


def load(path):
  acc = collections.defaultdict(list)
  with open(path, encoding="utf-8") as f:
    f.readline()
    for line in f:
      parts = line.rstrip("\n").split("\t")
      if len(parts) < 8 or parts[0] == "policy":
        continue
      try:
        surv = float(parts[3])
      except ValueError:
        continue
      acc[(parts[0], parts[1], parts[2])].append(surv)
  return {k: sum(v) / len(v) for k, v in acc.items() if v}


def main():
  path = sys.argv[1] if len(sys.argv) > 1 else "logs/go2_v6a_occlusion.tsv"
  out = sys.argv[2] if len(sys.argv) > 2 else "docs/img/go2-v6-occlusion.png"
  t = load(path)
  pols = sorted({k[0] for k in t})
  order = [p for p in ("v5a_best", "v6a_best") if p in pols] + \
          [p for p in pols if p not in ("v5a_best", "v6a_best")]
  n = len(order)
  fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 3.6), sharey=True)
  if n == 1:
    axes = [axes]
  x = np.arange(len(TERRAINS))
  w = 0.2
  for ax, pol in zip(axes, order):
    for i, cfg in enumerate(CFGS):
      vals = [t.get((pol, ter, cfg), np.nan) for ter in TERRAINS]
      bars = ax.bar(x + (i - 1.5) * w, vals, w, label=CFG_LABELS[i],
                    color=CFG_COLORS[i], edgecolor="white", linewidth=0.5)
      for b, v in zip(bars, vals):
        if np.isfinite(v):
          ax.text(b.get_x() + b.get_width() / 2, v + 1.0, "%.0f" % v,
                  ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(TERRAINS)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(pol.replace("_", " "), fontsize=9)
    ax.set_ylabel("survival % (warp physics, 384 env x 6 s)")
  axes[-1].legend(fontsize=6.5, loc="lower left", framealpha=0.9)
  fig.suptitle("Go2 terrain survival under lidar occlusion / blind scan", fontsize=10)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  os.makedirs(os.path.dirname(out), exist_ok=True)
  fig.savefig(out, dpi=140)
  print("[plot] 写出", out)


if __name__ == "__main__":
  main()
