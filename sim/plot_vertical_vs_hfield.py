#!/usr/bin/env python
"""双口径对比图: 真竖面地形 (box geom) vs 原 hfield 地形, 四个策略的 384env 6s 存活率。

  python sim/plot_vertical_vs_hfield.py logs/go2_v13_eval docs/img/go2-v13-vertical-vs-hfield.png
"""
import collections
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VERT = ("bg200", "stg1", "hyb1", "hyb2", "hyb3")
HF = ("mix", "stairs", "slope", "rough", "boxes")
ORDER = ("v6a", "v13_20.32", "v17_8.60", "v20_11.47", "v20_20.07", "v20_8.60")
LABEL = {"v6a": "v6a (hfield-trained)", "r1_12.12": "r1 (vertical boxes 12M)",
         "v13_15.24": "v13 @15.2M", "v13_20.32": "v13 128x128",
         "v17_8.60": "v17 512-256 @8.6M",
         "v20_11.47": "v20 1024-512 @11.5M (delivery)",
         "v20_8.60": "v20 @8.6M", "v20_20.07": "v20 @20.1M"}
COLORS = ("#8c8c8c", "#5b9bd5", "#2e7d32", "#e08a2e", "#7b1fa2", "#26a69a")


def load(path):
    t = {}
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            t[(p[0], p[1])] = float(p[2])      # 列序: policy terrain surv vx z curve
    return t


def main():
    indir = sys.argv[1] if len(sys.argv) > 1 else "logs/go2_v13_eval"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/img/go2-v13-vertical-vs-hfield.png"
    tv = load(os.path.join(indir, "vertical.tsv"))
    th = load(os.path.join(indir, "hfield.tsv"))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8))
    for ax, keys, title in ((axes[0], VERT, "A) True vertical faces (box geoms)"),
                            (axes[1], HF, "B) hfield ramps (historical comparability)")):
        x = np.arange(len(keys))
        present = tuple(p for p in ORDER
                        if any((p, k) in tv or (p, k) in th for k in VERT + HF))
        order = present + tuple(sorted({p for p, _ in list(tv) + list(th)
                                        if p not in present}))
        n = len(order)
        w = 0.8 / n
        for i, pol in enumerate(order):
            vals = [tv.get((pol, k), th.get((pol, k), np.nan)) if keys is VERT
                    else th.get((pol, k), np.nan) for k in keys]
            bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w,
                          label=LABEL.get(pol, pol), color=COLORS[i],
                          edgecolor="white", linewidth=0.4)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1.2,
                            "%.0f" % v, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(keys)
        ax.set_ylim(0, 100)
        ax.set_ylabel("survival % (384 env x 6 s)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle("Go2 terrain survival: vertical-face geometry vs hfield ramps "
                 "(warp physics, 384 env x 6s, random spawn)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    print("[plot] 写出", out)


if __name__ == "__main__":
    main()
