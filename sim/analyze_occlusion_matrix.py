#!/usr/bin/env python
"""遮挡矩阵 TSV -> Markdown 分析表 (策略 x 地形 x 遮挡档)。

输入: sim/eval_occlusion_matrix.sh 产出的 TSV
  policy terrain cfg surv vx z scan_vis scan_info
输出: 每个策略一张 "地形 x 遮挡档" 存活率表 + 派生量:
  * clean->nominal 的退化 (遮挡代价)
  * clean->harsh / clean->blind 的退化
  * harsh - blind = "退化状态下感知仍然有用" 的收益 (调研 §5.5 判据)

用法:
  python sim/analyze_occlusion_matrix.py logs/go2_v6a_occlusion.tsv
"""
import collections
import os
import sys

TERRAINS = ("mix", "stairs", "slope", "rough", "boxes")
CFGS = ("perfect", "nominal", "harsh", "blind")


def load(path):
  acc = collections.defaultdict(list)
  with open(path, encoding="utf-8") as f:
    header = f.readline()
    for line in f:
      parts = line.rstrip("\n").split("\t")
      if len(parts) < 8 or parts[0] == "policy":
        continue
      pol, terrain, cfg = parts[0], parts[1], parts[2]
      try:
        surv = float(parts[3]) if parts[3] not in ("NA", "") else None
        vx = float(parts[4]) if parts[4] not in ("NA", "") else None
      except ValueError:
        continue
      if surv is None:
        continue
      acc[(pol, terrain, cfg)].append((surv, vx))
  table = {}
  for k, vals in acc.items():
    sv = [v[0] for v in vals]
    vx = [v[1] for v in vals if v[1] is not None]
    table[k] = (sum(sv) / len(sv), (sum(vx) / len(vx)) if vx else None)
  return table


def main():
  path = sys.argv[1] if len(sys.argv) > 1 else "logs/go2_occlusion_matrix.tsv"
  if not os.path.isfile(path):
    raise SystemExit("找不到 %s" % path)
  t = load(path)
  pols = sorted({k[0] for k in t})
  out = ["# 遮挡矩阵分析: %s\n" % os.path.basename(path)]

  for pol in pols:
    out.append("## %s\n" % pol)
    out.append("存活率 % (warp 训练物理, 384 env x 6s, 随机出生; 单点噪声 ±2~3pp):\n")
    out.append("| 地形 | " + " | ".join(CFGS) + " |")
    out.append("|---|" + "---|" * len(CFGS))
    for ter in TERRAINS:
      cells = []
      for cfg in CFGS:
        v = t.get((pol, ter, cfg))
        cells.append("—" if v is None else "%.1f" % v[0])
      out.append("| %s | %s |" % (ter, " | ".join(cells)))
    # 派生量
    out.append("")
    out.append("| 地形 | 遮挡代价 clean-nominal | clean-harsh | clean-blind | harsh-blind (退化下感知收益) |")
    out.append("|---|---|---|---|---|")
    for ter in TERRAINS:
      g = lambda c: (t[(pol, ter, c)][0] if (pol, ter, c) in t else None)
      p, n, h, b = g("perfect"), g("nominal"), g("harsh"), g("blind")
      f = lambda a, c: ("—" if (a is None or c is None) else "%+.1f" % (c - a))
      # 注意: "遮挡代价" = nominal - perfect (负 = 退化)
      out.append("| %s | %s | %s | %s | %s |" %
                 (ter, f(p, n), f(p, h), f(p, b),
                  f(h, b) if (h is not None and b is not None) else "—"))
    out.append("")

  txt = "\n".join(out)
  print(txt)
  outp = os.path.splitext(path)[0] + "_analysis.md"
  with open(outp, "w", encoding="utf-8") as f:
    f.write(txt)
  print("\n[analysis] 写出 %s" % outp)


if __name__ == "__main__":
  main()
