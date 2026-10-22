#!/usr/bin/env python
"""把双口径评测 TSV (sim/eval_vertical_vs_hfield.sh 产物) 汇总成 markdown 表。

  python sim/summarize_dual_suite.py logs/go2_v16_eval
"""
import os
import sys

VERT = ("bg200", "stg1", "hyb1", "hyb2")
HF = ("mix", "stairs", "slope", "rough", "boxes")


def load(path):
    t = {}
    if not os.path.isfile(path):
        return t
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            try:
                t[(p[0], p[1])] = float(p[2])
            except ValueError:
                pass
    return t


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "logs/go2_v16_eval"
    tv = load(os.path.join(d, "vertical.tsv"))
    th = load(os.path.join(d, "hfield.tsv"))
    pols = sorted({k[0] for k in list(tv) + list(th)})
    print("\n### A) True vertical faces (box geoms), 384 env x 6 s, nominal lidar\n")
    print("| policy | " + " | ".join(VERT) + " | mean |")
    print("|---|" + "---|" * (len(VERT) + 1))
    for p in pols:
        vals = [tv.get((p, k)) for k in VERT]
        mean = "%.1f" % (sum(v for v in vals if v is not None)
                         / max(1, sum(1 for v in vals if v is not None)))
        print("| %s | %s | %s |" % (p, " | ".join(
            "%.1f" % v if v is not None else "—" for v in vals), mean))
    print("\n### B) hfield terrains (historical comparability)\n")
    print("| policy | " + " | ".join(HF) + " | mean |")
    print("|---|" + "---|" * (len(HF) + 1))
    for p in pols:
        vals = [th.get((p, k)) for k in HF]
        mean = "%.1f" % (sum(v for v in vals if v is not None)
                         / max(1, sum(1 for v in vals if v is not None)))
        print("| %s | %s | %s |" % (p, " | ".join(
            "%.1f" % v if v is not None else "—" for v in vals), mean))
    print()


if __name__ == "__main__":
    main()
