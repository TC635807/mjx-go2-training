#!/usr/bin/env python
"""生成 **大混合地形** (5x5 patch, 场范围 ±7m): 一次训练里同时见到
真竖面 boxes / 真竖面 stairs / **碎石 pebble** (密集小 box) / hfield stairs/boxes/rough/slope。

动机: 分阶段地形特化会"学一个忘一个" (v16/v19/v22 都出现)。更大、patch 更多的地形让
单次训练覆盖所有地形族。

用法:
  python sim/gen_terrain_big.py --prefix big1 --seed 1 --n-patch 5
输出: assets/<prefix>.png / <prefix>.json / models/go2/scene_mjx_terrain_<prefix>.xml

注意:
  * hfield 半宽 ext = field_half + 2.5, 保证训练环形出生 (r 7.35~8.3m) 落在平地上;
  * boxgrid 高度表 = 每 0.4m 格最大高度 (env 的 terrain_height/终止参考用),
    pebble patch 的密排小 box 实际接触由 clearance 判据处理;
  * 训练需要 --random_init_range >= field_half + 1.5 (否则环形出生上界被截断)。
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_terrain_hybrid import t_rough, BOX_TMPL  # noqa: E402
from gen_terrain_go2_mix import t_stairs_up, t_boxes  # noqa: E402

SCENE_TMPL = '''<mujoco model="go2 big multi-patch terrain scene">
  <!-- 由 sim/gen_terrain_big.py 生成, 不要手改。
       {nb} 个 box geom (真竖面 boxes/stairs + 碎石 pebble) + hfield 平滑 patch,
       场半宽 {field_half:.1f}m, hfield 半宽 {ext:.1f}m, patch {patch:.2f}m。 -->
  <include file="go2_mjx_position.xml"/>
  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>
  <visual>
    <rgba force="1 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>
  <asset>
    <texture type="2d" name="mix_ground" builtin="checker" mark="edge"
      rgb1="0.24 0.30 0.24" rgb2="0.16 0.20 0.17" markrgb="0.5 0.6 0.4"
      width="300" height="300"/>
    <material name="mix_ground" texture="mix_ground" texuniform="true" texrepeat="8 8" reflectance=".2"/>
    <material name="box_mat" rgba="0.55 0.45 0.35 1"/>
    <material name="pebble_mat" rgba="0.45 0.45 0.50 1"/>
    <hfield name="terrain" file="{prefix}.png" size="{ext:.2f} {ext:.2f} {elev:.4f} 0.0960"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="terrain" material="mix_ground"
      pos="0 0 0" friction="1.0 0.02 0.01"/>
{boxes}
  </worldbody>
</mujoco>
'''

PEBBLE_TMPL = ('    <geom name="{name}" type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
               'size="{hx:.4f} {hy:.4f} {hz:.4f}" material="pebble_mat" '
               'contype="1" conaffinity="1" friction="0.95 0.02 0.01"/>')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="big1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-patch", type=int, default=5, help="每边 patch 数")
    ap.add_argument("--patch", type=float, default=2.8)
    ap.add_argument("--grid", type=float, default=0.4)
    ap.add_argument("--pad", type=float, default=2.5, help="hfield 超出场边界的平地带宽")
    ap.add_argument("--hf_n", type=int, default=192)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    gw = float(args.grid)
    npatch = max(1, int(round(args.patch / gw)))
    patch = npatch * gw
    n = args.n_patch
    field_half = patch * n / 2.0
    ext = field_half + args.pad
    x0 = -field_half
    y0 = -field_half

    # 5x5 patch 类型: 覆盖所有地形族, 每种多块 + 随机种子
    types = ["pebble", "boxes_vb", "stairs_va", "rough", "boxes_va",
             "slope", "boxes_vc", "pebble", "stairs_vb", "boxes_hf",
             "stairs_hf", "rough", "boxes_vb", "slope", "pebble",
             "boxes_va", "stairs_va", "rough", "boxes_vc", "slope",
             "stairs_vb", "boxes_hf", "boxes_vb", "stairs_hf", "flat"]
    layout = [types[pr * n:(pr + 1) * n] for pr in range(n)]
    box_hmax = {"boxes_va": 0.13, "boxes_vb": 0.07, "boxes_vc": 0.10}
    stair_h = {"stairs_va": 0.11, "stairs_vb": 0.06}

    ncell = npatch * n
    H = np.zeros((ncell, ncell), np.float64)
    boxes = []
    nb = 0
    for pr in range(n):
        for pc in range(n):
            kind = types[pr * n + pc]
            i0, i1 = pc * npatch, (pc + 1) * npatch
            j0, j1 = pr * npatch, (pr + 1) * npatch
            xl = x0 + pc * patch
            yl = y0 + pr * patch
            if kind in box_hmax:
                hm = box_hmax[kind]
                blk = rng.uniform(0.0, hm, (npatch, npatch))
                keep = rng.uniform(size=(npatch, npatch)) < 0.5
                H[i0:i1, j0:j1] = np.where(keep, blk, 0.0)
            elif kind in ("stairs_va", "stairs_vb"):
                h = stair_h[kind]
                prof = np.asarray([0.0, h, 2 * h, 2 * h, h, 0.0, 0.0])
                if kind == "stairs_va":
                    H[i0:i1, j0:j1] = prof[:, None]
                else:
                    H[i0:i1, j0:j1] = prof[None, :]
            elif kind == "pebble":
                # 0.14m 子网格密排小 box, 高 1.5~7cm, 保留率 0.55
                sub = 0.14
                ns = int(round(patch / sub))
                sg = np.zeros((ns, ns))
                hh = rng.uniform(0.015, 0.07, (ns, ns))
                keep = rng.uniform(size=(ns, ns)) < 0.55
                sg = np.where(keep, hh, 0.0)
                for si in range(ns):
                    for sj in range(ns):
                        h = float(sg[si, sj])
                        if h < 0.012:
                            continue
                        bx = xl + (si + 0.5) * sub
                        by = yl + (sj + 0.5) * sub
                        boxes.append(PEBBLE_TMPL.format(
                            name="pb_%04d" % nb, x=bx, y=by, z=h * 0.5,
                            hx=sub * 0.45, hy=sub * 0.45, hz=h * 0.5))
                        nb += 1
                # boxgrid 每 0.4m 格 = 该格内最大 pebble 高度
                ratio = int(round(gw / sub))
                for ci in range(npatch):
                    for cj in range(npatch):
                        blk = sg[ci * ratio:(ci + 1) * ratio,
                                 cj * ratio:(cj + 1) * ratio]
                        H[i0 + ci, j0 + cj] = float(blk.max())
    H[H < 0.012] = 0.0

    # 普通 boxes/stairs patch 的 box geoms
    for i in range(ncell):
        for j in range(ncell):
            h = float(H[i, j])
            if h < 0.012:
                continue
            cx = x0 + (i + 0.5) * gw
            cy = y0 + (j + 0.5) * gw
            boxes.append(BOX_TMPL.format(
                name="bx_%04d" % nb, x=cx, y=cy, z=h * 0.5,
                hx=gw * 0.5, hy=gw * 0.5, hz=h * 0.5))
            nb += 1

    # ---- hfield: 平滑 patch (stairs_hf/boxes_hf/rough/slope) ----
    axis = np.linspace(-ext, ext, args.hf_n)
    XX, YY = np.meshgrid(axis, axis, indexing="ij")
    HF = np.zeros_like(XX)
    for pr in range(n):
        for pc in range(n):
            kind = types[pr * n + pc]
            xl = x0 + pc * patch
            yl = y0 + pr * patch
            m = ((XX >= xl) & (XX < xl + patch) & (YY >= yl) & (YY < yl + patch))
            if kind == "stairs_hf":
                hh = 0.10 if pr % 2 == 0 else 0.08
                fn = t_stairs_up(step_h=hh, step_w=0.32, platform_w=0.8,
                                 half=patch / 2, max_h=0.20)
                HF = np.where(m, fn(XX - xl, YY - yl, rng), HF)
            elif kind == "boxes_hf":
                fn = t_boxes(grid_w=0.4, height=0.13, half=patch / 2)
                HF = np.where(m, fn(XX - xl, YY - yl, rng), HF)
            elif kind == "rough":
                r = t_rough(0.045, 2.0, patch / 2, rng)
                li = np.clip(((XX - xl) / patch * 39).astype(int), 0, 39)
                lj = np.clip(((YY - yl) / patch * 39).astype(int), 0, 39)
                HF = np.where(m, r[li, lj], HF)
            elif kind == "slope":
                lx = (XX - (xl + patch / 2)) / (patch / 2)
                ly = (YY - (yl + patch / 2)) / (patch / 2)
                rr = np.maximum(np.abs(lx), np.abs(ly))
                HF = np.where(m, 0.12 * (1.0 - np.clip(rr, 0.0, 1.0)), HF)
    hmax = float(HF.max())
    elev = float(np.ceil(hmax / 0.05) * 0.05) if hmax > 0 else 0.05
    from PIL import Image
    png = np.flipud(HF.T)
    Image.fromarray(np.clip(png / elev * 255.0, 0, 255).astype(np.uint8)).save(
        os.path.join(ASSETS, args.prefix + ".png"))

    meta = dict(kind="boxgrid", grid_w=gw, n=ncell, x0=float(x0), y0=float(y0),
                x_min=float(x0), y_min=float(y0),
                height_max=float(max(H.max(), hmax)), seed=args.seed,
                heights=H.tolist(), n_boxes=nb, layout=layout,
                field_half=float(field_half), hfield_half=float(ext),
                notes="big multi-patch: pebble/boxes/stairs 真竖面 + hfield smooth")
    with open(os.path.join(ASSETS, args.prefix + ".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    scene = SCENE_TMPL.format(nb=nb, field_half=field_half, ext=ext, patch=patch,
                              elev=elev, prefix=args.prefix,
                              boxes="\n".join(boxes))
    sp = os.path.join(ROOT, "models", "go2",
                      "scene_mjx_terrain_%s.xml" % args.prefix)
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(scene)
    print("[big] %dx%d patches, field ±%.1fm, hfield ±%.1fm -> %d box geoms"
          % (n, n, field_half, ext, nb))
    print("[big] 写入 %s\n  %s" % (os.path.join(ASSETS, args.prefix + ".json"), sp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
