#!/usr/bin/env python
"""生成 "真竖面" 离散方块地形 (box geoms) + 平坦 hfield 底, 用于 R1 性能/几何验证。

动机 (docs/go2-terrain-recipe-survey.md §0.2/§6 R1): hfield 没有竖面, 0.094m 格会把
0.08-0.15m 箱棱抹成 ~50° 斜坡; 真机/真实箱子是垂直面。这里用 **box geom** 搭离散方块场,
地面仍是 hfield (全 0), 后续需要 env 用解析高度查询 + clearance 判接触。

用法:
  python sim/gen_terrain_boxgeom.py --n-box 200 --ext 6 --height 0.13 --keep 0.5 \
      --seed 1 --out_prefix terrain_boxgeom200
输出: assets/<prefix>.png (全 0 hfield), assets/<prefix>.json (方格高度表),
      models/go2/scene_mjx_terrain_<prefix>.xml
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")

SCENE_TMPL = '''<mujoco model="go2 boxgeom terrain scene">
  <!-- 由 sim/gen_terrain_boxgeom.py 生成, 不要手改。
       {nb} 个 box geom 组成离散方块场 (真竖面), 底面是平坦 hfield (全 0).
       方格 {gw:.2f}m, 高度 <= {hmax:.3f}m, 场范围 {ext:.1f}x{ext:.1f}m。 -->
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
    <material name="mix_ground" texture="mix_ground" texuniform="true" texrepeat="4 4" reflectance=".2"/>
    <material name="box_mat" rgba="0.55 0.45 0.35 1"/>
    <hfield name="terrain" file="{prefix}.png" size="{ext:.1f} {ext:.1f} 0.0500 0.0960"/>
  </asset>

  <worldbody>
    <!-- floor 名字不能改: 足端接触传感器绑定 geom2="floor" -->
    <geom name="floor" type="hfield" hfield="terrain" material="mix_ground"
      pos="0 0 0" friction="1.0 0.02 0.01"/>
{boxes}
  </worldbody>
</mujoco>
'''

BOX_TMPL = ('    <geom name="{name}" type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
            'size="{hx:.4f} {hy:.4f} {hz:.4f}" material="box_mat" '
            'contype="{ct}" conaffinity="{ca}" friction="0.9 0.02 0.01"/>')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-box", type=int, default=200, help="期望 box 数量 (实际受 keep 影响)")
    ap.add_argument("--ext", type=float, default=6.0, help="半边长 (m)")
    ap.add_argument("--grid", type=float, default=0.4, help="方格边长 (m)")
    ap.add_argument("--height", type=float, default=0.13, help="最大高度 (m)")
    ap.add_argument("--keep", type=float, default=0.5, help="保留概率 (与 hfield 版一致)")
    ap.add_argument("--x0", type=float, default=0.0, help="方块场中心 x (m)")
    ap.add_argument("--y0", type=float, default=0.0, help="方块场中心 y (m)")
    ap.add_argument("--span", type=float, default=0.0,
                    help="方块场半边长 (0=用 ext; 小于 ext 时周围是平地)")
    ap.add_argument("--contype", type=int, default=1)
    ap.add_argument("--conaffinity", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_prefix", default="terrain_boxgeom")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    span = args.span if args.span > 0 else args.ext
    rng = np.random.default_rng(args.seed)
    gw = float(args.grid)
    # 方格中心网格 (以 x0,y0 附近为中心), 覆盖 [-span, span]
    n = max(1, int(np.floor(2 * span / gw)))
    xs = args.x0 + (np.arange(n) + 0.5) * gw - 0.5 * n * gw
    ys = args.y0 + (np.arange(n) + 0.5) * gw - 0.5 * n * gw
    H = rng.uniform(0.0, args.height, (n, n)) * (rng.uniform(size=(n, n)) < args.keep)
    # 出生点附近留空 (避免出生即穿模): 半径 max(0.8, 2*gw) 内清零
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    r = np.sqrt((X - args.x0) ** 2 + (Y - args.y0) ** 2)
    H[r < max(0.9, 2.0 * gw)] = 0.0
    # 高度太小的格子不放 geom (等效平地)
    boxes = []
    nb = 0
    for i in range(n):
        for j in range(n):
            h = float(H[i, j])
            if h < 0.012:
                H[i, j] = 0.0          # 太小不放 geom -> 高度表也清零, 与物理严格一致
                continue
            boxes.append(BOX_TMPL.format(
                name="bx_%03d" % nb, x=float(xs[i]), y=float(ys[j]), z=h * 0.5,
                hx=gw * 0.5, hy=gw * 0.5, hz=h * 0.5,
                ct=args.contype, ca=args.conaffinity))
            nb += 1
    # 方格高度表 (给 env 解析查询用; row 沿 y, col 沿 x 与 hfield 约定一致)
    grid_meta = dict(
        kind="boxgrid", grid_w=gw, n=n, x0=float(args.x0), y0=float(args.y0),
        x_min=float(xs[0] - gw / 2), y_min=float(ys[0] - gw / 2),
        height_max=args.height, keep=args.keep, seed=args.seed,
        heights=H.tolist(), n_boxes=nb,
        axis_note="heights[i,j]: i 沿 +x, j 沿 +y; 世界坐标 x=x_min+(i+0.5)*gw, y=y_min+(j+0.5)*gw")
    print("[boxgeom] grid %dx%d gw=%.2fm span=%.1fm -> %d boxes (h<=%.3f)"
          % (n, n, gw, span, nb, args.height))
    if args.dry_run:
        return 0

    os.makedirs(ASSETS, exist_ok=True)
    from PIL import Image
    png = os.path.join(ASSETS, args.out_prefix + ".png")
    Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(png)
    with open(os.path.join(ASSETS, args.out_prefix + ".json"), "w", encoding="utf-8") as fh:
        json.dump(grid_meta, fh)
    scene = SCENE_TMPL.format(nb=nb, gw=gw, hmax=args.height, ext=args.ext,
                              prefix=args.out_prefix, boxes="\n".join(boxes))
    sp = os.path.join(ROOT, "models", "go2", "scene_mjx_terrain_%s.xml" % args.out_prefix)
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(scene)
    print("[boxgeom] 写入 %s\n  %s" % (png, sp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
