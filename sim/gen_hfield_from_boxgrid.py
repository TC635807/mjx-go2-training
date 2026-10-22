#!/usr/bin/env python
"""把 gen_terrain_boxgeom.py 的 box 方格高度表转成 **hfield 版** 场景。

用途: 同一份地形, 一个用 box geom (真竖面), 一个用 hfield (斜边) —— 同出生协议
(环形出生) 下对比策略存活率, 量化 "hfield 把竖面抹成斜坡" 对评测的影响。

用法: python sim/gen_hfield_from_boxgrid.py --bg_json models/go2/assets/bg200.json \
        --prefix bg200_hf --n 128 --ext 6
输出: assets/<prefix>.png + models/go2/scene_mjx_terrain_<prefix>.xml
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")

SCENE_TMPL = '''<mujoco model="go2 hfield-from-boxgrid scene">
  <!-- 由 sim/gen_hfield_from_boxgrid.py 生成 (把 boxgeom 方格高度表采样成 hfield).
       n={n}x{n}, 格距 {cell:.3f}m, 高度 <= {elev:.3f}m。 -->
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
    <hfield name="terrain" file="{prefix}.png" size="{ext:.1f} {ext:.1f} {elev:.4f} 0.0960"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="terrain" material="mix_ground"
      pos="0 0 0" friction="1.0 0.02 0.01"/>
  </worldbody>
</mujoco>
'''


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--bg_json", required=True)
  ap.add_argument("--prefix", default="bg_hf")
  ap.add_argument("--n", type=int, default=128)
  ap.add_argument("--ext", type=float, default=6.0)
  args = ap.parse_args(argv)
  meta = json.load(open(args.bg_json if os.path.isabs(args.bg_json)
                        else os.path.join(ROOT, args.bg_json)))
  H = np.asarray(meta["heights"], np.float64)
  gw, nb = float(meta["grid_w"]), int(meta["n"])
  x0, y0 = float(meta["x_min"]), float(meta["y_min"])
  axis = np.linspace(-args.ext, args.ext, args.n)
  XX, YY = np.meshgrid(axis, axis, indexing="ij")     # row=x, col=y? -> 用世界约定
  # 与 hfield 约定一致: terrain_mix 生成器是 png=flipud(h.T), 世界(x,y)==(i,j);
  # 这里直接按 (row=x, col=y) 建 H, 写 png 时 flipud(h.T) 抵消。
  i = np.floor((XX - x0) / gw).astype(int)
  j = np.floor((YY - y0) / gw).astype(int)
  ok = (i >= 0) & (i < nb) & (j >= 0) & (j < nb)
  Hf = np.where(ok, H[np.clip(i, 0, nb - 1), np.clip(j, 0, nb - 1)], 0.0)
  # 注意: MuJoCo 会把 hfield PNG **按图像最大值归一化到 [0,1]** (实测: 像素 220/220
  # -> 1.0; 128/220 -> 0.582), 所以实际高度 = pixel/max_pixel * elevation。
  # 要让高度表与 box geom 版本逐格一致, 必须令 elevation = Hf.max() (而不是
  # 向上取整到 5cm), 否则地形会被按 elev/Hmax 的比例整体放大。
  elev = max(1e-4, float(Hf.max()))
  from PIL import Image
  os.makedirs(ASSETS, exist_ok=True)
  png = np.flipud(Hf.T)
  Image.fromarray((png / elev * 255).astype(np.uint8)).save(
      os.path.join(ASSETS, args.prefix + ".png"))
  sp = os.path.join(ROOT, "models", "go2",
                    "scene_mjx_terrain_%s.xml" % args.prefix)
  with open(sp, "w", encoding="utf-8") as fh:
    fh.write(SCENE_TMPL.format(prefix=args.prefix, n=args.n,
                               cell=2 * args.ext / (args.n - 1), elev=elev,
                               ext=args.ext))
  print("[hfield_from_bg] %s -> png %s, scene %s (elev %.3f, hmax %.3f)"
        % (args.bg_json, png.shape, sp, elev, Hf.max()))
  return 0


if __name__ == "__main__":
  sys.exit(main())
