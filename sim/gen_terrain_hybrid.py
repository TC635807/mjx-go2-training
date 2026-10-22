#!/usr/bin/env python
"""生成 **混合几何** 地形: 离散障碍 (boxes/stairs) 用 **box geom 真竖面**, 平滑地形
(rough/slope) 用 hfield。环境侧 terrain_height = max(hfield, boxgrid) 已支持
(envs/go2_walk.py: terrain_boxgrid_json)。

为什么: hfield 无竖面 (0.094m 格把 0.1m 台阶抹成 ~50° 斜坡), 实测同一地形
hfield 版 v6a 存活 80.5% vs box geom 版 26.8% —— 策略会利用斜边, 是 sim2real 隐患。
(Rudin 2022 / Miki 2022 同结论; 见 docs/go2-terrain-recipe-survey.md)

布局 (patch 2.8m = 7 x 0.4m 格, 场范围 ±4.2m, 外圈 4.2~6m 平地用于出生):
  row0: boxes(0.09)   stairs_x   rough(hfield)
  row1: stairs_y      boxes(0.13) slope(hfield)
  row2: rough(hfield) stairs_x   boxes(0.06)

用法:
  python sim/gen_terrain_hybrid.py --prefix hyb1 --seed 1
输出: assets/<prefix>.png (hfield: smooth patches), assets/<prefix>.json (box grid),
      models/go2/scene_mjx_terrain_<prefix>.xml
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_terrain_go2_mix import t_boxes, t_stairs_up  # noqa: E402

SCENE_TMPL = '''<mujoco model="go2 hybrid terrain scene (hfield smooth + box geoms discrete)">
  <!-- 由 sim/gen_terrain_hybrid.py 生成, 不要手改。
       smooth patches: hfield (elev {elev:.4f}); discrete patches: {nb} 个 box geom (真竖面). -->
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
    <hfield name="terrain" file="{prefix}.png" size="{ext:.1f} {ext:.1f} {elev:.4f} 0.0960"/>
  </asset>
  <worldbody>
    <!-- floor 名字不能改: 足端接触传感器绑定 geom2="floor" (本场景用 clearance 判接触) -->
    <geom name="floor" type="hfield" hfield="terrain" material="mix_ground"
      pos="0 0 0" friction="1.0 0.02 0.01"/>
{boxes}
  </worldbody>
</mujoco>
'''

BOX_TMPL = ('    <geom name="{name}" type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
            'size="{hx:.4f} {hy:.4f} {hz:.4f}" material="box_mat" '
            'contype="1" conaffinity="1" friction="0.9 0.02 0.01"/>')


def t_rough(amp, corr, half, rng):
    from scipy.ndimage import gaussian_filter
    n = 40
    w = rng.standard_normal((n, n))
    w = gaussian_filter(w, sigma=corr, mode="nearest")
    w = w / (np.abs(w).max() + 1e-9)
    return amp * (0.5 * (w + 1.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="hyb1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ext", type=float, default=6.0)
    ap.add_argument("--patch", type=float, default=2.8, help="patch 边长 (7x0.4m)")
    ap.add_argument("--grid", type=float, default=0.4)
    ap.add_argument("--hf_n", type=int, default=128)
    ap.add_argument("--layout",
                    choices=["mix", "stairs_only", "both", "vbox"], default="mix")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    gw = float(args.grid)
    npatch = max(1, int(round(args.patch / gw)))       # 每 patch 格数 (7)
    patch = npatch * gw                                # 实际 patch 边长 (消浮点误差)
    half = patch * 1.5                                 # 场半宽 = 4.2

    # 3x3 patch 类型
    if args.layout == "stairs_only":
        layout = [
            ["stairs_x", "stairs_y", "stairs_x"],
            ["stairs_y", "stairs_x", "stairs_y"],
            ["stairs_x", "stairs_y", "stairs_x"],
        ]
    elif args.layout == "vbox":
        # 竖面方块专项: boxes_v 5/9 面积 (0.13/0.07/0.10) + stairs_v 2/9 + 平滑 hfield 2/9。
        # 用于从 v15 再短训 10M, 把竖面方块从 ~60% 往上推。
        layout = [
            ["boxes_va", "stairs_vb", "boxes_vb"],
            ["rough", "flat", "slope"],
            ["boxes_vc", "stairs_va", "boxes_va"],
        ]
    elif args.layout == "both":
        # 最终训练地形: **同时**包含真竖面 (box geom) 与 hfield 版离散/平滑地形。
        # 目的: 让一个策略同时适应"真机竖面世界"和"历史 hfield 评测口径", 两边都不丢。
        layout = [
            ["boxes_va", "stairs_hf", "flat"],
            ["stairs_va", "boxes_hf", "rough"],
            ["slope", "boxes_vb", "stairs_vb"],
        ]
    else:
        layout = [
            ["boxes_a", "stairs_x", "rough"],
            ["stairs_y", "boxes_b", "slope"],
            ["rough_x", "stairs_x", "boxes_c"],
        ]
    box_hmax = {"boxes_a": 0.09, "boxes_b": 0.13, "boxes_c": 0.06,
                "boxes_va": 0.13, "boxes_vb": 0.07, "boxes_vc": 0.10}
    stair_hrow = [0.05, 0.08, 0.11]     # 每行难度 -> 单级踏步高
    stair_prof = [0.0, 0.06, 0.12, 0.18, 0.12, 0.06, 0.0]   # mix 档沿行进方向踏步高

    ncell = npatch * 3
    x0 = -half
    y0 = -half
    H = np.zeros((ncell, ncell), np.float64)          # 方格高度 (m)
    is_discrete = np.zeros((ncell, ncell), bool)
    for pr in range(3):                 # pr = patch 行 (y)
        for pc in range(3):             # pc = patch 列 (x)
            kind = layout[pr][pc]
            i0, i1 = pc * npatch, (pc + 1) * npatch      # i -> x
            j0, j1 = pr * npatch, (pr + 1) * npatch      # j -> y
            if kind in ("boxes_va", "boxes_vb", "boxes_vc"):
                hm = box_hmax[kind]
                blk = rng.uniform(0.0, hm, (npatch, npatch))
                keep = rng.uniform(size=(npatch, npatch)) < 0.5
                H[i0:i1, j0:j1] = np.where(keep, blk, 0.0)
                is_discrete[i0:i1, j0:j1] = True
            elif kind in box_hmax:
                hm = box_hmax[kind]
                blk = rng.uniform(0.0, hm, (npatch, npatch))
                keep = rng.uniform(size=(npatch, npatch)) < 0.5
                H[i0:i1, j0:j1] = np.where(keep, blk, 0.0)
                is_discrete[i0:i1, j0:j1] = True
            elif kind in ("stairs_va", "stairs_vb"):
                h = 0.11 if kind == "stairs_va" else 0.06
                prof = np.asarray([0.0, h, 2 * h, 2 * h, h, 0.0, 0.0])
                if pr == 0:
                    H[i0:i1, j0:j1] = prof[:, None]
                else:
                    H[i0:i1, j0:j1] = prof[None, :]
                is_discrete[i0:i1, j0:j1] = True
            elif kind.startswith("stairs"):
                if args.layout == "stairs_only":
                    h = stair_hrow[pr]
                    prof = np.asarray([0.0, h, 2 * h, 2 * h, h, 0.0, 0.0])
                else:
                    prof = np.asarray(stair_prof)
                if kind == "stairs_x":
                    H[i0:i1, j0:j1] = prof[:, None]
                else:
                    H[i0:i1, j0:j1] = prof[None, :]
                is_discrete[i0:i1, j0:j1] = True
            # rough/slope 留 0, 由 hfield 负责
    # 离散格高度过小 -> 0 (不放 geom, 与高度表一致)
    H[H < 0.012] = 0.0
    # 出生圈外 (|x| 或 |y| > half) 一定为 0
    # (方格场只覆盖 ±half, 外圈平地)

    # ---- hfield: 只放 smooth patch, 离散 patch/外圈为 0 ----
    axis = np.linspace(-args.ext, args.ext, args.hf_n)
    XX, YY = np.meshgrid(axis, axis, indexing="ij")     # row=x, col=y
    HF = np.zeros_like(XX)
    for pr in range(3):
        for pc in range(3):
            kind = layout[pr][pc]
            xl = x0 + pc * patch
            yl = y0 + pr * patch
            m = ((XX >= xl) & (XX < xl + patch)
                 & (YY >= yl) & (YY < yl + patch))
            if kind == "stairs_hf":
                hh = 0.10 if pr == 0 else 0.08
                fn = t_stairs_up(step_h=hh, step_w=0.32, platform_w=0.8,
                                 half=patch / 2, max_h=0.20)
                HF = np.where(m, fn(XX - xl, YY - yl, rng), HF)
            elif kind == "boxes_hf":
                hm = 0.13 if pr == 1 else 0.08
                fn = t_boxes(grid_w=0.4, height=hm, half=patch / 2)
                HF = np.where(m, fn(XX - xl, YY - yl, rng), HF)
            elif kind in ("rough", "rough_x"):
                r = t_rough(0.045, 2.0, patch / 2, rng)
                li = np.clip(((XX - xl) / patch * 39).astype(int), 0, 39)
                lj = np.clip(((YY - yl) / patch * 39).astype(int), 0, 39)
                HF = np.where(m, r[li, lj], HF)
            elif kind == "slope":
                # 从 patch 边到中心升到 0.12 (金字塔斜坡)
                lx = (XX - (xl + patch / 2)) / (patch / 2)
                ly = (YY - (yl + patch / 2)) / (patch / 2)
                rr = np.maximum(np.abs(lx), np.abs(ly))
                HF = np.where(m, 0.12 * (1.0 - np.clip(rr, 0.0, 1.0)), HF)
    # hfield 边缘淡出无需 (patch 边界处为 0)
    elev = max(1e-4, float(HF.max()))
    # ---- 写 hfield png (MuJoCo 按图像最大值归一化, 故 elev=HF.max() 保证精确) ----
    from PIL import Image
    png = np.flipud(HF.T)
    Image.fromarray(np.clip(png / elev * 255.0, 0, 255).astype(np.uint8)).save(
        os.path.join(ASSETS, args.prefix + ".png"))

    # ---- box geoms ----
    boxes = []
    nb = 0
    for i in range(ncell):
        for j in range(ncell):
            h = float(H[i, j])
            if h < 0.012:
                continue
            cx = x0 + (i + 0.5) * gw
            cy = y0 + (j + 0.5) * gw
            boxes.append(BOX_TMPL.format(
                name="bx_%03d" % nb, x=cx, y=cy, z=h * 0.5,
                hx=gw * 0.5, hy=gw * 0.5, hz=h * 0.5))
            nb += 1
    # ---- JSON 高度表 (env 查表; 格式与 gen_terrain_boxgeom.py 相同) ----
    meta = dict(kind="boxgrid", grid_w=gw, n=ncell, x0=float(x0), y0=float(y0),
                x_min=float(x0), y_min=float(y0), height_max=float(H.max()),
                seed=args.seed, heights=H.tolist(), n_boxes=nb,
                layout=layout, notes="discrete patches = box geoms; rough/slope = hfield")
    with open(os.path.join(ASSETS, args.prefix + ".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    sp = os.path.join(ROOT, "models", "go2",
                      "scene_mjx_terrain_%s.xml" % args.prefix)
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(SCENE_TMPL.format(prefix=args.prefix, ext=args.ext,
                                   elev=elev, nb=len(boxes),
                                   boxes="\n".join(boxes)))
    print("[hybrid] %s: %d boxes, grid %dx%d gw=%.2f, hfield elev=%.3f (hmax %.3f)"
          % (args.prefix, nb, ncell, ncell, gw, elev, HF.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
