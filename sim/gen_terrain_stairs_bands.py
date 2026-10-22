#!/usr/bin/env python
"""生成 **"只有台阶" 的 7 条全长楼梯带 (bands)** 真竖面 box 地形 (v28 台阶专项)。

动机 (T3 调研 + stairs1 实测缺陷):
  * stairs1/stairs2 (sim/gen_terrain_stairs_box.py) 台阶格覆盖率只有 37.2%/33.5%,
    平地占 60%+, 随机出生仍可绕开 -> 台阶梯度被稀释;
  * 本生成器把 **7 条沿 ±x 的楼梯带铺满整个 22x22 格场地** (覆盖 86.4%),
    每条带 = 入口平 -> 逐级上 (踏面 0.4m) -> 顶部平台 -> 逐级下 -> 出口平;
  * 相邻带剖面不同 -> 带间有 0~0.44m 竖壁, 强迫沿带爬 (符合"只有台阶");
  * 每条带两端写入 spawn_entries (x=±4.6, y=带中心, yaw 朝带内),
    训练 env 可据此"楼梯口出生", 保证每回合都在爬 (见 random_init.spawn_stairs_prob)。

带宽/参数 (22 格 = 22 行, 宽度 4/4/3/3/3/3/2 合计 22):
  A: riser 0.06 x6 + 平台 8 格 (总升 0.36m, 宽 4=1.6m)   <- 真实楼梯比例
  B: riser 0.08 x5 + 平台 10 格 (0.40m, 宽 4=1.6m)
  C: riser 0.10 x4 + 平台 12 格 (0.40m, 宽 3=1.2m)
  D: riser 0.12 x3 + 平台 14 格 (0.36m, 宽 3=1.2m)
  E: riser 0.14 x3 + 平台 14 格 (0.42m, 宽 3=1.2m)     <- 训练主轴上限 (52% 站高)
  F: riser 0.18 x2 + 平台 16 格 (0.36m, 宽 3=1.2m)     <- 极限档 (不设通过门)
  G: riser 0.22 x2 + 平台 16 格 (0.44m, 宽 2=0.8m)     <- 极限档 (82% 站高)

用法:
  python sim/gen_terrain_stairs_bands.py --prefix bands1            # 混合梯度 (训练用)
  python sim/gen_terrain_stairs_bands.py --prefix bands_easy --profile easy
  python sim/gen_terrain_stairs_bands.py --prefix bands_u010 --uniform_h 0.10  # 评测高度阶梯
输出: models/go2/assets/<prefix>.png (全 0 hfield), <prefix>.json (boxgrid + spawn_entries),
      models/go2/scene_mjx_terrain_<prefix>.xml
"""
import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")

SCENE_TMPL = '''<mujoco model="go2 stairs-bands boxgeom scene">
  <!-- 由 sim/gen_terrain_stairs_bands.py 生成, 不要手改。
       {nb} 个 box geom, {nb_band} 条沿 ±x 的全长楼梯带 (真竖面), 底面 hfield 全 0。
       方格 {gw:.2f}m, 场 ±{field:.1f}m, 外圈 ±{ext:.1f}m 平地用于出生。
       台阶格覆盖 {cov:.1f}%; 单级 riser {hmin:.2f}~{hmax_riser:.2f}m, 带顶平台 {hmax:.2f}m。 -->
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
    <hfield name="terrain" file="{prefix}.png" size="{ext:.1f} {ext:.1f} 0.0010 0.0960"/>
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

# (name, riser h, n_up, platform cells, width cells)
BANDS = [
    # (name, riser h, n_up, platform cells, width cells)
    # 长度 = 2*n_up + plat - 1 = 22 (no_flat_ends) 或 2*n_up + plat + 2 = 22 (flat_ends)
    ("A", 0.06, 6, 11, 4),
    ("B", 0.08, 5, 13, 4),
    ("C", 0.10, 4, 15, 3),
    ("D", 0.12, 3, 17, 3),
    ("E", 0.14, 3, 17, 3),
    ("F", 0.18, 2, 19, 3),
    ("G", 0.22, 2, 19, 2),
]
PROFILES = {
    "normal": [0.06, 0.08, 0.10, 0.12, 0.14, 0.18, 0.22],
    "easy":   [0.06, 0.06, 0.08, 0.08, 0.10, 0.10, 0.12],
    "hard":   [0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.22],
}


def band_profile(h, n_up, plat, no_flat_ends=False):
    """楼梯带剖面: 上行 n_up 级 -> 顶部平台 -> 下行 n_up-1 级。

    默认 (no_flat_ends=False): [入口平] + 上行 + 平台 + 下行 + [出口平 x2]
      长度 = 2*n_up + plat + 2 = 22。
    no_flat_ends=True: 首格/末格直接取 h (无入口/出口平), 长度 = 2*n_up + plat - 1 = 22。
      动机 (B 负结果): 默认剖面两端各有 1~2 格平地, 在场地 x 两端形成沿 y 的平走廊;
      策略发现"转身沿走廊走"也能拿到速度奖励 -> 绕开台阶 (B 训练 8M 后 climb=0.007m)。
      去掉平端后入口即是 riser, 不爬就无法前进。
    """
    up = [h * (k + 1) for k in range(n_up)]
    plat_cells = [h * n_up] * int(plat)
    down = [h * (k + 1) for k in range(n_up - 2, -1, -1)]
    if no_flat_ends:
        return up + plat_cells + down
    return [0.0] + up + plat_cells + down + [0.0, 0.0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="bands1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid", type=float, default=0.4)
    ap.add_argument("--ext", type=float, default=6.0)
    ap.add_argument("--field", type=float, default=4.4)
    ap.add_argument("--profile", choices=list(PROFILES), default="normal")
    ap.add_argument("--uniform_h", type=float, default=0.0,
                    help=">0 时所有带用同一单级 riser 高度 (评测高度阶梯用)")
    ap.add_argument("--no_spawn_entries", action="store_true")
    ap.add_argument("--flat_ends", action="store_true",
                    help="保留两端平地 (旧布局; B 负结果证明策略会沿平走廊绕过台阶)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    gw = float(args.grid)
    field = float(args.field)
    n = int(round(2.0 * field / gw))          # 22
    if n != len(band_profile(0.1, 3, 14)):
        raise SystemExit(f"grid/field 不匹配: n={n}, 需要 22 格")
    risers = PROFILES[args.profile]
    if args.uniform_h > 0:
        risers = [float(args.uniform_h)] * len(BANDS)

    H = np.zeros((n, n), np.float64)     # H[i, j]: i 沿 +x, j 沿 +y
    bands_meta = []
    spawn_entries = []
    j0 = 0
    nb_band = 0
    for (name, _h, n_up, plat, width), h in zip(BANDS, risers):
        prof = band_profile(h, n_up, plat,
                            no_flat_ends=not args.flat_ends)
        if len(prof) != n:
            raise SystemExit(
                f"band {name} profile len {len(prof)} != {n} "
                f"(no_flat_ends={not args.flat_ends}; 检查 BANDS 的 plat 配置)")
        for k in range(n):
            H[k, j0:j0 + width] = prof[k]
        yc = -field + (j0 + width / 2.0) * gw
        bands_meta.append(dict(name=name, riser=float(h), n_up=int(n_up),
                               platform_cells=int(plat), width_cells=int(width),
                               width_m=float(width * gw), y_center=float(yc),
                               top_h=float(prof[n // 2] if False else h * n_up),
                               j0=int(j0), j1=int(j0 + width)))
        if not args.no_spawn_entries:
            # 左端 (x=-field-0.2) 朝 +x; 右端 (x=+field+0.2) 朝 -x
            xL = -field - 0.2
            xR = field + 0.2
            spawn_entries.append(dict(x=xL, y=yc, yaw=0.0, band=name))
            spawn_entries.append(dict(x=xR, y=yc, yaw=float(np.pi), band=name))
        j0 += width
        nb_band += 1
    if j0 != n:
        raise SystemExit(f"带宽合计 {j0} != {n}")
    H[H < 0.012] = 0.0

    # box geoms
    boxes = []
    for i in range(n):
        for j in range(n):
            h = float(H[i, j])
            if h <= 0.0:
                continue
            cx = -field + (i + 0.5) * gw
            cy = -field + (j + 0.5) * gw
            boxes.append(BOX_TMPL.format(
                name="bx_%03d" % len(boxes), x=cx, y=cy, z=h * 0.5,
                hx=gw * 0.5, hy=gw * 0.5, hz=h * 0.5))
    nb = len(boxes)
    nz = int((H > 0).sum())
    cov = 100.0 * nz / H.size
    meta = dict(kind="boxgrid", grid_w=gw, n=n,
                x0=float(-field), y0=float(-field),
                x_min=float(-field), y_min=float(-field),
                height_max=float(H.max()), seed=args.seed,
                profile=args.profile, uniform_h=float(args.uniform_h),
                n_boxes=nb, heights=H.tolist(),
                bands=bands_meta, spawn_entries=spawn_entries,
                coverage_pct=float(cov),
                notes="stairs-bands boxgeom terrain (gen_terrain_stairs_bands.py); "
                      "floor hfield all zero; env uses max(hfield, boxgrid)")
    print("[bands] %s profile=%s uniform_h=%.2f: %dx%d gw=%.2f -> %d boxes, "
          "台阶格覆盖 %.1f%% (T3 目标 >=85%%)"
          % (args.prefix, args.profile, args.uniform_h, n, n, gw, nb, cov))
    for b in bands_meta:
        print("[bands]   %-2s riser %.2fm x%d + 平台 %d 格 (宽 %.1fm) 顶高 %.2fm"
              % (b["name"], b["riser"], b["n_up"], b["platform_cells"],
                 b["width_m"], b["top_h"]))
    if args.dry_run:
        return 0

    os.makedirs(ASSETS, exist_ok=True)
    from PIL import Image
    Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(
        os.path.join(ASSETS, args.prefix + ".png"))
    with open(os.path.join(ASSETS, args.prefix + ".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    sp = os.path.join(ROOT, "models", "go2", "scene_mjx_terrain_%s.xml" % args.prefix)
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(SCENE_TMPL.format(prefix=args.prefix, ext=args.ext, field=args.field,
                                   gw=gw, nb=nb, nb_band=nb_band, cov=cov,
                                   hmax=float(H.max()), hmax_riser=float(max(risers)),
                                   hmin=float(min(risers)),
                                   boxes="\n".join(boxes)))
    print("[bands] 写出", os.path.join(ASSETS, args.prefix + ".png/json"), "和", sp)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
