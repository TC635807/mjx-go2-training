#!/usr/bin/env python
"""生成 **"只有台阶"** 的真竖面 box 地形 (v28 台阶专项训练用)。

动机 (2026-10-21, 用户要求"构造一个只有台阶的环境进行训练"):
  * 现有 terrains 里台阶只占 1/9~2/9 patch, 随机出生 + 随机朝向时, 一个 episode
    里真正遇到台阶的概率很低 -> 台阶的梯度信号被平地/粗糙面稀释;
  * hfield 版台阶没有竖面 (0.094m 格把 0.10m 踏步抹成 ~50° 坡), 策略会用斜边;
  * 所以用 **box geom 真竖面** 的台阶场 + boxgrid 高度表 (训练 env 的
    terrain_height = max(hfield, boxgrid), contact_mode=clearance), 台阶占满整个
    场地的可走区域, 让 8192 个随机出生环境**总是**在爬台阶。

布局 (grid 0.4m; 场 ±field, 外圈 [field, ext] 平地用于环形出生):
  4 条长楼梯 (沿 ±x/±y, 放在四个互不重叠的带内):
      A 型: 踏步 0.08m x 5 级 (总升 0.40m, 真实楼梯比例) + 0.8m 平台 + 下行
      B 型: 踏步 0.15m x 3 级 (总升 0.45m, 高难度) + 平台 + 下行
      C 型: 单级 0.22m 大台阶 (跨上/跨下, 极限高度)
      D 型: 踏步 0.10m x 6 级 (长缓梯, 总升 0.60m) + 平台 + 下行
  1 座金字塔台阶 (四面可上, 每面 2 级 0.10m, 顶面平台)
  若干离散单块 (0.08~0.20m, 1x1 / 1x2 格) 作为"路缘石"
  其余为平地 (供接近/转身)。

用法:
  python sim/gen_terrain_stairs_box.py --prefix stairs1 --seed 1
  python sim/gen_terrain_stairs_box.py --prefix stairs2 --seed 2 --level hard
输出: models/go2/assets/<prefix>.png (全 0 hfield),
      models/go2/assets/<prefix>.json (boxgrid 高度表, env 查表用),
      models/go2/scene_mjx_terrain_<prefix>.xml (真竖面 box geoms)
训练示例:
  python -u -m train.train_go2 --terrain \
    --terrain_xml models/go2/scene_mjx_terrain_stairs1.xml \
    --boxgrid_json models/go2/assets/stairs1.json --contact_mode clearance --spawn_ring ...
"""
import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")

SCENE_TMPL = '''<mujoco model="go2 stairs-only boxgeom scene">
  <!-- 由 sim/gen_terrain_stairs_box.py 生成, 不要手改。
       {nb} 个 box geom 组成"只有台阶"的场地 (真竖面), 底面是平坦 hfield (全 0)。
       方格 {gw:.2f}m, 场范围 ±{field:.1f}m, 外圈 ±{ext:.1f}m 平地用于出生。
       高度统计: max={hmax:.3f}m, >0.02m 的格 {nz}/{nn} ({cov:.0f}%)。 -->
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


def stair_profile(h, n_up, platform_cells):
    """上行 n_up 级 (每级 h) -> 平台 -> 下行 n_up 级 -> 平地一格 (0)。

    长度 = n_up + platform_cells + (n_up-1) + 1; 末格高 0 表示回到地面。
    """
    up = [h * (s + 1) for s in range(n_up)]
    plat = [h * n_up] * int(platform_cells)
    down = [h * (s + 1) for s in range(n_up - 2, -1, -1)] if n_up > 1 else []
    return up + plat + down + [0.0]


class Layout:
    """占用栅格 + 高度表 (格心高度; 每格一个 box geom)。"""

    def __init__(self, gw, field):
        self.gw = float(gw)
        self.field = float(field)
        self.n = int(round(2.0 * field / gw))
        self.H = np.zeros((self.n, self.n), np.float64)
        self.occ = np.zeros((self.n, self.n), bool)   # 占位(含 1 格安全间隙)
        self.structures = []

    def cell_xy(self, i, j):
        return (-self.field + (i + 0.5) * self.gw,
                -self.field + (j + 0.5) * self.gw)

    def free_region(self, i0, i1, j0, j1):
        if i0 < 0 or j0 < 0 or i1 > self.n or j1 > self.n:
            return False
        return not self.occ[i0:i1, j0:j1].any()

    def mark_occ(self, i0, i1, j0, j1, margin=1):
        a0, a1 = max(0, i0 - margin), min(self.n, i1 + margin)
        b0, b1 = max(0, j0 - margin), min(self.n, j1 + margin)
        self.occ[a0:a1, b0:b1] = True

    def put_profile(self, name, i0, j0, profile, width, axis):
        """沿 axis('x'|'y') 铺一条台阶剖面; profile = 高度序列(单格)。"""
        n = len(profile)
        if axis == "x":
            i1, j1 = i0 + n, j0 + width
        else:
            i1, j1 = i0 + width, j0 + n
        if not self.free_region(i0, i1, j0, j1):
            return False
        for k, h in enumerate(profile):
            if axis == "x":
                self.H[i0 + k, j0:j1] = h
            else:
                self.H[i0:i1, j0 + k] = h
        self.mark_occ(i0, i1, j0, j1, margin=1)
        self.structures.append(dict(name=name, kind="stair", axis=axis,
                                    i0=int(i0), j0=int(j0), width=int(width),
                                    n_up=int(sum(1 for v in profile if v > 0)) and None,
                                    profile=[float(v) for v in profile]))
        return True

    def put_block(self, name, i0, j0, h, w=1, d=1):
        if not self.free_region(i0, i0 + w, j0, j0 + d):
            return False
        self.H[i0:i0 + w, j0:j0 + d] = h
        self.mark_occ(i0, i0 + w, j0, j0 + d, margin=1)
        self.structures.append(dict(name=name, kind="block", i0=int(i0), j0=int(j0),
                                    w=int(w), d=int(d), h=float(h)))
        return True

    def put_pyramid(self, name, i0, j0, ring_h, half_cells):
        """金字塔台阶: 中心平台 + 向外逐级降低 (每级 ring_h)。"""
        side = half_cells * 2
        i1, j1 = i0 + side, j0 + side
        if not self.free_region(i0, i1, j0, j1):
            return False
        for di in range(side):
            for dj in range(side):
                ci = abs(di - (side - 1) / 2.0)
                cj = abs(dj - (side - 1) / 2.0)
                lev = int(np.floor(half_cells - max(ci, cj) + 0.5))
                self.H[i0 + di, j0 + dj] = float(max(0, lev) * ring_h)
        self.mark_occ(i0, i1, j0, j1, margin=1)
        self.structures.append(dict(name=name, kind="pyramid", i0=int(i0),
                                    j0=int(j0), side=int(side),
                                    half_cells=int(half_cells)))
        return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="stairs1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid", type=float, default=0.4, help="方格边长 (m)")
    ap.add_argument("--ext", type=float, default=6.0, help="hfield/世界半边长 (m)")
    ap.add_argument("--field", type=float, default=4.4, help="box 场半边长 (m)")
    ap.add_argument("--level", choices=["easy", "normal", "hard"], default="normal")
    ap.add_argument("--n-scatter", type=int, default=12, help="散点单块数量")
    ap.add_argument("--n-pyramid", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    L = Layout(args.grid, args.field)
    n = L.n

    # 四条长楼梯的 (名字, 踏步高, 上行级数, 平台格数, 走向, 起点, 宽度)
    if args.level == "easy":
        stairs = [("A", 0.06, 6, 2, "x", 2, 2, 4), ("B", 0.10, 4, 2, "y", 17, 2, 4),
                  ("C", 0.15, 1, 3, "y", 9, 9, 4), ("D", 0.08, 6, 2, "x", 2, 16, 3)]
    elif args.level == "hard":
        stairs = [("A", 0.10, 4, 2, "x", 2, 2, 4), ("B", 0.16, 3, 2, "y", 17, 2, 4),
                  ("C", 0.22, 1, 3, "y", 9, 9, 4), ("D", 0.12, 4, 2, "x", 2, 16, 3)]
    else:
        stairs = [("A", 0.08, 5, 2, "x", 2, 2, 4), ("B", 0.15, 3, 2, "y", 17, 2, 4),
                  ("C", 0.22, 1, 3, "y", 9, 9, 4), ("D", 0.10, 6, 2, "x", 2, 16, 3)]
    for name, h, n_up, plat, axis, i0, j0, width in stairs:
        prof = stair_profile(h, n_up, plat)
        ok = L.put_profile("stair_%s" % name, i0, j0, prof, width, axis)
        if not ok:
            raise SystemExit("[stairs] 布局冲突: stair_%s @ (%d,%d) len=%d" % (name, i0, j0, len(prof)))

    # 金字塔 (拒绝采样放剩余空间)
    for k in range(args.n_pyramid):
        placed = False
        for _attempt in range(500):
            half = 3
            i0 = int(rng.integers(1, n - 2 * half - 1))
            j0 = int(rng.integers(1, n - 2 * half - 1))
            if L.put_pyramid("pyr_%d" % k, i0, j0, ring_h=0.10, half_cells=half):
                placed = True
                break
        if not placed:
            print("[stairs] 警告: 金字塔 %d 没找到位置" % k)

    # 散点单块 ("路缘石"): 1x1 / 1x2, 高度 0.08~0.20
    hs = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    for k in range(args.n_scatter):
        for _attempt in range(200):
            w = 1 if rng.random() < 0.7 else 2
            d = 1 if rng.random() < 0.7 else 2
            i0 = int(rng.integers(1, n - w - 1))
            j0 = int(rng.integers(1, n - d - 1))
            if L.put_block("blk_%02d" % k, i0, j0, float(rng.choice(hs)), w, d):
                break

    H = L.H
    H[H < 0.012] = 0.0   # 与 box geom 保持一致: 太矮不放 geom
    nz = int((H > 0.0).sum())
    nn = H.size
    boxes = []
    for i in range(n):
        for j in range(n):
            h = float(H[i, j])
            if h <= 0.0:
                continue
            cx, cy = L.cell_xy(i, j)
            boxes.append(BOX_TMPL.format(
                name="bx_%03d" % len(boxes), x=cx, y=cy, z=h * 0.5,
                hx=args.grid * 0.5, hy=args.grid * 0.5, hz=h * 0.5))
    nb = len(boxes)
    meta = dict(kind="boxgrid", grid_w=args.grid, n=n,
                x0=float(-args.field), y0=float(-args.field),
                x_min=float(-args.field), y_min=float(-args.field),
                height_max=float(H.max()), seed=args.seed,
                level=args.level, n_boxes=nb, heights=H.tolist(),
                structures=L.structures,
                notes="stairs-only boxgeom terrain (gen_terrain_stairs_box.py); "
                      "floor hfield all zero; env uses max(hfield, boxgrid)")
    print("[stairs] %s level=%s: grid %dx%d gw=%.2f field=±%.1f ext=±%.1f -> %d boxes"
          % (args.prefix, args.level, n, n, args.grid, args.field, args.ext, nb))
    print("[stairs] 高度: max=%.2fm  >0.02m: %d/%d (%.0f%%)  >0.10m: %d"
          % (H.max(), nz, nn, 100.0 * nz / nn, int((H > 0.10).sum())))
    for s in L.structures:
        print("[stairs]   %-10s %s" % (s["name"], s["kind"]))
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
                                   gw=args.grid, nb=nb, hmax=H.max(), nz=nz, nn=nn,
                                   cov=100.0 * nz / nn, boxes="\n".join(boxes)))
    print("[stairs] 写出", os.path.join(ASSETS, args.prefix + ".png/json"), "和", sp)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
