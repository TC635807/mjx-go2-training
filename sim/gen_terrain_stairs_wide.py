#!/usr/bin/env python
"""生成"一条宽楼梯"的真竖面 box 地形 (v29 台阶课程, 仿 legged_gym/PIE pyramid stairs).

动机 (2026-10-21, 用户要求解决"上台阶成功率低 + 绕开台阶"):
* 开源实现 (legged_gym utils/terrain.py, IsaacLab terrain_generator, mjlab/parkour_mjlab PIE)
  的台阶都是 **每 env 一块 8x8m 地形瓦片 + 10 级难度课程**:
  legged_gym: step_height = 0.05 + 0.18*difficulty (10 行 = 0.05..0.23m), step_width=0.31m;
  PIE (Go2, parkour_mjlab): step_height_range=(0.03,0.25), step_width=0.30,
  10 行 x 20 列, 每 env 一块瓦片, 出生在瓦片原点、yaw=0、命令前向偏置
  (obstacle 地形 80% 直行 / 20% 小yaw)。
  课程晋升: 走够半块地形 (distance > size/2) 且未终止 -> 升一级; 终止或进度不足一半 -> 降级。
  我们的训练栈 (MJX warp + brax) 只有**一张共享地形**, 不能每 env 一块瓦片,
  所以用 **"多阶段 warm-start 课程"**: 每阶段一张单级 riser 高度的宽楼梯地形,
  从 0.02m 起逐级加高 (0.02/0.03/0.05/0.08/0.11/0.14), 每阶段训练 2-3M 步。

地形: 全场 (±4.4m, 22x22 个 0.4m 格) 一条**通宽**楼梯:
  入口第 1 格即为 riser (无平走廊, 防止绕行); 上行 n_up 级 -> 顶部平台 -> 下行 n_up-1 级;
  长度 2*n_up + plat - 1 = 22; 楼梯面通宽 (22 行), 侧向无台阶差异 -> 只能向前爬。
  spawn_entries: 左右两端各一个 (x=±(field+0.3), y=0, yaw 朝场内), env 的
  random_init.spawn_stairs_prob=1.0 时机器人固定在楼梯口出生并朝向楼梯。

用法:
  python sim/gen_terrain_stairs_wide.py --prefix wide_h003 --riser 0.03
  python sim/gen_terrain_stairs_wide.py --prefix wide_h002 --riser 0.02 --n-up 6
输出: models/go2/assets/<prefix>.png/json + models/go2/scene_mjx_terrain_<prefix>.xml
"""
import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")

SCENE_TMPL = '''<mujoco model="go2 wide stairs boxgeom scene">
  <!-- 由 sim/gen_terrain_stairs_wide.py 生成, 不要手改。
       {nb} 个 box geom 组成一条通宽楼梯 (真竖面), 底面 hfield 全 0。
       单级 riser {riser:.3f}m, 上行 {n_up} 级 + 平台 {plat} 格 + 下行; 场 ±{field:.1f}m。 -->
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
    <geom name="floor" type="hfield" hfield="terrain" material="mix_ground"
      pos="0 0 0" friction="1.0 0.02 0.01"/>
{boxes}
  </worldbody>
</mujoco>
'''

BOX_TMPL = ('    <geom name="{name}" type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
            'size="{hx:.4f} {hy:.4f} {hz:.4f}" material="box_mat" '
            'contype="1" conaffinity="1" friction="0.9 0.02 0.01"/>')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="wide_h003")
    ap.add_argument("--riser", type=float, default=0.03, help="单级踏步高 (m)")
    ap.add_argument("--n-up", type=int, default=6, help="上行级数")
    ap.add_argument("--entrance-cells", type=int, default=3,
                    help="入口平道格数 (助跑; 长度约束 plat = 22 - entrance - 2*n_up + 1)")
    ap.add_argument("--plat", type=int, default=-1,
                    help="顶部平台格数 (-1 = 自动)")
    ap.add_argument("--grid", type=float, default=0.4)
    ap.add_argument("--ext", type=float, default=6.0)
    ap.add_argument("--field", type=float, default=4.4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no_spawn_entries", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    gw, field, n = args.grid, args.field, int(round(2 * args.field / args.grid))
    n_up = int(args.n_up)
    ent = int(args.entrance_cells)
    plat = args.plat if args.plat > 0 else (n + 1 - ent - 2 * n_up)
    if ent + 2 * n_up + plat - 1 != n:
        raise SystemExit(f"长度不匹配: ent+2*n_up+plat-1="
                         f"{ent + 2 * n_up + plat - 1} != {n}")
    h = float(args.riser)
    prof = ([0.0] * ent + [h * (k + 1) for k in range(n_up)]
            + [h * n_up] * plat
            + [h * (k + 1) for k in range(n_up - 2, -1, -1)])
    assert len(prof) == n

    H = np.zeros((n, n), dtype=np.float64)   # (i=x, j=y) 通宽
    for k in range(n):
        H[k, :] = prof[k]

    boxes = []
    for i in range(n):
        for j in range(n):
            hh = float(H[i, j])
            if hh <= 0:
                continue
            boxes.append(BOX_TMPL.format(
                name="bx_%03d" % len(boxes),
                x=-field + (i + 0.5) * gw, y=-field + (j + 0.5) * gw,
                z=hh * 0.5, hx=gw * 0.5, hy=gw * 0.5, hz=hh * 0.5))
    spawn_entries = []
    if not args.no_spawn_entries:
        # 起点在入口平道内 (field 内), 确保出生在 hfield 上; 朝向 +x 正对楼梯。
        # 右侧 (下行端) 不再给入口: hfield 只覆盖 ±field 时场外没有地面。
        spawn_entries = [
            dict(x=-field + 1.5 * gw, y=0.0, yaw=0.0, band="wide_entry"),
        ]
    nz = int((H > 0).sum())
    cov = 100.0 * nz / H.size
    meta = dict(kind="boxgrid", grid_w=gw, n=n, x0=float(-field), y0=float(-field),
                x_min=float(-field), y_min=float(-field), height_max=float(H.max()),
                seed=args.seed, n_boxes=len(boxes), heights=H.tolist(),
                riser=h, n_up=n_up, plat=plat, spawn_entries=spawn_entries,
                coverage_pct=float(cov),
                notes="wide single staircase; riser=%.3f; no flat ends" % h)
    print("[wide] %s riser=%.3fm n_up=%d plat=%d -> %d boxes, hmax=%.2fm, 覆盖 %.1f%%"
          % (args.prefix, h, n_up, plat, len(boxes), H.max(), cov))
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
        fh.write(SCENE_TMPL.format(prefix=args.prefix, ext=args.ext, field=field,
                                   nb=len(boxes), riser=h, n_up=n_up, plat=plat,
                                   boxes="\n".join(boxes)))
    print("[wide] 写出", sp)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
