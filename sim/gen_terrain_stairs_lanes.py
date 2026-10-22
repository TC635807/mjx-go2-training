#!/usr/bin/env python
"""生成"多难度楼梯赛道"地形 (v29 混合课程, 仿 legged_gym 一张地形多难度瓦片)。

动机:
* v29 单阶段 warm-start 课程 (0.02->0.03->0.05) 有"遗忘"现象: 训 0.05 后 0.03 的
  成功率回落 (上 1 级 93.6% -> 89.5%, 上 2 级 17.6% -> 6.5%)。
* legged_gym/IsaacLab/mjlab 是**一张地形里放 10 行 x 20 列不同难度的瓦片**, 每个 env
  随机/按课程分配到某块 -> 一次训练同时见到所有难度, 不会遗忘。
* 我们用"多条互相隔离的楼梯赛道"复刻这个语义: 4 条 2m 宽赛道, 每条 riser 不同
  (0.03/0.05/0.08/0.11), 赛道之间用 0.5m 高墙隔开 (机器人不能横穿换道);
  每条赛道入口有 3 格平道 (助跑), 入口写入 spawn_entries, 每个 env 均匀分到一条道。
  训练时 2560 个 env 同时覆盖 4 个难度 -> 不会遗忘, 且比单阶段更省训练轮次。

布局 (22x22 格, 0.4m): 4 条赛道 x 5 格 (2.0m 宽) + 3 道隔墙 x 1 格 = 22 列 (y)。
  剖面 (沿 x, 22 格): 3 格入口平道 -> 上行 6 级 -> 平台 8 格 -> 下行 5 级。
用法:
  python sim/gen_terrain_stairs_lanes.py --prefix lanes1 --risers 0.03,0.05,0.08,0.11
输出: models/go2/assets/<prefix>.png/json + models/go2/scene_mjx_terrain_<prefix>.xml
"""
import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")

SCENE_TMPL = '''<mujoco model="go2 stairs lanes boxgeom scene">
  <!-- 由 sim/gen_terrain_stairs_lanes.py 生成, 不要手改。
       {nlanes} 条隔离楼梯赛道 (真竖面 box), 底面 hfield 全 0。
       赛道 riser: {risers}; 场 ±{field:.1f}m; 隔墙高 {wall_h:.2f}m。 -->
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
    <material name="wall_mat" rgba="0.45 0.42 0.50 1"/>
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
            'size="{hx:.4f} {hy:.4f} {hz:.4f}" material="{mat}" '
            'contype="1" conaffinity="1" friction="0.9 0.02 0.01"/>')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="lanes1")
    ap.add_argument("--risers", default="0.03,0.05,0.08,0.11",
                    help="每条赛道的单级 riser (逗号分隔)")
    ap.add_argument("--lane-width", type=int, default=5, help="赛道最小宽度 (格, 实际按剩余列均分)")
    ap.add_argument("--n-up", type=int, default=6)
    ap.add_argument("--entrance", type=int, default=3)
    ap.add_argument("--plat", type=int, default=-1)
    ap.add_argument("--wall-h", type=float, default=0.5)
    ap.add_argument("--grid", type=float, default=0.4)
    ap.add_argument("--ext", type=float, default=4.4)
    ap.add_argument("--field", type=float, default=4.4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    gw, field = args.grid, args.field
    n = int(round(2 * field / gw))
    risers = [float(x) for x in args.risers.split(",") if x.strip()]
    min_lw = int(args.lane_width)
    walls = len(risers) - 1
    if len(risers) * min_lw + walls > n:
        raise SystemExit(f"赛道太多/太窄: {len(risers)}*{min_lw}+{walls} > {n}")
    # 把 (n - walls) 列尽量平均分给各赛道 (每条 >= min_lw, 余数给前面几条)
    free = n - walls
    base = free // len(risers)
    rem = free % len(risers)
    widths = [base + (1 if k < rem else 0) for k in range(len(risers))]
    if min(widths) < min_lw:
        raise SystemExit(f"赛道太窄: {widths}")
    lane_cols = []          # (j0, width)
    wall_cols = []
    j = 0
    for k in range(len(risers)):
        lane_cols.append((j, widths[k]))
        j += widths[k]
        if k < len(risers) - 1:
            wall_cols.append(j)
            j += 1
    plat = args.plat if args.plat > 0 else (n + 1 - args.entrance - 2 * args.n_up)
    if args.entrance + 2 * args.n_up + plat - 1 != n:
        raise SystemExit(f"剖面长度不匹配: {args.entrance}+2*{args.n_up}+{plat}-1 != {n}")
    H = np.zeros((n, n), dtype=np.float64)             # (i=x, j=y)
    is_wall = np.zeros((n, n), dtype=bool)
    spawn_entries = []
    for k, (j0, w) in enumerate(lane_cols):
        h = risers[k]
        prof = ([0.0] * args.entrance
                + [h * (m + 1) for m in range(args.n_up)]
                + [h * args.n_up] * plat
                + [h * (m + 1) for m in range(args.n_up - 2, -1, -1)])
        for i in range(n):
            H[i, j0:j0 + w] = prof[i]
        yc = -field + (j0 + w / 2.0) * gw
        spawn_entries.append(dict(x=-field + 1.5 * gw, y=float(yc), yaw=0.0,
                                  band="lane%d_r%.3f" % (k, h)))
    for jw in wall_cols:
        for i in range(n):
            H[i, jw] = float(args.wall_h)
            is_wall[i, jw] = True
    # 生成 boxes
    boxes = []
    for i in range(n):
        for j in range(n):
            hh = float(H[i, j])
            if hh <= 0:
                continue
            mat = "wall_mat" if is_wall[i, j] else "box_mat"
            boxes.append(BOX_TMPL.format(
                name="bx_%03d" % len(boxes),
                x=-field + (i + 0.5) * gw, y=-field + (j + 0.5) * gw,
                z=hh * 0.5, hx=gw * 0.5, hy=gw * 0.5, hz=hh * 0.5, mat=mat))
    meta = dict(kind="boxgrid", grid_w=gw, n=n, x0=float(-field), y0=float(-field),
                x_min=float(-field), y_min=float(-field), height_max=float(H.max()),
                seed=args.seed, n_boxes=len(boxes), heights=H.tolist(),
                risers=risers, lane_cols=lane_cols, wall_cols=wall_cols,
                wall_h=float(args.wall_h), spawn_entries=spawn_entries,
                notes="multi-difficulty stair lanes (v29 mixed curriculum)")
    print("[lanes] %s risers=%s lanes=%s walls=%s -> %d boxes, hmax=%.2f"
          % (args.prefix, risers, lane_cols, wall_cols, len(boxes), H.max()))
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
                                   nlanes=len(risers), risers=risers,
                                   wall_h=args.wall_h, boxes="\n".join(boxes)))
    print("[lanes] 写出", sp)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
