#!/usr/bin/env python
"""混合地形课程 (v32): 一条地形里 5 个难度档的"台阶 + 离散箱体 + 碎石"混合赛道。

动机 (用户要求): v31 的大网络+纯台阶课程把 0.03-0.05m 台阶做到上 2 级 70-86%,
但训练完再用单一 hyb2 巩固会侵蚀台阶与部署稳健性。正确做法是**混合地形课程**:
每条赛道是"入口平道 -> 上行台阶 -> 平台 -> 下行 -> 离散箱体区 -> 碎石区 -> 出口"
的混合障碍, 不同赛道难度递增; per-env 课程按位移升降级, 一次训练同时学
爬台阶 + 跨箱体 + 碎石鲁棒性。

布局: 26x26 格 (0.4m, 场 ±5.2m), 5 条互隔赛道 (宽 5/5/4/4/4 格) + 4 道 0.45m 隔墙。
每条赛道剖面 (26 格):
  0-1   入口平道 (0.8m)
  2-4   上行台阶 h, 2h, 3h
  5-6   平台 3h
  7-8   下行 2h, h
  9-13  离散箱体区 (高度 1.0h/0.5h/1.5h/0.8h/1.2h, 打乱)
  14-17 碎石区 (0.01-0.03m 小起伏)
  18-25 出口平道
难度档: h = 0.02/0.04/0.06/0.08/0.10 m; spawn_entries 在每条赛道入口 (x=-4.6, yaw=0)。
用法: python sim/gen_terrain_mixed_curriculum.py --prefix mixed1
输出: models/go2/assets/mixed1.png/json + models/go2/scene_mjx_terrain_mixed1.xml
"""
import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")

SCENE_TMPL = '''<mujoco model="go2 mixed curriculum boxgeom scene">
  <!-- 由 sim/gen_terrain_mixed_curriculum.py 生成, 不要手改。
       {nboxes} 个 box geom: 5 条混合障碍赛道 (台阶+箱体+碎石), 难度 riser {risers}。
       场 ±{field:.1f}m, hfield 全 0, 隔墙高 {wall_h:.2f}m。 -->
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


def lane_profile(h, rng):
    """返回长度 26 的单条赛道高度剖面。"""
    prof = [0.0, 0.0]
    prof += [h, 2 * h, 3 * h]
    prof += [3 * h, 3 * h]
    prof += [2 * h, h]
    box = [1.0, 0.5, 1.5, 0.8, 1.2]
    rng.shuffle(box)
    prof += [v * h for v in box]
    prof += [0.01, 0.03, 0.02, 0.015]
    prof += [0.0] * 8
    assert len(prof) == 26, len(prof)
    return prof


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mixed1")
    ap.add_argument("--risers", default="0.02,0.04,0.06,0.08,0.10")
    ap.add_argument("--wall-h", type=float, default=0.45)
    ap.add_argument("--grid", type=float, default=0.4)
    ap.add_argument("--field", type=float, default=5.2)
    ap.add_argument("--ext", type=float, default=5.2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    gw, field = args.grid, args.field
    n = int(round(2 * field / gw))
    risers = [float(x) for x in args.risers.split(",") if x.strip()]
    lanes = len(risers)
    walls = lanes - 1
    free = n - walls
    base = free // lanes
    rem = free % lanes
    widths = [base + (1 if k < rem else 0) for k in range(lanes)]
    assert min(widths) >= 4, widths
    lane_cols, wall_cols = [], []
    j = 0
    rng = np.random.default_rng(args.seed)
    for k in range(lanes):
        lane_cols.append((j, widths[k]))
        j += widths[k]
        if k < lanes - 1:
            wall_cols.append(j)
            j += 1
    H = np.zeros((n, n), dtype=np.float64)
    is_wall = np.zeros((n, n), dtype=bool)
    spawn_entries = []
    for k, (j0, w) in enumerate(lane_cols):
        prof = lane_profile(risers[k], rng)
        for i in range(n):
            H[i, j0:j0 + w] = prof[i]
        yc = -field + (j0 + w / 2.0) * gw
        spawn_entries.append(dict(x=-field + 1.5 * gw, y=float(yc), yaw=0.0,
                                  band="mixed%d_r%.3f" % (k, risers[k])))
    for jw in wall_cols:
        H[:, jw] = float(args.wall_h)
        is_wall[:, jw] = True
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
                notes="mixed obstacles: stairs + discrete boxes + rubble; per-env curriculum")
    print("[mixed] %s risers=%s lanes=%s walls=%s -> %d boxes, hmax=%.2f"
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
                                   nboxes=len(boxes), risers=risers,
                                   wall_h=args.wall_h, boxes="\n".join(boxes)))
    print("[mixed] 写出", sp)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
