#!/usr/bin/env python
"""把 mjx-go2-training 的训练地形导出成部署仓可直接加载的 MuJoCo 场景。

非侵入: 只在部署仓新增 sim_env/models/go2/ 资产 (场景 XML + hfield PNG + 清单),
不改任何部署代码; 观测器 (scripts/view_policy.py) 只读清单并加载对应场景。

用法:
  python sim/export_deploy_terrains.py                 # 导出全部
  python sim/export_deploy_terrains.py --only big1,hyb2,rubble
输出:
  ../mjx-go2-mujoco2ros2/sim_env/sim_env/models/go2/
      terrains.json          # 清单: 名称/场景路径/场半径/描述
      scene_<name>.xml       # include ../go2.xml + hfield/box geoms
      <name>.png             # hfield (从训练地形复制)
"""
import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEF_DEPLOY = os.path.abspath(os.path.join(ROOT, "..", "mjx-go2-mujoco2ros2"))

SCENE_TMPL = """<mujoco model="go2 terrain {name}">
  <!-- 由 mjx-go2-training/sim/export_deploy_terrains.py 生成, 不要手改。
       来源: models/go2/scene_mjx_terrain_{name}.xml (训练地形)。
       部署版: 机器人 include 换成 ../go2.xml (力矩电机)。
       box geoms: {nb} (group4: 物理碰撞/raycast 可见, 渲染隐藏);
       视觉 hfield: {hfield}; 视觉网格: {mesh_file}; 碰撞盒瓦片: {nc} 个 (group4, alpha0)。

       视觉统一 (2026-10-22, 用户反馈修复):
       1) 所有地形表面 (hfield / 方块 / 台阶) 共用同一个地面材质 + 网格线纹理,
          不再出现"棕色方块 vs 地面色 hfield"两种颜色;
       2) 有 box 的地形额外生成一个 **视觉网格 OBJ** (顶面+侧面) 负责渲染; 碰撞
          方块与倾斜瓦片放 group=4, 观察器默认隐藏 -> 渲染只剩 1 个地形 geom
          (big1: 3168 -> ~65 个可见 geom, 帧率大幅提升);
       3) 统一材质带网格线纹理 + 保留台阶竖面/棱 -> 高度可辨识;
       4) 物理与 raycast 语义不变: hfield group=1 (视觉, ray 可见), 碰撞盒/瓦片
          alpha 与 group 只影响渲染, raycast 仍按 group/alpha 规则命中。 -->
  <include file="go2.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>

  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.32 0.32 0.32" specular="0 0 0"/>
    <rgba haze="0.16 0.26 0.36 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1280" offheight="720"/>
    <quality shadowsize="512" offsamples="1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.42 0.40 0.35" rgb2="0.30 0.29 0.25" markrgb="0.86 0.83 0.72"
      width="256" height="256"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="10 10" reflectance="0.05"/>
    <material name="box_mat" texture="groundplane" texuniform="true" texrepeat="10 10"/>
    <material name="pebble_mat" texture="groundplane" texuniform="true" texrepeat="10 10"/>
    <material name="wall_mat" texture="groundplane" texuniform="true" texrepeat="10 10"/>
    {mesh_asset}
    <hfield name="terrain" file="terrain_{name}.png" size="{sx:.3f} {sy:.3f} {elev:.4f} {base:.4f}"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="hfield" hfield="terrain" material="groundplane"
      pos="0 0 0" friction="1.0 0.02 0.01" contype="0" conaffinity="0" group="1"/>
    <geom name="floor_plane" type="plane" size="0 0 0.05" material="groundplane"
      pos="0 0 -0.002" friction="1.0 0.02 0.01" group="4"/>
    {mesh_geom}
{collision}
{boxes}
  </worldbody>
</mujoco>
"""

DESC = {
    "mix": "hfield 混合: 台阶/方块/波浪/粗糙 + 中心平地",
    "mix_hard": "hfield 混合 (hard 版)",
    "stairs_only": "hfield 楼梯 (8 个金字塔台阶 + 中心平地)",
    "slope_only": "hfield 斜坡",
    "rough_only": "hfield 粗糙地面",
    "boxes_only": "hfield 方块 (无竖面, 历史口径)",
    "hyb1": "3x3 混合: 竖面 boxes/stairs + hfield rough/slope",
    "hyb2": "3x3 混合 (v13/v17/v20/v25b 巩固地形)",
    "hyb3": "3x3 混合 (竖面方块偏多)",
    "bg200": "真竖面方块场 20x20 (0.4m 网格, 高<=0.13m)",
    "stg1": "真竖面楼梯 (0.06/0.11m 单级)",
    "rubble": "碎石: 0.18m 网格 519 个 1-7cm 小竖面 box",
    "big1": "5x5 大混合 (v23/v25 主训练: 碎石/方块/楼梯/粗糙/斜坡)",
    "v7": "boxes 专项 (0.13/0.09/0.06 竖面)",
    "v8": "平衡版混合 (v8)",
    # v28 (2026-10-21): 台阶专项训练场 + 高度阶梯评测场
    "stairs1": "v28 台阶专项 (真竖面长楼梯 0.08/0.15/0.22 + 金字塔, 覆盖 37%)",
    "stairs2": "v28 台阶专项 hard 版 (0.10/0.16/0.22, 覆盖 33%)",
    "bands1": "v28 台阶专项 bands (7 条全长楼梯带, riser 0.06~0.22, 覆盖 86%)",
    "bands_u006": "v28 台阶高度阶梯 riser 0.06m (评测用)",
    "bands_u010": "v28 台阶高度阶梯 riser 0.10m (评测用)",
    "bands_u014": "v28 台阶高度阶梯 riser 0.14m (评测用)",
    "bands_u018": "v28 台阶高度阶梯 riser 0.18m (评测用)",
    "bands_u022": "v28 台阶高度阶梯 riser 0.22m (评测用, 能力边界)",
    # v29: 通宽楼梯 + 入口平道 (无外圈平地), 6 个难度阶段
    "wide_h020": "v29 台阶课程 S0: 通宽楼梯 riser 0.02m (入口平道 1.2m, 总高 0.12m)",
    "wide_h030": "v29 台阶课程 S1: riser 0.03m (总高 0.18m)",
    "wide_h050": "v29 台阶课程 S2: riser 0.05m (总高 0.30m)",
    "wide_h080": "v29 台阶课程 S3: riser 0.08m (总高 0.48m, 逼近 Go2 能力上限)",
    "wide_h110": "v29 极限档: riser 0.11m (总高 0.66m)",
    "wide_h140": "v29 极限档: riser 0.14m (总高 0.84m)",
    # v32: 混合地形课程 (台阶 + 离散箱体 + 碎石, 5 条隔离赛道 0.02-0.10m)
    "mixed1": "v32 混合地形课程: 5 条赛道 (入口平道+台阶+箱体区+碎石区), riser 0.02/0.04/0.06/0.08/0.10m",
}


def parse_training_scene(prefix):
    sp = os.path.join(ROOT, "models", "go2", "scene_mjx_terrain_%s.xml" % prefix)
    if not os.path.isfile(sp):
        return None
    txt = open(sp, encoding="utf-8").read()
    m = re.search(r"<hfield name=.terrain. file=.([^\"]+). size=.([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+).", txt)
    if not m:
        return None
    png = m.group(1)
    sx, sy, elev, base = [float(v) for v in (m.group(2), m.group(3), m.group(4), m.group(5))]
    box_lines = []
    for line in txt.splitlines():
        s = line.strip()
        if re.match(r"<geom name=.(bx|pb)_\d+.", s):
            # v33 视觉统一: 去掉训练场景里的棕色 material/rgba, 统一用 groundplane,
            # 并放进 group=4 (观察器隐藏渲染; 物理碰撞与 raycast 仍可见)
            s2 = re.sub(r'\s(material|rgba)="[^"]*"', "", s)
            s2 = s2.replace("<geom ", '<geom group="4" material="groundplane" ', 1)
            box_lines.append("    " + s2)
    return dict(prefix=prefix, png=os.path.join(ROOT, "models", "go2", "assets", png),
                sx=sx, sy=sy, elev=elev, base=base, boxes=box_lines)


def field_half(prefix, sx):
    jp = os.path.join(ROOT, "models", "go2", "assets", prefix + ".json")
    if os.path.isfile(jp):
        j = json.load(open(jp))
        H = np.asarray(j["heights"])
        gw = float(j["grid_w"])
        x_min = float(j.get("x_min", j["x0"] - 0.5 * H.shape[0] * gw))
        y_min = float(j.get("y_min", j["y0"] - 0.5 * H.shape[1] * gw))
        return max(abs(x_min), abs(x_min + H.shape[0] * gw),
                   abs(y_min), abs(y_min + H.shape[1] * gw))
    return float(sx)


def _quat_from_z(nv) -> tuple:
    """把 +z 轴旋到 nv 的四元数 (w,x,y,z)。"""
    nv = np.asarray(nv, dtype=np.float64)
    nv = nv / max(float(np.linalg.norm(nv)), 1e-12)
    c = float(nv[2])
    if c > 1.0 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if c < -1.0 + 1e-9:
        return (0.0, 1.0, 0.0, 0.0)
    ax = np.cross(np.array([0.0, 0.0, 1.0]), nv)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-12)
    ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
    s = np.sin(ang / 2.0)
    return (float(np.cos(ang / 2.0)), float(ax[0] * s), float(ax[1] * s), float(ax[2] * s))


def hfield_collision_tiles(scene_path: str, cell: float = 0.2,
                           thickness: float = 0.12, max_tiles: int = 2600,
                           overlap: float = 1.05):
    """把训练场景的 hfield 表面采样成**倾斜 box 碰撞瓦片**（视觉仍用 hfield）。

    为什么: CPU MuJoCo 里球-hfield 接触每个足会返回 6 个重复接触点（plane 只有 1 个），
    策略在地形上会抽搐；球-box 只有一个接触点，与 plane 同语义。
    只给"非平坦"格点 + 其邻域生成瓦片，平坦区域交给场景里的 plane 兜底。

    2026-10-21 修正 (T1 实测, /tmp/t1_wedge/report.md):
      * **精确尺寸瓦片会在相邻格之间留下比足球窄的缝**：足型球落在接缝处有
        52%(合成)/79%(真实 hyb2) 的落点出现持续侧面嵌入，整机实测足球被两块
        瓦片的近竖直侧面夹住、压入 2.1-2.5cm、法向力 1.3-1.7kN 完全卡死。
      * 瓦片 xy **放大 1.05 (每边重叠 5mm)** 后嵌入率降到 0% (合成/真实区都是)，
        连续三角面也是 0%。所以默认 overlap=1.05，只改 size，不增几何数量
        (性能零代价)。此前"精确尺寸更好"的注释是错的，已按实测纠正。
      * **瓦片保持 alpha=0 (对 raycast 隐形), 视觉 hfield 用 group=1**:
        T2 实测 (a) 重叠瓦片若 raycast 可见 (alpha=0.01), height_scan 会看到
        "碰撞包络"(重叠产生的凸棱/抬高 1~2cm), 策略 tilt_p95 从 11.5° 升到 17.6°
        (hyb2 24seed 存活 75%->66.7%) - 扫描应看"地形视觉面"而不是碰撞近似面;
        (b) 但 CPU MuJoCo 3.12 的 mj_ray 跳过 alpha==0 的 geom, 旧场景 hfield
        group=3 又与环境组掩码冲突 -> 出生点地面探测 raycast 只能打到 floor_plane
        (uniform 出生四足嵌入瓦片 2-8cm, T1 实测 6/16 假卡死)。
        最终方案: 瓦片 alpha=0 (物理用), 视觉 hfield group=1 (raycast 可见,
        height_scan 与出生探测都打到训练同款高度面, 误差 <1cm)。
    返回 (geoms 行列表, 实际 cell 大小)。
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(scene_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    hf_ids = [g for g in range(int(m.ngeom))
              if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD]
    if not hf_ids:
        return [], cell
    hf_gid = hf_ids[0]
    sx, sy = float(m.hfield_size[0][0]), float(m.hfield_size[0][1])
    ox, oy = float(m.geom_pos[hf_gid][0]), float(m.geom_pos[hf_gid][1])
    # 机器人 geom 在 group 2/3；地面/box 在 group 0 -> 只打 group 0
    mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    gid = np.zeros(1, dtype=np.int32)

    def ground(x, y):
        dist = mujoco.mj_ray(m, d, np.array([x, y, 3.0]), np.array([0.0, 0.0, -1.0]),
                             mask, 1, -1, gid)
        if dist is None or dist >= 3.0:
            return 0.0, -1
        return 3.0 - float(dist), int(gid[0])

    while True:
        n = max(2, int(round(2.0 * sx / cell)))
        n = max(2, int(round(2.0 * sy / cell))) if abs(sx - sy) < 1e-9 else n
        xs = ox - sx + (np.arange(n) + 0.5) * cell
        ys = oy - sy + (np.arange(n) + 0.5) * cell
        H = np.zeros((n, n))
        K = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                h, g = ground(xs[i], ys[j])
                H[i, j] = h
                K[i, j] = (g == hf_gid)
        nz = H > 0.002
        grow = nz.copy()
        grow[1:, :] |= nz[:-1, :]
        grow[:-1, :] |= nz[1:, :]
        grow[:, 1:] |= nz[:, :-1]
        grow[:, :-1] |= nz[:, 1:]
        keep = K & grow
        if int(keep.sum()) <= max_tiles:
            break
        cell *= 1.25
        if cell > 1.0:
            break

    # 倾斜盒瓦片: 顶面 = 局部切平面(过格心高度)。实测比"水平阶梯盒"走得好
    # (hyb2 66.7% vs 58.3%)。注意用**精确尺寸**(不做 1.002 放大), 否则相邻倾斜盒
    # 会互相穿插, 形成窄 V 形缝, 足端球体可能楔死("脚卡到地面里动不了")。
    rows = []
    eps = cell * 0.5
    for i in range(n):
        for j in range(n):
            if not keep[i, j]:
                continue
            h = H[i, j]
            ip, im = min(i + 1, n - 1), max(i - 1, 0)
            jp, jm = min(j + 1, n - 1), max(j - 1, 0)
            gx = (H[ip, j] - H[im, j]) / max((ip - im) * cell, 1e-9)
            gy = (H[i, jp] - H[i, jm]) / max((jp - jm) * cell, 1e-9)
            nv = np.array([-gx, -gy, 1.0])
            q = _quat_from_z(nv)
            nn = nv / np.linalg.norm(nv)
            cx = xs[i] - nn[0] * thickness * 0.5
            cy = ys[j] - nn[1] * thickness * 0.5
            cz = h - nn[2] * thickness * 0.5
            eps_ov = eps * float(overlap)   # 1.05 -> 每边重叠 5mm, 封住接缝
            rows.append(
                '    <geom name="tc_%03d_%03d" type="box" pos="%.4f %.4f %.4f" '
                'size="%.4f %.4f %.4f" quat="%.6f %.6f %.6f %.6f" '
                'rgba="1 1 1 0" contype="1" conaffinity="1" group="4" '
                'friction="1.0 0.02 0.01"/>'
                % (i, j, cx, cy, cz, eps_ov, eps_ov, thickness * 0.5,
                   q[0], q[1], q[2], q[3]))
    return rows, cell



def write_visual_mesh(prefix, out_dir, uv_scale=2.0):
    """从 boxgrid 高度表生成视觉网格 OBJ (顶面 + 暴露竖面), 写到部署 assets/。

    目的: 方块地形原来靠几百上千个 box geom 渲染, 在 WSLg/llvmpipe 下极慢;
    合并成 1 个 mesh 后渲染量降 1-2 个数量级, 同时保留台阶/箱体的竖面。
    碰撞仍是原 box geom (group=4, 渲染隐藏), 物理/raycast 不变。
    """
    jp = os.path.join(ROOT, "models", "go2", "assets", prefix + ".json")
    if not os.path.isfile(jp):
        return None
    j = json.load(open(jp))
    H = np.asarray(j.get("heights", []), dtype=float)
    if H.size == 0 or H.shape[0] < 2 or H.max() <= 1e-6:
        return None
    n = H.shape[0]
    gw = float(j["grid_w"])
    x0 = float(j.get("x_min", j.get("x0", -0.5 * n * gw)))
    y0 = float(j.get("y_min", j.get("y0", -0.5 * n * gw)))
    V, VT, F = [], [], []

    def add_quad(p0, p1, p2, p3, normal, uv_mode="plan"):
        # 保证法线朝 normal: 用 (p1-p0)x(p3-p0) 判方向
        import numpy as _np
        c = _np.cross(_np.subtract(p1, p0), _np.subtract(p3, p0))
        if float(_np.dot(c, normal)) < 0.0:
            p1, p3 = p3, p1
        idx = []
        for pt in (p0, p1, p2, p3):
            if uv_mode == "plan":
                u, v = pt[0] / uv_scale, pt[1] / uv_scale
            else:
                u, v = (pt[0] + pt[1]) / uv_scale, pt[2] / uv_scale
            V.append(pt)
            VT.append((u, v))
            idx.append((len(V), len(VT)))
        a, b, c2, d = idx
        F.append((a, b, c2))
        F.append((a, c2, d))

    z_lo = 0.0
    for i in range(n):
        xa, xb = x0 + i * gw, x0 + (i + 1) * gw
        for jj in range(n):
            hh = float(H[i, jj])
            if hh <= 1e-6:
                continue
            ya, yb = y0 + jj * gw, y0 + (jj + 1) * gw
            add_quad((xa, ya, hh), (xb, ya, hh), (xb, yb, hh), (xa, yb, hh),
                     np.array([0.0, 0.0, 1.0]), "plan")
            hs = {
                "e": float(H[i + 1, jj]) if i + 1 < n else z_lo,
                "w": float(H[i - 1, jj]) if i - 1 >= 0 else z_lo,
                "n": float(H[i, jj + 1]) if jj + 1 < n else z_lo,
                "s": float(H[i, jj - 1]) if jj - 1 >= 0 else z_lo,
            }
            if hh > hs["e"] + 1e-6:   # 东面, 法线 +x
                add_quad((xb, ya, hs["e"]), (xb, yb, hs["e"]),
                         (xb, yb, hh), (xb, ya, hh),
                         np.array([1.0, 0.0, 0.0]), "side")
            if hh > hs["w"] + 1e-6:   # 西面, 法线 -x
                add_quad((xa, ya, hs["w"]), (xa, yb, hs["w"]),
                         (xa, yb, hh), (xa, ya, hh),
                         np.array([-1.0, 0.0, 0.0]), "side")
            if hh > hs["n"] + 1e-6:   # 北面, 法线 +y
                add_quad((xa, yb, hs["n"]), (xb, yb, hs["n"]),
                         (xb, yb, hh), (xa, yb, hh),
                         np.array([0.0, 1.0, 0.0]), "side")
            if hh > hs["s"] + 1e-6:   # 南面, 法线 -y
                add_quad((xa, ya, hs["s"]), (xb, ya, hs["s"]),
                         (xb, ya, hh), (xa, ya, hh),
                         np.array([0.0, -1.0, 0.0]), "side")
    if not F:
        return None
    obj_dir = os.path.join(out_dir, "assets")
    os.makedirs(obj_dir, exist_ok=True)
    fn = "terrain_%s_vis.obj" % prefix
    out = os.path.join(obj_dir, fn)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# visual-only terrain mesh for %s (generated by export_deploy_terrains.py)\n" % prefix)
        for (x, y, z) in V:
            fh.write("v %.5f %.5f %.5f\n" % (x, y, z))
        for (u, v) in VT:
            fh.write("vt %.5f %.5f\n" % (u, v))
        for (a, b, c) in F:
            fh.write("f %d/%d %d/%d %d/%d\n" % (a[0], a[1], b[0], b[1], c[0], c[1]))
    return fn



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy_root", default=DEF_DEPLOY)
    ap.add_argument("--only", default="", help="逗号分隔的地形名 (默认全部)")
    ap.add_argument("--out_subdir", default="sim_env/models/go2")
    ap.add_argument("--collision_overlap", type=float, default=1.05,
                    help="倾斜碰撞瓦片 xy 重叠倍数 (1.05 = 每边 ~5mm; T1 实测 "
                         "1.00 时 79%% 接缝嵌入 -> 1.05 时 0%%。只改 size, 不增几何)")
    args = ap.parse_args(argv)
    out_dir = os.path.join(os.path.abspath(args.deploy_root), args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    if not names:
        names = [n for n in DESC if os.path.isfile(
            os.path.join(ROOT, "models", "go2", "scene_mjx_terrain_%s.xml" % n))]
        pref = ["hyb2", "hyb1", "hyb3", "big1", "rubble", "bg200", "stg1",
                "mix", "stairs_only", "slope_only", "rough_only", "boxes_only"]
        names = [n for n in pref if n in names] + [n for n in names if n not in pref]
    entries = [dict(name="flat", scene="sim_env/models/go2/scene_flat.xml", kind="flat",
                    field_half=0.0, hfield_half=0.0,
                    desc="平地 (原始部署场景)")]
    for prefix in names:
        info = parse_training_scene(prefix)
        if info is None:
            print("[skip] 找不到训练场景:", prefix)
            continue
        png_out = os.path.join(out_dir, "assets", "terrain_" + prefix + ".png")
        os.makedirs(os.path.dirname(png_out), exist_ok=True)
        shutil.copyfile(info["png"], png_out)
        scene_name = "scene_terrain_%s.xml" % prefix
        train_scene = os.path.join(ROOT, "models", "go2",
                                   "scene_mjx_terrain_%s.xml" % prefix)
        tiles, cell_used = hfield_collision_tiles(
            train_scene, overlap=float(args.collision_overlap))
        mesh_file = write_visual_mesh(prefix, out_dir)
        mesh_asset = ("<mesh name=\"terrain_vis\" file=\"%s\" smoothnormal=\"false\"/>" % mesh_file
                      if mesh_file else "")
        mesh_geom = ("<geom name=\"terrain_vis\" type=\"mesh\" mesh=\"terrain_vis\" "
                     "material=\"groundplane\" pos=\"0 0 0\" contype=\"0\" "
                     "conaffinity=\"0\" group=\"1\"/>" if mesh_file else "")
        scene = SCENE_TMPL.format(
            name=prefix, nb=len(info["boxes"]), hfield="terrain_%s.png" % prefix,
            sx=info["sx"], sy=info["sy"], elev=info["elev"], base=info["base"],
            nc=len(tiles), collision="\n".join(tiles),
            ov=float(args.collision_overlap), boxes="\n".join(info["boxes"]),
            mesh_file=(mesh_file or "(none)"), mesh_asset=mesh_asset,
            mesh_geom=mesh_geom)
        with open(os.path.join(out_dir, scene_name), "w", encoding="utf-8") as fh:
            fh.write(scene)
        fh_half = field_half(prefix, info["sx"])
        entries.append(dict(
            name=prefix, scene="%s/%s" % (args.out_subdir, scene_name),
            kind=("hybrid" if info["boxes"] else "hfield"),
            field_half=round(float(fh_half), 3),
            hfield_half=round(float(info["sx"]), 3),
            n_boxes=len(info["boxes"]),
            n_collision_boxes=len(tiles),
            collision_cell=round(float(cell_used), 3),
            collision_overlap=float(args.collision_overlap),
            collision_raycast_visible=True,
            desc=DESC.get(prefix, prefix)))
        print("[export] %-12s -> %s (boxes=%d, 碰撞瓦片=%d@%.2fm, "
              "field_half=%.2fm, hfield_half=%.2fm)"
              % (prefix, scene_name, len(info["boxes"]), len(tiles), cell_used,
                 fh_half, info["sx"]))
    manifest = dict(
        _note="由 mjx-go2-training/sim/export_deploy_terrains.py 生成; 观测器 --terrain 读这个清单",
        terrains=entries)
    with open(os.path.join(out_dir, "terrains.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print("[export] 清单 ->", os.path.join(out_dir, "terrains.json"),
          "(%d 个地形)" % len(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())