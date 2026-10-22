#!/usr/bin/env python
"""把训练/评测用到的地形资产画成俯视高度图 (hfield + box geom 合成), 便于直观检查。

用法:
  python sim/plot_terrain_maps.py
输出:
  docs/img/go2-terrain-maps.png      # 6 个代表性地形的高度图网格
  docs/img/go2-big1-terrain-3d.png   # big1 (v23/v25 训练地形) 3D 表面
"""
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ASSETS = os.path.join(ROOT, "models", "go2", "assets")
SCENES = os.path.join(ROOT, "models", "go2")


def load_scene_meta(prefix):
    sp = os.path.join(SCENES, "scene_mjx_terrain_%s.xml" % prefix)
    if not os.path.isfile(sp):
        return None
    txt = open(sp, encoding="utf-8").read()
    m = re.search(r'hfield name="terrain" file="([^"]+)" size="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"', txt)
    if not m:
        return None
    return dict(png=m.group(1), half_x=float(m.group(2)), half_y=float(m.group(3)),
                elev=float(m.group(4)))


def load_terrain(prefix):
    """用 MuJoCo 引擎读 hfield_data (权威数据, 16/8-bit PNG 都正确) + boxgrid JSON。

    约定 (envs/go2_walk.py::terrain_height 与生成器一致):
      * hfield table[i_row, i_col]: i_row <-> y, i_col <-> x;
        高度 = table * hfield_size[2] (elev);
      * boxgrid heights[i, j]: i <-> +x, j <-> +y; 世界 x = x0+(i+0.5)*gw。
    """
    import mujoco
    sp = os.path.join(SCENES, "scene_mjx_terrain_%s.xml" % prefix)
    if not os.path.isfile(sp):
        return None
    m = mujoco.MjModel.from_xml_path(sp)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    hid = int(m.geom_dataid[gid])
    nrow = int(m.hfield_nrow[hid]); ncol = int(m.hfield_ncol[hid])
    sx, sy, elev, base = [float(v) for v in m.hfield_size[hid]]
    adr = int(m.hfield_adr[hid])
    table = np.asarray(m.hfield_data[adr:adr + nrow * ncol], dtype=np.float64).reshape(nrow, ncol)
    h = table * elev                      # (nrow=y, ncol=x)
    xs = np.linspace(-sx, sx, ncol)
    ys = np.linspace(-sy, sy, nrow)
    box = None
    jp = os.path.join(ASSETS, prefix + ".json")
    if os.path.isfile(jp):
        j = json.load(open(jp))
        H = np.asarray(j["heights"], dtype=np.float64)     # (i=x, j=y)
        gw = float(j["grid_w"])
        n = H.shape[0]
        # 注意: JSON 的 x0/y0 是**方格场中心参数**, 方格中心 = x0+(i+.5)gw-0.5*n*gw;
        # 直接用 x_min (= xs[0]-gw/2) 更稳 (两种生成器都写了这个字段)。
        x_min = float(j.get("x_min", j["x0"] - 0.5 * n * gw + 0.0))
        y_min = float(j.get("y_min", j["y0"] - 0.5 * n * gw + 0.0))
        bx = x_min + (np.arange(n) + 0.5) * gw
        by = y_min + (np.arange(n) + 0.5) * gw
        box = (bx, by, H)
    return dict(h=h, xs=xs, ys=ys, box=box, prefix=prefix)


def display_grid(t, n=220):
    """在 [-half, half]^2 上合成 hfield + boxgrid 高度图, 返回 (x, y, Z[y, x])。"""
    h = t["h"]
    sx = float(t["xs"][-1]); sy = float(t["ys"][-1])
    half = max(sx, sy)
    gx = np.linspace(-half, half, n)
    gy = np.linspace(-half, half, n)
    ix = np.clip(np.searchsorted(t["xs"], gx), 0, len(t["xs"]) - 1)
    iy = np.clip(np.searchsorted(t["ys"], gy), 0, len(t["ys"]) - 1)
    out = h[np.ix_(iy, ix)]                 # rows=y, cols=x
    if t["box"] is not None:
        bx, by, H = t["box"]
        jx = np.clip(np.searchsorted(bx, gx), 0, len(bx) - 1)
        jy = np.clip(np.searchsorted(by, gy), 0, len(by) - 1)
        # 只取 box 场范围内的格; 范围外是平地 (不能把边缘列 clamp 成"长条")
        inside = ((gy[:, None] >= by[0] - 0.2) & (gy[:, None] <= by[-1] + 0.2)
                  & (gx[None, :] >= bx[0] - 0.2) & (gx[None, :] <= bx[-1] + 0.2))
        out = np.maximum(out, np.where(inside, H[np.ix_(jx, jy)].T, 0.0))
    return gx, gy, out


PANELS = [
    ("big1", "big1: 5x5 multi-patch (v23/v25 training)"),
    ("hyb2", "hyb2: 3x3 hybrid, vertical faces + hfield"),
    ("rubble", "rubble: 0.18m grid, 519 small boxes (eval)"),
    ("mix", "mix: hfield blend (v5a)"),
    ("stairs_only", "stairs_only: hfield stairs"),
    ("rough_only", "rough_only: hfield rough"),
    ("boxes_only", "boxes_only: hfield boxes"),
    ("bg200", "bg200: vertical-face boxes 20x20"),
    ("stg1", "stg1: vertical-face stairs"),
]

def main():
    terrains = {}
    for prefix, _ in PANELS:
        t = load_terrain(prefix)
        if t is not None:
            terrains[prefix] = t
    print("[plot] 找到地形:", sorted(terrains))
    ncol = 3
    nrow = int(np.ceil(len(PANELS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow))
    axes = np.atleast_2d(axes)
    for k, (prefix, title) in enumerate(PANELS):
        ax = axes[k // ncol, k % ncol]
        t = terrains.get(prefix)
        if t is None:
            ax.axis("off")
            continue
        gx, gy, Z = display_grid(t)
        im = ax.imshow(Z, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]],
                       cmap="viridis", vmin=0.0,
                       vmax=max(0.05, float(np.percentile(Z, 99.5)) * 1.05),
                       interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("height (m)", fontsize=7)
    fig.suptitle("Go2 training / evaluation terrains (top view, hfield + box geoms merged)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(ROOT, "docs", "img", "go2-terrain-maps.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print("[plot] 写出", out)

    # big1 3D
    t = terrains.get("big1")
    if t is not None:
        gx, gy, Z = display_grid(t, n=180)
        X, Y = np.meshgrid(gx, gy)
        fig3 = plt.figure(figsize=(9, 6.4))
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
        ax3.set_title("big1: 5x5 multi-patch training terrain (pebble / boxes / stairs / hfield)",
                      fontsize=10)
        ax3.set_xlabel("x (m)", fontsize=8)
        ax3.set_ylabel("y (m)", fontsize=8)
        ax3.set_zlabel("height (m)", fontsize=8)
        ax3.set_zlim(0, max(0.05, float(Z.max()) * 1.15))
        ax3.view_init(elev=35, azim=-60)
        out3 = os.path.join(ROOT, "docs", "img", "go2-big1-terrain-3d.png")
        fig3.tight_layout()
        fig3.savefig(out3, dpi=140)
        print("[plot] 写出", out3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())