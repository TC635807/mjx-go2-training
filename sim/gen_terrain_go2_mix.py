#!/usr/bin/env python3
"""混合地形生成器 v2 —— 把 mjlab / IsaacLab 的 ROUGH_TERRAINS 配方搬进我们的 hfield 流程。

为什么需要它 (见 docs/go2-terrain-survey.md):
  现有 terrain.png 只有 "平滑起伏 + 6 块台地 + 1 条楼梯", 且**整张图只有一种风格**。
  mjlab (MuJoCo Warp 训练框架, Unitree 官方 unitree_rl_mjlab 的 Go2 任务在用) 的
  ROUGH_TERRAINS_CFG 是 "按 patch 拼多种子地形 + 按难度分档", IsaacLab 的
  ROUGH_TERRAINS_CFG 也是同一套思路 (pyramid stairs / inverted stairs / boxes /
  random rough / slopes, 各占固定比例)。

本脚本把同一套地形类型用 numpy 重新实现, 输出**仍然是一张 hfield PNG**, 因此:
  * 单张共享地形 (warp 后端的限制) 也能让策略在一个 episode 里见到 8-9 种子地形;
  * 随机出生 (±5m) 自然覆盖所有 patch, 不需要改环境代码;
  * 分辨率可调 (n=128/192/256), 用来解决 9.4cm 格距 vs 2.2cm 足端的 CPU/warp
    hfield 接触分歧 (格距 4.7cm @ n=256)。

地形类型 (每个 patch 一种):
  flat            平地 (原点所在, 保证 reset 一致)
  stairs_up_*     金字塔台阶 (向上), 难度分档: 台阶 0.05 / 0.09 m
  slope_up_*      金字塔斜坡 (向上), 坡度分档
  rough_*         随机起伏 (高斯滤波白噪声), 幅度分档
  wave            多频正弦波
  boxes           随机方块 (网格化台阶块)
  perlin          多倍频 Perlin 风格噪声 (用高斯滤波金字塔近似)
  obstacles       离散障碍块 (随机矩形凸台)

与 mjlab/IsaacLab 的差异 (诚实标注):
  * 我们只能输出高度场 (hfield), 不能输出 trimesh, 所以 "倒金字塔/负高度" 类地形
    用不了的, 统一改成 h>=0 的 "上坡-平台-下坡" 版本;
  * 所有 patch 边缘都做线性过渡到 h=0, 避免 patch 之间出现垂直墙 (会把机器人绊倒);
  * 难度是按 patch 位置分档 (行=难度), 不是按 episode 课程学习 —— 我们的训练环境
    没有 per-env 地形, 只能靠随机出生覆盖。

用法:
  python sim/gen_terrain_go2_mix.py --n 256            # 生成默认混合地形
  python sim/gen_terrain_go2_mix.py --n 192 --preset stairs
  python sim/gen_terrain_go2_mix.py --dry_run          # 只打印统计
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "..")
ASSETS = os.path.join(ROOT, "models", "go2", "assets")
SCENE = os.path.join(ROOT, "models", "go2", "scene_mjx_terrain_mix.xml")


# --------------------------------------------------------------------- 基础工具
def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _border_fade(x: np.ndarray, y: np.ndarray, half: float, border: float) -> np.ndarray:
    """到 patch 边缘的距离 -> 0..1 的权重 (边缘 0, 内部 1)。"""
    d = half - np.maximum(np.abs(x), np.abs(y))
    return _smoothstep(d / max(border, 1e-9))


def _box_metric(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(x), np.abs(y))


# --------------------------------------------------------------------- 地形函数
# 每个函数签名: fn(x, y, rng) -> h (>=0), x/y 是 patch 局部坐标 (米, 中心 0)
def t_flat(x, y, rng):
    return np.zeros_like(x)


def t_stairs_up(step_h, step_w, platform_w, half, max_h=0.22):
    """金字塔台阶: 中心平台最高, 向外逐级下降。台阶数按 max_h 自动截断。"""
    def fn(x, y, rng):
        r = _box_metric(x, y)
        r0 = platform_w / 2.0
        n_fit = max(1, int(np.floor((half - r0) / step_w)))
        n_cap = max(1, int(np.floor(max_h / step_h)))
        n = min(n_fit, n_cap)
        idx = np.clip(np.floor((r - r0) / step_w) + 1.0, 0.0, float(n))
        # 中心平台 = step_h*(n+1); 用 max_h 封顶, 保证台阶总高不超过配置值
        return np.minimum(step_h * (n - idx + 1.0), max_h)
    return fn


def t_slope_up(max_h, platform_w, half):
    def fn(x, y, rng):
        r = _box_metric(x, y)
        r0 = platform_w / 2.0
        ramp = np.clip((r - r0) / max(half - r0, 1e-9), 0.0, 1.0)
        return max_h * (1.0 - ramp)
    return fn


def t_rough(amp, corr_cells):
    def fn(x, y, rng):
        # 白噪声 -> 高斯滤波 (相关长度 corr_cells) -> 非负
        w = rng.standard_normal(x.shape)
        w = gaussian_filter(w, sigma=corr_cells, mode="nearest")
        w = w / (np.abs(w).max() + 1e-9)
        return amp * (0.5 * (w + 1.0))          # [0, amp]
    return fn


def t_wave(amp, num_waves):
    def fn(x, y, rng):
        h = np.zeros_like(x)
        for k in range(num_waves):
            fx = rng.uniform(0.4, 1.8, 2)
            ph = rng.uniform(0, 2 * np.pi, 2)
            h += np.sin(fx[0] * x + ph[0]) * np.cos(fx[1] * y + ph[1])
        h = h / max(len(range(num_waves)), 1) / 2.0
        return amp * (0.5 * (h + 1.0))
    return fn


def t_boxes(grid_w, height, half):
    def fn(x, y, rng):
        gx = np.floor((x + half) / grid_w).astype(int)
        gy = np.floor((y + half) / grid_w).astype(int)
        hmap = rng.uniform(0.0, height, (gx.max() + 2, gy.max() + 2))
        h = hmap[np.clip(gx, 0, hmap.shape[0] - 1), np.clip(gy, 0, hmap.shape[1] - 1)]
        # 只保留 ~50% 的方块, 其余为 0 (更像离散踏脚石)
        keep = rng.uniform(size=hmap.shape) > 0.5
        hmap = hmap * keep
        h = hmap[np.clip(gx, 0, hmap.shape[0] - 1), np.clip(gy, 0, hmap.shape[1] - 1)]
        return h
    return fn


def t_perlin(max_h, octaves):
    def fn(x, y, rng):
        h = np.zeros_like(x)
        amp, sigma, norm = 1.0, 8.0, 0.0
        for _ in range(octaves):
            w = rng.standard_normal(x.shape)
            w = gaussian_filter(w, sigma=sigma, mode="nearest")
            w = w / (np.abs(w).max() + 1e-9)
            h += amp * w
            norm += amp
            amp *= 0.5
            sigma *= 0.5
        h = h / max(norm, 1e-9)
        return max_h * (0.5 * (h + 1.0))
    return fn


def t_obstacles(max_h, n_obstacles, half):
    def fn(x, y, rng):
        h = np.zeros_like(x)
        for _ in range(n_obstacles):
            cx, cy = rng.uniform(-half * 0.85, half * 0.85, 2)
            sx, sy = rng.uniform(0.15, 0.45, 2)
            hh = rng.uniform(0.3, 1.0) * max_h
            m = (np.abs(x - cx) <= sx) & (np.abs(y - cy) <= sy)
            h = np.maximum(h, np.where(m, hh, 0.0))
        return h
    return fn


# --------------------------------------------------------------------- patch 拼装
def build(n: int, ext: float, preset: str, seed: int,
          hardness: float = 1.0) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    axis = np.linspace(-ext, ext, n, endpoint=True)
    XX, YY = np.meshgrid(axis, axis)          # (n, n), [row=y, col=x]
    H = np.zeros((n, n), dtype=np.float64)

    P = 3                                      # 3x3 patches
    patch = 2 * ext / P                        # 4.0 m @ ext=6
    half = patch / 2.0
    border = 0.45

    hr = max(float(hardness), 0.2)

    # 每个 patch 的地形工厂: (rows=难度, cols=类型); hr 是全局难度倍数
    def make(kind, diff, r0):
        if kind == "flat":
            return t_flat
        if kind == "stairs":
            return t_stairs_up(step_h=(0.05 + 0.03 * diff) * hr,
                               step_w=max(0.24, 0.36 - 0.06 * diff),
                               platform_w=0.8, half=half,
                               max_h=min(0.20 * hr, 0.24))
        if kind == "slope":
            return t_slope_up(max_h=(0.10 + 0.12 * diff) * hr, platform_w=0.8, half=half)
        if kind == "rough":
            return t_rough(amp=(0.04 + 0.06 * diff) * hr, corr_cells=2.0)
        if kind == "wave":
            return t_wave(amp=(0.10 + 0.06 * diff) * hr, num_waves=4)
        if kind == "boxes":
            return t_boxes(grid_w=0.4, height=(0.05 + 0.07 * diff) * hr, half=half)
        if kind == "perlin":
            return t_perlin(max_h=(0.08 + 0.10 * diff) * hr, octaves=4)
        if kind == "obstacles":
            return t_obstacles(max_h=(0.06 + 0.10 * diff) * hr, n_obstacles=25, half=half)
        raise ValueError(kind)

    _SINGLE = {"stairs_only": "stairs", "slope_only": "slope", "rough_only": "rough",
               "boxes_only": "boxes", "wave_only": "wave", "perlin_only": "perlin",
               "obstacles_only": "obstacles", "flat_only": "flat"}
    if preset == "mix":
        layout = [
            ["stairs", "rough",  "slope"],
            ["boxes",  "flat",   "wave"],
            ["stairs", "rough",  "perlin"],
        ]
        diff_of_row = [0.0, None, 1.0]
    elif preset == "mix_hard":
        layout = [
            ["stairs", "rough",  "slope"],
            ["boxes",  "flat",   "wave"],
            ["stairs", "rough",  "perlin"],
        ]
        diff_of_row = [0.5, None, 1.0]
        hr = max(hr, 1.5)
    elif preset in _SINGLE:
        k = _SINGLE[preset]
        layout = [[k, k, k] for _ in range(3)]
        diff_of_row = [0.0, 0.5, 1.0]
    elif preset == "v5":
        # v5 课程第二段: 台阶/斜坡加权 (台阶 4, 斜坡 3, 起伏 1, 平地 1),
        # 难度按行递增; 全局硬度 1.5 -> 台阶 11~15cm, 斜坡到 ~25deg.
        layout = [
            ["stairs", "slope", "stairs"],
            ["slope",  "flat",  "stairs"],
            ["slope",  "rough", "stairs"],
        ]
        diff_of_row = [0.3, 0.6, 1.0]
        hr = max(hr, 1.5)
    elif preset == "v7":
        # v7 (2026-10-19): boxes/stairs 专项。动机来自摔倒诊断 (sim/diag_falls.sh):
        #   * boxes_only 384 env: 44.5% 摔倒; 摔倒处地形高度中位 0.085 m、局部
        #     坡度中位 0.24 -> 是**踩在方块棱上**被绊, 不是翻车 (tilt 中位 10°);
        #   * 训练用的 mix 地形里 boxes 只占 1/9 面积、最高仅 0.085 m,
        #     而评测 boxes_only 最高 0.113 m -> 训练覆盖不足。
        # 本 preset: boxes 4/9 + stairs 2/9 (最高 0.24 m) + rough/slope/flat 各 1/9,
        # 全局难度 1.25 -> boxes 0.089/0.115/0.15 m, 台阶 0.074/0.085/0.10 m。
        layout = [
            ["boxes",  "stairs", "boxes"],
            ["rough",  "flat",   "slope"],
            ["boxes",  "stairs", "boxes"],
        ]
        diff_of_row = [0.3, 0.6, 1.0]
        hr = max(hr, 1.25)
    elif preset == "v8":
        # v8 (2026-10-19): **平衡版** —— v7 (boxes 4/9) 实测 boxes +7pp 但
        # mix/stairs/slope 掉 2~6pp (遗忘)。v8 把 boxes 降到 3/9, 保留
        # stairs 2/9 + rough/slope/wave/flat 各 1/9, 难度 1.1 (贴近评测难度):
        #   boxes 0.078/0.101/0.132 m (评测最高 0.113)
        #   stairs 总高 0.22 m (评测 0.20), 单步 0.061/0.082/0.099
        layout = [
            ["boxes", "stairs", "rough"],
            ["boxes", "flat",   "slope"],
            ["boxes", "stairs", "wave"],
        ]
        diff_of_row = [0.3, 0.6, 1.0]
        hr = max(hr, 1.1)
    elif preset == "stairs":          # 旧名保留: 3x3 台阶 + 中心平地
        layout = [["stairs", "stairs", "stairs"],
                  ["stairs", "flat",   "stairs"],
                  ["stairs", "stairs", "stairs"]]
        diff_of_row = [0.0, None, 1.0]
    elif preset == "rough":
        layout = [["rough", "wave", "perlin"],
                  ["boxes", "flat", "obstacles"],
                  ["rough", "wave", "perlin"]]
        diff_of_row = [0.0, None, 1.0]
    else:
        raise ValueError(preset)

    layout_meta = []
    for r in range(P):
        for c in range(P):
            kind = layout[r][c]
            diff = 0.5 if diff_of_row[r] is None else diff_of_row[r]
            fn = make(kind, diff, r)
            # patch 中心 (世界坐标)
            cx = -ext + patch * (c + 0.5)
            cy = -ext + patch * (r + 0.5)
            x = XX - cx
            y = YY - cy
            inside = (np.abs(x) <= half) & (np.abs(y) <= half)
            if not inside.any():
                continue
            h = fn(x, y, rng)
            h = np.clip(h, 0.0, 0.32)
            # patch 边缘淡出到 0, 避免 patch 之间出现竖墙
            h = h * _border_fade(x, y, half, border)
            if kind == "flat":
                h = np.zeros_like(h)
            H = np.where(inside, h, H)
            if inside.any():
                layout_meta.append(dict(
                    row=int(r), col=int(c), type=str(kind), difficulty=float(diff),
                    x0=float(cx - half), x1=float(cx + half),
                    y0=float(cy - half), y1=float(cy + half),
                    h_max_patch=float(np.max(h[inside]))))

    # 原点压平 (reset 与平地档一致; 中心 patch 本来就是 flat, 这里只是兜底)
    r = np.hypot(XX, YY)
    fade = 1.0 - _smoothstep((1.15 - r) / 0.45)     # r<0.7 -> 1, r>1.15 -> 0
    fade = np.clip(fade, 0.0, 1.0)
    H = H * fade

    H = np.clip(H, 0.0, 0.32)
    meta = dict(n=n, ext=ext, patch_m=patch, patches=P, preset=preset, seed=seed,
                h_min=float(H.min()), h_max=float(H.max()), h_mean=float(H.mean()),
                axis_convention="png = flipud(h.T); 世界(x,y) == 生成器(x,y)",
                hardness=float(hr), layout=layout_meta)
    return H, meta


def write_png(H: np.ndarray, elevation: float, path: str) -> None:
    from PIL import Image
    a = H / max(elevation, 1e-9)
    # 实测约定 (sim/_tmp 小实验, 引擎 stb/lodepng 读入): 对 n x n 图,
    #   hfield_data = flipud(png)            (行翻转, 无转置)
    #   世界 (ix, iy) 的高度 = hfield_data[iy, ix]
    # 而本脚本的 H 就是 H[iy, ix] (meshgrid: 行=y, 列=x)。
    # 因此写 png = flipud(H) 即得 "世界(x,y) == 生成器(x,y)"。
    arr = (np.flipud(a) * 65535.0).round().astype(np.uint16)
    Image.fromarray(arr).save(path)


SCENE_TMPL = '''<mujoco model="go2 terrain mix scene">
  <!-- 由 sim/gen_terrain_go2_mix.py 生成, 不要手改。
       {preset} 混合地形: {patches}x{patches} 个 {patch_m:.1f}m patch,
       {n}x{n} 格 ({cell_m:.3f} m/格), 高度 0..{elev:.3f} m。 -->
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
    <!-- 注意路径口径: MJCF 的 meshdir="assets" (来自 go2_mjx_position.xml) 会作用到
         hfield/texture 的 file 上, 所以这里写文件名而不是 "assets/xxx" ——
         写 assets/ 会变成 assets/assets/xxx。 -->
    <texture type="2d" name="mix_ground" builtin="checker" mark="edge"
      rgb1="0.24 0.30 0.24" rgb2="0.16 0.20 0.17" markrgb="0.5 0.6 0.4"
      width="300" height="300"/>
    <material name="mix_ground" texture="mix_ground" texuniform="true" texrepeat="4 4" reflectance=".2"/>
    <!-- size = "radius_x radius_y elevation_z base_z"; base_z 只撑包围盒, 不影响形状 -->
    <hfield name="terrain" file="{prefix}.png" size="{ext} {ext} {elev:.4f} 0.0960"/>
  </asset>

  <worldbody>
    <!-- floor 名字不能改: 足端接触传感器绑定 geom2="floor" -->
    <geom name="floor" type="hfield" hfield="terrain" material="mix_ground"
      pos="0 0 0" friction="1.0 0.02 0.01"/>
  </worldbody>
</mujoco>
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=256, help="hfield 分辨率 (默认 256 = 4.7cm 格距)")
    ap.add_argument("--ext", type=float, default=6.0, help="半边长 (米), 默认 6 -> 12x12m")
    ap.add_argument("--preset", default="mix",
                    choices=["mix", "mix_hard", "v5", "stairs", "rough",
                             "stairs_only", "slope_only", "rough_only", "boxes_only",
                             "wave_only", "perlin_only", "obstacles_only",
                             "flat_only", "v7", "v8"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hardness", type=float, default=1.0,
                    help="全局难度倍数 (1.0=当前; mix_hard 自动 >=1.5)")
    ap.add_argument("--elevation", type=float, default=None,
                    help="hfield 高度量程 (默认 = 地形最大高度, 向上取整到 cm)")
    ap.add_argument("--out_prefix", default="terrain_mix", help="输出文件名前缀")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    H, meta = build(args.n, args.ext, args.preset, args.seed, args.hardness)
    elev = args.elevation if args.elevation is not None else float(np.ceil(H.max() * 20) / 20)
    elev = max(elev, 0.05)
    cell = 2 * args.ext / args.n
    meta.update(elevation=elev, cell_m=cell, prefix=args.out_prefix)

    print("[mix] preset=%s n=%d (%0.3f m/格) ext=%.1f -> %.1fx%.1f m" %
          (args.preset, args.n, cell, args.ext, 2 * args.ext, 2 * args.ext))
    print("[mix] h: min=%.4f max=%.4f mean=%.4f  elevation_z=%.3f" %
          (H.min(), H.max(), H.mean(), elev))
    q = np.quantile(H, [0.5, 0.75, 0.9, 0.95, 1.0])
    print("[mix] 高度分位 50/75/90/95/100%% = %s" % np.round(q, 3).tolist())
    # 每格坡度 (用来判断是否过于尖锐)
    gx = np.abs(np.diff(H, axis=1)) / cell
    gy = np.abs(np.diff(H, axis=0)) / cell
    sl = np.concatenate([gx.ravel(), gy.ravel()])
    print("[mix] 格间坡度 rise/run: mean=%.3f p90=%.3f p99=%.3f max=%.2f (p99<=1 即 45deg)"
          % (sl.mean(), np.quantile(sl, .9), np.quantile(sl, .99), sl.max()))
    print("[mix] 平均格间高差 %.2f mm" % (np.abs(np.diff(H, axis=1)).mean() * 1000))
    if args.dry_run:
        return 0

    os.makedirs(ASSETS, exist_ok=True)
    png = os.path.join(ASSETS, args.out_prefix + ".png")
    write_png(H, elev, png)
    # 视觉贴图: 高度着色, 便于在 viewer 里看清地形 (不影响物理)
    from PIL import Image
    a = np.clip(H / max(elev, 1e-9), 0, 1)
    tint = np.stack([0.35 + 0.55 * a, 0.32 + 0.45 * a, 0.30 + 0.55 * a], axis=-1)
    Image.fromarray((np.clip(tint, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(ASSETS, args.out_prefix + "_texture.png"))
    meta_path = os.path.join(ASSETS, args.out_prefix + ".json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    scene = SCENE_TMPL.format(**meta, elev=elev)
    scene_path = os.path.join(ROOT, "models", "go2",
                              "scene_mjx_%s.xml" % args.out_prefix)
    with open(scene_path, "w", encoding="utf-8") as fh:
        fh.write(scene)
    print("[mix] 写入:\n  %s\n  %s\n  %s" % (png, meta_path, scene_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
