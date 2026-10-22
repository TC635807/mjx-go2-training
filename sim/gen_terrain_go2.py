#!/usr/bin/env python
"""Go2 训练地形生成器 —— 小幅度凹凸 + 小台阶 (hfield)。

从 sim/gen_terrain.py (Go1 版) 派生, 生成逻辑逐行继承 (坐标约定/归一化/原点压平
全部不动), 只改了输出路径 (models/go2/assets) 与同步的 XML
(models/go2/scene_mjx_terrain.xml), 并把原本解析了却没接上的
--flat_radius/--flat_blend 接到 make_terrain (默认值与原硬编码的 0.5/0.9 相同,
故默认档行为逐位不变)。

设计约束 (为什么这样设计)
------------------------
1. **平地基准必须与原地形逐位一致**: reset 把机器人放在 home keyframe
   (qpos z=0.32), 足端落在地面 z=0。若地形在原点处不为 0, reset 会穿地或
   悬空。故本生成器保证 h(原点)=0, 且起伏**只向上** (h>=0), 平地档与原来的
   `plane` 地板等效。
2. **保留 geom 名 "floor"**: env 的 `{f}_floor_found` 接触传感器绑定了
   geom2="floor" (go2_mjx_position.xml 的 sensor 段), 改名会让四个足端接触
   信号全断 → feet_air_time/奖励全废。故地形 geom 仍叫 "floor"。
3. **确定性**: 固定 seed 生成, 训练与 view 载入同一 PNG → 地形严格一致。
4. **MJX/warp 兼容**: warp 的 collision_driver 支持 HFIELD (CONVEX 路径),
   故 hfield 可用; 但需实测确认不 NaN (见 docs/go2-model-report.md)。

地形构成
--------
  base    : 平滑随机起伏 (高斯滤波噪声), 幅度 bump_amp, 缓坡 -> "凹凸不平"
  steps   : 若干矩形台地, 抬升 step_height -> "小台阶"
  stairs  : 一条三阶小楼梯 (每阶 step_height) -> 连续台阶
  原点    : 强制为 0 并向外平滑过渡 (保证 reset 一致)

用法
----
  # 生成默认档 (起伏 0.05m, 台阶 0.06m, 5 级阶梯 x2)
  /home/tc63/mujoco/.venv/bin/python sim/gen_terrain_go2.py

  # 换档 / 换种子 / 换分辨率
  ... --bump_amp 0.03 --step_height 0.05
  ... --seed 2 --n 320

  # 只打印不写文件 (预览统计)
  ... --dry_run
"""
import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "..")
ASSETS = os.path.join(ROOT, "models", "go2", "assets")
DEFAULT_PNG = os.path.join(ASSETS, "terrain.png")
DEFAULT_META = os.path.join(ASSETS, "terrain.json")

# 与 scene_mjx_position.xml 的平地场景对齐: 地面 z=0
NOMINAL_Z = 0.0


def smooth_noise(rng, n, sigma, amp):
    """高斯滤波白噪声 -> 平滑起伏。amp 是峰值幅度 (米)。"""
    from scipy.ndimage import gaussian_filter
    w = rng.standard_normal((n, n))
    s = gaussian_filter(w, sigma=sigma, mode="wrap")
    # 归一化到 [-1, 1] 再乘 amp (避免 sigma 影响幅度)
    m = np.max(np.abs(s))
    if m > 0:
        s = s / m
    return s * amp


def add_rect_step(h, ext, rect, height, ramp_cells=1, n=None):
    """在矩形区域抬升 height (米)。ramp_cells>1 时边缘有缓坡 (避免垂直壁)。

    rect 用**米**给 (x0, x1, y0, y1), 内部换算到网格; 便于参数可读。
    """
    x0, x1, y0, y1 = rect
    R = ext
    ii = np.arange(n)
    # 网格坐标 -> 米
    xs = (ii / (n - 1)) * 2 * R - R
    ys = xs.copy()
    X, Y = np.meshgrid(xs, ys, indexing="ij")   # X[i,j]=x 坐标

    inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    if ramp_cells <= 1:
        h[inside] += height
        return h

    # 缓坡: 对 inside 掩码做距离渐变 (用多次高斯/迭代膨胀近似)
    cell = 2 * R / (n - 1)
    w = ramp_cells * cell
    # 到矩形边界的有符号距离
    dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    t = np.clip(1.0 - dist / w, 0.0, 1.0)       # 矩形内 1, 向外线性衰减
    t = t * t * (3 - 2 * t)                     # smoothstep
    h += height * t
    return h


def flatten_origin(h, ext, n, radius=0.7, blend=1.2):
    """把原点处强制压到 0, 并向外平滑过渡 (保证 reset 落地一致)。"""
    ii = np.arange(n)
    xs = (ii / (n - 1)) * 2 * ext - ext
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)
    # r<=radius 权重 1 (完全压平), radius~radius+blend 线性过渡到 0
    t = np.clip((r - radius) / blend, 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    h *= t
    return h


def add_staircase(h, ext, x0, y0, step_height, stair_steps, pitch,
                  width, ramp_cells, n, along="x", side_deg=25.0):
    """真正的逐级爬升楼梯: 第 k 级平台高度 = (k+1)*step_height。

    **三处修正 (2026-09-14)**
    1. 旧代码对 k=0,1,2 三段矩形各加**同样的** step_height, 三段并排 -> 是一个
       等高长平台, 根本没爬升 (注释写"三阶小楼梯"但实际不是)。
    2. 不能沿用 `add_rect_step` 的**累加**语义: 其缓坡向外衰减, 级高递增再逐级
       add 时每级会额外吃到上一级抬升。实测 5 级×6cm 累积到 **0.571m**
       (期望 0.30m), 坡度 75°。改用**绝对高度剖面**直接取 max。
    3. 侧壁淡出长度必须由**总高**决定, 不能用踏步面长度: 总高 30cm 用 18.8cm
       淡出 = 58° 陡壁 (实测 64°); 反过来用 5×18.8=94cm 又超过了楼梯半宽,
       中心永远升不到顶 (实测 max 只有 0.119m, 应为 0.30m)。现按目标侧坡角
       `side_deg` 反算: side_len = 总高 / tan(side_deg), 并**校验宽度够不够**。

    along="x" 沿 +x 爬升, "y" 沿 +y 爬升。返回 (h, 实际侧坡淡出长度)。
    """
    R = ext
    ii = np.arange(n)
    coords = (ii / (n - 1)) * 2 * R - R
    cell = 2 * R / (n - 1)
    ramp_len = ramp_cells * cell          # 踏步面上升段长度 (米)
    total_h = step_height * stair_steps   # 楼梯总高

    # 侧壁淡出长度: 由总高与目标侧坡角决定, 保证侧壁也是可走的缓坡。
    # 注意 smoothstep s(x)=3x²-2x³ 的中段最大斜率是 1.5, 实际最大坡度 =
    # atan(1.5 * total_h / side_len) —— 若不补偿, 目标 25° 会实测成 34.7°。
    # 故先把 side_len 放大 1.5 倍, 使峰值斜率恰好落在目标角。
    side_len = 1.5 * total_h / max(np.tan(np.radians(side_deg)), 1e-6)
    # 宽度不足以容纳两侧淡出时自动加宽 (而不是报错让用户手工调参)
    width = max(width, 2 * side_len + 0.4)

    # 宽度校验: 两侧淡出 + 中间至少 0.4m 平台 (上面已自动加宽, 这里只作断言)
    min_width = 2 * side_len + 0.4
    if width < min_width - 1e-9:
        raise ValueError(
            f"楼梯宽度 {width:.2f}m 不足: 总高 {total_h:.2f}m 在 {side_deg}° "
            f"侧坡下需要 {min_width:.2f}m (两侧淡出 {side_len:.2f}m + 平台 0.4m)。")

    # 网格 (indexing="ij": X[i,j]=coords[i] 沿 x, Y[i,j]=coords[j] 沿 y)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    if along == "x":
        T, S = X - x0, Y - y0
    else:
        T, S = Y - y0, X - x0

    # --- 一维剖面: 每级 = 平台 (pitch - ramp) + 上升段 (ramp) ---
    t = np.clip(T, 0.0, None)
    k = np.minimum(np.floor(t / pitch), stair_steps - 1)   # 末级之后不再升
    within = t - k * pitch
    rise = np.clip((within - (pitch - ramp_len)) / ramp_len, 0.0, 1.0)
    rise = rise * rise * (3 - 2 * rise)                    # smoothstep 踏步面
    prof = step_height * (k + rise)

    # --- 横向: [0, width] 内, 两侧按 side_len 淡出 ---
    lat = np.clip(np.minimum(S, width - S) / side_len, 0.0, 1.0)
    lat = lat * lat * (3 - 2 * lat)

    # --- 入口淡入 (楼梯起点不要一步登天) ---
    entry = np.clip(t / ramp_len, 0.0, 1.0)
    entry = entry * entry * (3 - 2 * entry)

    stair = prof * lat * entry
    return np.maximum(h, stair), side_len



def make_terrain(n=128, ext=6.0, bump_amp=0.05, sigma=2.0,
                 step_height=0.06, seed=0, n_steps=6, stairs=True,
                 flat_origin=True, stair_steps=5, n_staircases=2,
                 stair_pitch=0.55, stair_ramp=2, side_deg=30.0,
                 flat_radius=0.5, flat_blend=0.9):
    """生成高度场 (米), 形状 (n,n), 值域 [0, ~step_height*stair_steps]。

    注意 sigma 是**格数**单位: 换 n 时必须按比例调, 否则起伏尺度会变
    (n=384 配 sigma=6 与 n=128 配 sigma=2 才是同一物理尺度)。

    参数 (2026-09-14 加大档)
    ------------------------
    bump_amp    : 随机起伏幅度。0.02(旧) -> 0.05: 起伏更明显
    step_height : 单级台阶高。0.04(旧) -> 0.06
    stair_steps : 阶梯级数 (真爬升)。旧实现只有 3 段等高平台, 不爬升
    n_staircases: 阶梯条数 (一条沿 x, 一条沿 y)
    stair_ramp  : 踏步面缓坡格数。2 格 (18.8cm) 在 6cm 级高下 = 17.7° 踏步面,
                  像真楼梯; 4 格会变成 ~9° 的斜坡, 失去"阶梯"性质
    side_deg    : 楼梯侧壁目标坡度 (度)。侧壁淡出长度按它反算, 保证侧面也是
                  可走的缓坡而不是垂直墙
    flat_radius : 原点压平半径 (米)
    flat_blend  : 原点压平过渡带宽度 (米)
    """
    rng = np.random.default_rng(seed)

    # 1) 平滑起伏, 只保留正向 (h>=0)
    base = smooth_noise(rng, n, sigma, bump_amp)
    h = np.clip(base, 0.0, None)

    # 3) 阶梯地形 (真爬升) —— **先定楼梯占地, 台地与楼梯互斥**。
    # 顺序很重要: 若台地与楼梯重叠, 台地矩形缓坡会叠在台阶侧壁上, 实测最大坡度
    # 冲到 64°。所以先算楼梯占地, 台地生成时排除这些矩形。
    # 踏步面坡度 = atan(step_height / (ramp_cells*格距)); 9.4cm 格距配 6cm 级高、
    # 2 格缓坡 = 17.7°。台阶进深 pitch 必须 > 缓坡长度, 否则相邻坡面叠加成陡壁。
    # 楼梯宽度由 add_staircase 的 side_deg 反算 (总高 30cm 在 30° 侧坡下约需
    # 2*0.78+0.4 ≈ 1.96m); 这里给个下界, 函数内部必要时会再加宽。
    stair_width = 1.8
    stair_rects = []
    flights = []
    if stairs:
        span = stair_steps * stair_pitch
        # 预判实际宽度 (与 add_staircase 同一公式), 保证互斥区算对
        _side = 1.5 * (step_height * stair_steps) / np.tan(np.radians(side_deg))
        stair_width = float(max(stair_width, 2 * _side + 0.4))
        flights = [(ext * 0.05, -ext * 0.80, "x")]           # 沿用原设计: 第三象限
        if n_staircases > 1:
            flights.append((-ext * 0.80, ext * 0.05, "y"))   # 第四象限
        for x0, y0, along in flights[:n_staircases]:
            if along == "x":
                stair_rects.append((x0, x0 + span, y0, y0 + stair_width))
            else:
                stair_rects.append((x0, x0 + stair_width, y0, y0 + span))

    # 2) 台地: 避开原点平地与所有楼梯占地
    placed = []
    tries = 0
    while len(placed) < n_steps and tries < 400:
        tries += 1
        cx = rng.uniform(-ext * 0.75, ext * 0.75)
        cy = rng.uniform(-ext * 0.75, ext * 0.75)
        w = rng.uniform(0.5, 1.3)     # 半宽 (米)
        if np.sqrt(cx ** 2 + cy ** 2) < 1.6:
            continue                   # 别压在原点平地上
        # 避开楼梯占地 (含 0.6m 余量给缓坡)
        if any(cx + w > rx0 - 0.6 and cx - w < rx1 + 0.6
               and cy + w > ry0 - 0.6 and cy - w < ry1 + 0.6
               for rx0, rx1, ry0, ry1 in stair_rects):
            continue
        # 与已有台地保持距离, 避免叠成高墙
        if any(abs(cx - px) < 2.2 and abs(cy - py) < 2.2
               for px, py in placed):
            continue
        height = step_height * rng.uniform(0.7, 1.0)
        h = add_rect_step(h, ext, (cx - w, cx + w, cy - w, cy + w),
                          height, ramp_cells=4, n=n)
        placed.append((cx, cy))

    # 3b) 叠加楼梯 (绝对高度, 取 max)
    flights_info = []
    for x0, y0, along in flights[:n_staircases]:
        h, side_len = add_staircase(
            h, ext, x0, y0, step_height, stair_steps, stair_pitch,
            width=stair_width, ramp_cells=stair_ramp, n=n, along=along,
            side_deg=side_deg)
        flights_info.append((x0, y0, along, side_len))

    # 4) 原点压平 (半径比旧版小: 旧 0.7/1.2 把周围 1.9m 全抹平, 地形感被削掉;
    #    0.5/0.9 只保留 reset 必需的落地区, 更快进入地形)
    if flat_origin:
        h = flatten_origin(h, ext, n, radius=flat_radius, blend=flat_blend)

    return h, placed


def save(h, png_path, meta_path, ext, extra=None):
    """写 PNG (灰度) + JSON 元数据。PNG 是 hfield 的资产格式。

    **坐标约定 (2026-09-14 用引擎真值定死, 见 Go1 版实验记录)**

    生成器内部数组按 `h[i, j] = f(x=coords[i], y=coords[j])` 构造 (直觉约定:
    i 是 x, j 是 y)。但 MuJoCo 读 hfield PNG 时,
        `hfield_data[row, col]`, 其中 **col 沿 +x, row 沿 -y**, 且图像行 0 在顶部。
    实测 (单个高斯凸起放在生成器坐标 (+1.5,-0.9), 再用 `mj_ray` 找峰值位置):
    实际落在世界 (-0.9,-1.5), 即 生成器坐标 -> 世界坐标 = **转置 + y 翻转**。
    7 种候选变换里只有它匹配 (距离 0.00, 其余 >=0.85)。

    后果: 若不修正, 设计的"沿 +x 的楼梯"会变成沿 +y (整体转 90°), 且按设计
    坐标推理位置全错 —— 地形仍物理合法, 但布局不是设计的样子, 做地形观测时
    会采错位置 (静默 bug)。

    修正: 写出时用 `h.T[::-1]` (= flipud(h.T)), 使"生成器坐标 == 世界坐标"。
    """
    from PIL import Image
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    hmax = float(h.max())
    # hfield: height = base_z + data * elevation_z, data = pixel/65535 (16bit)
    # 用 16bit 提高精度 (量化误差 hmax/65535, 可忽略)
    elev_z = hmax if hmax > 1e-6 else 1.0
    data = np.clip(h / elev_z, 0.0, 1.0)
    # 转置 + 上下翻转: 对齐 MuJoCo 的 (row 沿 -y, col 沿 +x, 图像行 0 在顶)
    arr = (np.flipud(data.T) * 65535.0).round().astype(np.uint16)
    Image.fromarray(arr).save(png_path)   # uint16 -> I;16 自动 (勿传 mode, 已弃用)

    meta = {
        "model": "go2",
        "png": os.path.basename(png_path),
        "ext": ext,
        "n": int(h.shape[0]),
        "elevation_z": elev_z,
        "base_z": NOMINAL_Z,
        "h_min": float(h.min()),
        "h_max": hmax,
        "h_mean": float(h.mean()),
        "cell_m": 2 * ext / (h.shape[0] - 1),
        # 记录写出时的坐标变换, 便于以后核对 (生成器坐标 -> 世界坐标)
        "axis_convention": "png = flipud(h.T); 世界(x,y) == 生成器(x,y)",
    }
    if extra:
        meta.update(extra)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def sync_xml(meta, xml_path):
    """把 elevation_z 写回地形 XML 的 hfield size 行 (避免手工同步出错)。

    MuJoCo 对每个 geom-hfield 对最多生成 50 个接触, 超过会截断并警告
    "height field collision overflow"。格距越小越容易超 -> 分辨率不能太高
    (实测 384x384/3.1cm 会持续溢出; 128x128/9.5cm 正常)。
    """
    import re
    if not os.path.exists(xml_path):
        return None
    with open(xml_path, "r", encoding="utf-8") as f:
        s = f.read()
    # 匹配 <hfield ... size="ext ext elev base" ...>
    pat = re.compile(r'(<hfield[^>]*size=")([-\d.eE]+) ([-\d.eE]+) ([-\d.eE]+) ([-\d.eE]+)(")')

    def _sub(m):
        return f"{m.group(1)}{meta['ext']:g} {meta['ext']:g} {meta['elevation_z']:.4f} {m.group(5)}{m.group(6)}"

    new, n = pat.subn(_sub, s)
    if n == 0:
        return None
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(new)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128,
                    help="高度场分辨率 (n x n)。**不要太高**: MuJoCo 每个 "
                         "geom-hfield 对最多 50 个接触, 格距太小时会截断并刷 "
                         "'height field collision overflow' (实测 384/3.1cm 会, "
                         "128/9.5cm 不会)")
    ap.add_argument("--ext", type=float, default=6.0,
                    help="半边长 (米), 地形为 2*ext 见方")
    ap.add_argument("--bump_amp", type=float, default=0.05,
                    help="起伏幅度 (米)。旧默认 0.02 (太轻微), 现 0.05 明显起伏")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="高斯滤波 sigma (格数), 越大越平缓。注意是格数单位, "
                         "换 --n 时要按比例调 (n=384:6 / n=128:2 同物理尺度)")
    ap.add_argument("--step_height", type=float, default=0.06,
                    help="单级台阶高度 (米)。旧默认 0.04, 现 0.06")
    ap.add_argument("--n_steps", type=int, default=6, help="等高台地数量")
    ap.add_argument("--stair_steps", type=int, default=5,
                    help="阶梯级数 (真爬升)。旧实现是 3 段等高平台, 不爬升")
    ap.add_argument("--n_staircases", type=int, default=2,
                    help="阶梯条数 (1=只沿 x, 2=再沿 y 加一条)")
    ap.add_argument("--stair_pitch", type=float, default=0.55,
                    help="每级进深 (米)。必须 > 缓坡长度, 否则相邻坡面叠加成陡壁")
    ap.add_argument("--stair_ramp", type=int, default=2,
                    help="踏步面缓坡格数 (2 格≈18.8cm -> 17.7° 踏步面, 像真楼梯)")
    ap.add_argument("--flat_radius", type=float, default=0.5,
                    help="原点压平半径 (米)")
    ap.add_argument("--flat_blend", type=float, default=0.9,
                    help="原点压平过渡带宽度 (米)")
    ap.add_argument("--no_stairs", action="store_true", help="不要阶梯地形")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--png", default=DEFAULT_PNG)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--dry_run", action="store_true", help="只统计不写文件")
    args = ap.parse_args()

    h, placed = make_terrain(
        n=args.n, ext=args.ext, bump_amp=args.bump_amp, sigma=args.sigma,
        step_height=args.step_height, seed=args.seed,
        n_steps=args.n_steps, stairs=not args.no_stairs,
        stair_steps=args.stair_steps, n_staircases=args.n_staircases,
        stair_pitch=args.stair_pitch, stair_ramp=args.stair_ramp,
        flat_radius=args.flat_radius, flat_blend=args.flat_blend)

    cell = 2 * args.ext / (args.n - 1)
    print(f"地形参数: {args.n}x{args.n}  范围 ±{args.ext}m  "
          f"格距 {cell*100:.1f}cm")
    print(f"  起伏幅度={args.bump_amp}m (sigma={args.sigma})  "
          f"台阶高={args.step_height}m  台地数={len(placed)}  "
          f"阶梯={'无' if args.no_stairs else f'{args.n_staircases}条×{args.stair_steps}级'}"
          f"  seed={args.seed}")
    print(f"  高度: min={h.min():.4f} max={h.max():.4f} "
          f"mean={h.mean():.4f} m")
    print(f"  原点高度 h(0,0)={h[args.n//2, args.n//2]:.6f} m  "
          f"(必须≈0 否则 reset 落地不一致)")
    # 坡度统计 (相邻格差 / 格距)
    gx = np.abs(np.diff(h, axis=0)).max() / cell
    gy = np.abs(np.diff(h, axis=1)).max() / cell
    print(f"  最大坡度: x={np.degrees(np.arctan(gx)):.1f}° "
          f"y={np.degrees(np.arctan(gy)):.1f}°  "
          f"(足端半径 2.2cm vs 格距 {cell*100:.1f}cm)")

    if args.dry_run:
        print("(dry_run: 未写文件)")
        return 0

    meta = save(h, args.png, args.meta, args.ext,
                extra={"bump_amp": args.bump_amp, "step_height": args.step_height,
                       "seed": args.seed, "sigma": args.sigma,
                       "n_steps": len(placed)})
    print(f"\n已写: {args.png}")
    print(f"已写: {args.meta}")
    print(f"  elevation_z={meta['elevation_z']:.4f} (= max height, 用于 XML size)")

    xml_path = os.path.join(ROOT, "models", "go2", "scene_mjx_terrain.xml")
    n_synced = sync_xml(meta, xml_path)
    if n_synced:
        print(f"已同步 elevation_z -> {os.path.basename(xml_path)} (hfield size)")
    else:
        print(f"警告: 未能自动同步 {xml_path}, 请手工核对 size 第三项 "
              f"= {meta['elevation_z']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
