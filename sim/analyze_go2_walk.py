#!/usr/bin/env python3
"""Go2 行走质量审计 —— 从可视化记录 (frames.csv) 反推"波动到底大不大"。

回答三个问题, 全部用**实测数据**, 不靠感觉:
  1. 躯干高度 / 倾角 / 速度的波动幅度与**频率**是什么? (是不是步态本身?)
  2. 现有奖励项在稳态每一步已经扣了多少分? (是不是"没惩罚"?)
  3. 有没有步态偏差 (偏航漂移 / 对角支撑相不平衡 / 力矩饱和)?

用法:
  python sim/analyze_go2_walk.py --csv ../mjx-go2-mujoco2ros2/logs/go2_view/frames.csv
  python sim/analyze_go2_walk.py --csv /tmp/run/frames.csv --cmd 0.5 0 0 --warmup 3

奖励项公式与权重必须与 envs/go2_walk.py 的默认配置 + v2 训练配方一致
(线性/角速度跟踪 sigma=0.25, base_height=-200, vertical_vel=-2, orientation=-1,
 torque=-2e-4, xy_angular_vel=-0.05)。改了配方请同步 _REWARD 常量。

注意: CSV 是 15 Hz 抽帧 (每 3-4 个 50 Hz 控制步一行), 所以:
  * 频谱只能看到 <7.5 Hz —— 足够分辨步频 (1.8 Hz) 与 2 倍频 (3.6 Hz);
  * action_rate / 足端类奖励**无法**从该 CSV 还原 (不打印, 避免给出假数)。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# ---- 奖励常量 (对齐 envs/go2_walk.py default_config + train_go2.py v2 配方)
W_LIN, W_ANG = 2.0, 1.0
TRACK_SIGMA = 0.25
W_BASE_HEIGHT = -200.0
BASE_HEIGHT_TARGET = 0.268
W_VERTICAL_VEL = -2.0
W_ORIENTATION = -1.0
W_XY_ANGVEL = -0.05
W_TORQUE = -0.0002
TAU_LIMIT = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}
LEGS = ("FR", "FL", "RR", "RL")
JOINTS = ("hip", "thigh", "calf")


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise SystemExit("CSV 为空: " + path)
    cols = {}
    for k in rows[0]:
        try:
            cols[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            cols[k] = np.array([r[k] for r in rows])
    return cols


def peaks(x, fs, n=4):
    """返回幅度谱前 n 个峰 (freq_hz, 相对幅度)。去均值 + Hann 窗。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if x.size < 8:
        return []
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    f = np.fft.rfftfreq(x.size, 1.0 / fs)
    # 只保留非 DC 的局部极大, 避免把泄露裙边当成多个峰
    idx = [i for i in range(1, spec.size - 1) if spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1]]
    idx = sorted(idx, key=lambda i: -spec[i])[:n]
    top = spec[idx[0]] if idx else 1.0
    return [(float(f[i]), float(spec[i] / top) if top > 0 else 0.0) for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="记录器的 frames.csv")
    ap.add_argument("--cmd", type=float, nargs=3, default=(0.5, 0.0, 0.0),
                    help="当时的指令 (vx, vy, wz)")
    ap.add_argument("--warmup", type=float, default=3.0, help="跳过前 N 秒 (ramp/预摆位)")
    args = ap.parse_args(argv)

    d = load(args.csv)
    t = d["t"]
    m = t >= args.warmup
    if m.sum() < 10:
        raise SystemExit("稳态帧不足 (%d 行): 加长 --duration 或减小 --warmup" % m.sum())
    fs = 1.0 / float(np.median(np.diff(t)))

    def S(k):
        return np.asarray(d[k], dtype=np.float64)[m] if k in d else None

    print("=" * 78)
    print("Go2 行走质量审计   %s" % os.path.basename(args.csv))
    print("=" * 78)
    print("稳态 %d 帧 / %.2f-%.2f s, 抽帧 %.2f Hz, 指令 (%.2f, %.2f, %.2f)"
          % (m.sum(), t[m][0], t[m][-1], fs, *args.cmd))

    z, tilt = S("z"), S("tilt")
    vx, vy, vz = S("vx_truth"), S("vy_truth"), S("vz_truth")
    vxb, vyb = S("vx_body"), S("vy_body")

    # ---------------- 1. 波动幅度 + 频率
    print("\n[1] 波动幅度 (稳态)")
    print("  %-8s %8s %8s %8s %8s" % ("量", "mean", "std", "p5", "p95"))
    for name, x, unit in (("z", z, "m"), ("tilt", tilt, "deg"),
                          ("vx", vx, "m/s"), ("vy", vy, "m/s"), ("vz", vz, "m/s")):
        print("  %-8s %8.4f %8.4f %8.4f %8.4f  %s"
              % (name, x.mean(), x.std(), np.quantile(x, .05), np.quantile(x, .95),
                 "" if unit == "m/s" else unit))

    print("\n  频率成分 (步频参考 1.8 Hz; 2x = 每次落地一次)")
    for name, x in (("z", z), ("tilt", tilt)):
        print("    %-5s 峰: " % name
              + "  ".join("%.2fHz(%.2f)" % (f, a) for f, a in peaks(x, fs, 3)))

    # ---------------- 2. 奖励项预算 (平地: 地形高度恒 0, 参考面 = 0)
    print("\n[2] 现奖励项每步实际值 (平地, 按 env 公式复算; 只含可由 CSV 还原的项)")
    parts = []
    if vxb is not None and vyb is not None:
        cmd = np.array(args.cmd[:2])
        lin = W_LIN * np.exp(-(((vxb - cmd[0]) ** 2 + (vyb - cmd[1]) ** 2) / TRACK_SIGMA ** 2))
    else:  # 没机体系速度列时用 |v| 近似 (偏航漂移会污染, 仅作下界)
        lin = W_LIN * np.exp(-((np.hypot(vx, vy) - args.cmd[0]) ** 2 / TRACK_SIGMA ** 2))
    parts.append(("linear_vel_tracking(+)", lin))
    yaw_deg = wz = None
    if "yaw" in d:      # CSV 里 yaw 是弧度
        yaw_deg = np.degrees(np.unwrap(np.asarray(S("yaw"), dtype=np.float64)))
        wz = np.gradient(np.radians(yaw_deg), t[m])
        parts.append(("angular_vel_tracking(+)",
                      W_ANG * np.exp(-((wz - args.cmd[2]) ** 2) / TRACK_SIGMA ** 2)))
    parts.append(("base_height(-)", W_BASE_HEIGHT * (z - BASE_HEIGHT_TARGET) ** 2))
    parts.append(("vertical_vel(-)", W_VERTICAL_VEL * vz ** 2))
    if "roll" in d:
        roll, pitch = S("roll"), S("pitch")
        gx, gy = -np.sin(pitch), np.sin(roll) * np.cos(pitch)
        parts.append(("orientation(-)", W_ORIENTATION * (gx ** 2 + gy ** 2)))
        wx = np.gradient(roll, t[m])
        wy = np.gradient(pitch, t[m])
        parts.append(("xy_angular_vel(-)", W_XY_ANGVEL * (wx ** 2 + wy ** 2)))
    tau = np.stack([S("tau_%s_%s" % (leg, j)) for leg in LEGS for j in JOINTS])
    parts.append(("torque(-)", W_TORQUE * (tau ** 2).sum(axis=0)))

    for name, v in parts:
        print("    %-24s mean=%+8.4f  std=%7.4f  [%+.4f, %+.4f]"
              % (name, v.mean(), v.std(), v.min(), v.max()))
    pos = sum(float(v.mean()) for k, v in parts if k.endswith("(+)"))
    neg = sum(float(v.mean()) for k, v in parts if k.endswith("(-)"))
    print("    %-24s %+8.4f   (= 上列正/负项均值之和, 未含足端/接触类项)" % ("净合计", pos + neg))

    # 高度成本的构成: 常量偏移 vs 振荡
    off = float(z.mean() - BASE_HEIGHT_TARGET)
    osc = float(z.std())
    print("\n    高度成本拆解: 稳态站高 %+.3f m (目标 %.3f) -> 常量偏移项 %+.4f, 振荡项 %+.4f"
          % (off, BASE_HEIGHT_TARGET, W_BASE_HEIGHT * off ** 2, W_BASE_HEIGHT * osc ** 2))

    # ---------------- 3. 步态偏差
    print("\n[3] 步态偏差检查")
    if wz is not None:
        # 偏航漂移: 对 yaw(t) 线性拟合, 与瞬时 wz 抖动分开
        rate = float(np.polyfit(t[m] - t[m][0], yaw_deg, 1)[0])
        print("  偏航: yaw %+.2f -> %+.2f deg, 线性漂移 %+.2f deg/s (指令 wz=%+.2f rad/s)"
              % (yaw_deg[0], yaw_deg[-1], rate, args.cmd[2]))
        print("  瞬时 wz: mean %+.4f rad/s, std %.4f rad/s"
              % (wz.mean(), wz.std()))
        if abs(args.cmd[2]) < 0.05 and abs(wz.mean()) > 0.02:
            print("  !! 指令 wz=0 但存在 %+.3f rad/s 系统性偏航 -> 会走出圆弧 (需查对称性/角速度跟踪)"
                  % wz.mean())
    else:
        print("  偏航: CSV 无 yaw 列 (旧记录器), 无法判断 —— 用新版 record.py 重录")

    # 关节力矩饱和
    print("  力矩饱和 (稳态, >99%% 上限视为饱和):")
    for leg in LEGS:
        for j in JOINTS:
            a = np.abs(S("tau_%s_%s" % (leg, j)))
            lim = TAU_LIMIT[j]
            frac = float((a > 0.99 * lim).mean())
            if frac > 0.005:
                print("    %s_%-5s |tau|max=%6.2f/%5.1f  N*m  饱和帧占比 %.1f%%"
                      % (leg, j, a.max(), lim, 100 * frac))

    # 对角支撑相 (需要 foot_* 接触列; 该列单位是传感器读数, >阈值即触地)
    if all(k in d for k in ("foot_0", "foot_1", "foot_2", "foot_3")):
        f = np.stack([S("foot_%d" % i) for i in range(4)])
        thr = 0.5 * max(1.0, float(np.quantile(f, .9)) * 0.2)
        duty = (f > thr).mean(axis=1)
        print("  支撑相占空比 (FR,FL,RR,RL) = [%s]  对角对 A(FR+RL)=%.2f B(FL+RR)=%.2f"
              % (", ".join("%.2f" % x for x in duty), (duty[0] + duty[3]) / 2, (duty[1] + duty[2]) / 2))
        if abs((duty[0] + duty[3]) / 2 - (duty[1] + duty[2]) / 2) > 0.08:
            print("  !! 对角对占空比差 >0.08 -> 左右不平衡 (着地时间不均, 会推偏航/横滚)")

    print("\n判读要点:")
    print("  * z 的 2x 步频峰 / tilt 的 1x 步频峰 = **步态本身**的起伏, 不是控制器不稳定。")
    print("  * 想压这种起伏, 加惩罚会同时压掉步态 -> 更僵硬的腿 + 更高力矩, 通常不划算。")
    print("  * 真正该修的通常是: 偏航漂移 / 对角对不平衡 / 站高常量偏移 (本脚本 [3] 段)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
