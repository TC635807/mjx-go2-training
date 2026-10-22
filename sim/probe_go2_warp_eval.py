#!/usr/bin/env python
"""把策略放回**训练物理 (warp 后端)** 评测, 与 sim/eval_go2.py 的 CPU 物理对照。

动机: 40M 软接触训练出来的策略, 在 CPU MuJoCo (100/50) 的地形档上大量摔倒,
而训练时的 eval 里 ep_len 还有 12s。两个可能: (a) 策略本身不行; (b) 训练物理
(warp, 2/5 次迭代) 与 CPU 物理 (100/50) 在地形接触上不等价。
这个脚本用**同一个 XML、同一个出生点、同一条指令**在 warp 上跑一遍, 直接分辨。

用法:
  python sim/probe_go2_warp_eval.py --policy <orbax ckpt 目录> --envs 32 --steps 750
  python sim/probe_go2_warp_eval.py --policy <ckpt> --spawn origin --cmd 0.5 0 0
"""
import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(_ROOT))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from envs.go2_walk import Go2Walk, default_config  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", required=True, help="orbax 检查点目录 / pkl")
  ap.add_argument("--envs", type=int, default=32)
  ap.add_argument("--steps", type=int, default=750)
  ap.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
  ap.add_argument("--spawn", choices=["origin", "random"], default="origin")
  ap.add_argument("--terrain", action="store_true", default=True)
  ap.add_argument("--terrain_xml", type=str, default=None,
                  help="覆盖地形场景 XML (默认旧地形); 例: models/go2/scene_mjx_terrain_mix.xml")
  ap.add_argument("--boxgrid_json", type=str, default="",
                  help="v26/R1: box geom 地形的解析高度表 json (gen_terrain_boxgeom.py)")
  ap.add_argument("--contact_mode", choices=["sensor", "clearance"],
                  default="sensor", help="足端接触判据 (box geom 地形用 clearance)")
  ap.add_argument("--spawn_ring", action="store_true",
                  help="v26/R1: 环形出生 (外圈平地/朝向中心)")
  ap.add_argument("--obs_history_frames", type=int, default=0,
                  help="v27: actor 历史帧数 (与训练一致)")
  ap.add_argument("--ring_r", type=float, nargs=2, default=None,
                  help="hfield 地形环形出生半径 (m); bg 地形自动算")

  ap.add_argument("--scan_nx", type=int, default=None)
  ap.add_argument("--scan_ny", type=int, default=None)
  ap.add_argument("--scan_x", type=float, nargs=2, default=None)
  ap.add_argument("--scan_y", type=float, nargs=2, default=None)
  ap.add_argument("--step_offset", type=int, default=0)
  ap.add_argument("--diag", action="store_true",
                  help="打印摔倒位置/地形诊断 (摔倒时刻、地形高度/坡度、姿态、足端离地、"
                       "按出生地形分组的存活率)")
  # v6: 雷达遮挡模型 (评测时用来做鲁棒性扫描; 默认关 = 完美扫描)
  ap.add_argument("--lidar", action="store_true")
  ap.add_argument("--lidar_occ_recall", type=float, default=None)
  ap.add_argument("--lidar_dropout", type=float, default=None)
  ap.add_argument("--lidar_ray_samples", type=int, default=None)
  ap.add_argument("--no_lidar_self_occ", action="store_true")
  ap.add_argument("--lidar_mount", type=float, nargs=3, default=None)
  ap.add_argument("--lidar_vfov", type=float, nargs=2, default=None)
  ap.add_argument("--lidar_mount_jitter", type=float, default=None)
  ap.add_argument("--lidar_vfov_jitter", type=float, default=None)
  ap.add_argument("--lidar_sector_prob", type=float, default=None)
  ap.add_argument("--lidar_sector_width", type=float, nargs=2, default=None)
  ap.add_argument("--spawn_max_slope", type=float, default=0.0,
                  help="v24 评测口径: 出生点局部坡度 (rise/run) 上限, 0=不过滤。"
                       "0.2-0.3 可排除'出生就骑在箱棱/台阶上'的初始条件")
  ap.add_argument("--healthy_z_ref", choices=["feet", "body"], default="feet",
                  help="健康终止的 z 基准: feet=触地足下方地形 (v24 默认, 修 boxes "
                       "躯干跨箱顶假终止); body=旧版躯干正下方地形 (复现旧日志用)")
  ap.add_argument("--foot-contact", choices=["soft", "hard"], default="hard",
                  help="足端接触档: hard = XML 原样 (condim6/0.001, 与部署一致); "
                       "soft = env 覆盖成 condim3/0.023")
  args = ap.parse_args()

  cfg = default_config()
  cfg.terrain = True
  if args.terrain_xml:
    cfg.terrain_xml = args.terrain_xml
  cfg.terrain_boxgrid_json = args.boxgrid_json
  cfg.contact_mode = args.contact_mode
  cfg.random_init.spawn_ring = args.spawn_ring
  if args.obs_history_frames > 0:
    cfg.obs_history.enable = True
    cfg.obs_history.n_frames = int(args.obs_history_frames)
  if args.ring_r is not None:
    cfg.random_init.ring_r = tuple(args.ring_r)
  if args.scan_nx is not None:
    cfg.height_scan.nx = args.scan_nx
  if args.scan_ny is not None:
    cfg.height_scan.ny = args.scan_ny
  if args.scan_x is not None:
    cfg.height_scan.x_min, cfg.height_scan.x_max = args.scan_x
  if args.scan_y is not None:
    cfg.height_scan.y_min, cfg.height_scan.y_max = args.scan_y
  cfg.height_scan.enable = True
  if args.lidar:
    ld = cfg.height_scan.lidar
    ld.enable = True
    if args.lidar_occ_recall is not None:
      ld.occ_recall = args.lidar_occ_recall
    if args.lidar_dropout is not None:
      ld.dropout = args.lidar_dropout
    if args.lidar_ray_samples is not None:
      ld.n_ray_samples = args.lidar_ray_samples
    if args.no_lidar_self_occ:
      ld.self_occlusion = False
    if args.lidar_mount is not None:
      ld.mount = [float(v) for v in args.lidar_mount]
    if args.lidar_vfov is not None:
      ld.vfov = [float(v) for v in args.lidar_vfov]
    if args.lidar_mount_jitter is not None:
      ld.mount_jitter = [args.lidar_mount_jitter] * 3
    if args.lidar_vfov_jitter is not None:
      ld.vfov_jitter = args.lidar_vfov_jitter
    if args.lidar_sector_prob is not None:
      ld.sector_prob = args.lidar_sector_prob
    if args.lidar_sector_width is not None:
      ld.sector_width = [float(v) for v in args.lidar_sector_width]
  cfg.healthy_z_ref = args.healthy_z_ref
  cfg.random_init.spawn_max_slope = args.spawn_max_slope
  cfg.reward_config.scales.feet_phase = 2.0
  cfg.reward_config.scales.feet_slip = -0.5
  cfg.reward_config.scales.feet_height = -1.0
  cfg.reward_config.scales.feet_dangle = -0.5
  cfg.reward_config.scales.feet_impact = -2.0
  cfg.reward_config.scales.body_contact = -150.0
  cfg.reward_config.scales.ascent = 4.0
  cfg.reward_config.scales.symmetry = -0.4
  cfg.command_config.sample = False
  cfg.command_config.fixed = list(args.cmd)
  cfg.random_init.enable = (args.spawn == "random")
  # 必须与被评测策略的训练档一致, 否则测的是另一套接触物理:
  #   soft = condim3 + solimp 0.023 (v1 用的档)
  #   hard = XML 原样 (condim6 + solimp 0.001), 与部署模型一致 (v2 用的档)
  cfg.foot_soft_contact = (args.foot_contact == "soft")

  env = Go2Walk(cfg)
  print(f"env={type(env).__name__} action_size={env.action_size} "
        f"terrain={cfg.terrain} scan={cfg.height_scan.enable} "
        f"spawn={args.spawn} cmd={args.cmd}")
  print(f"obs: state={env.observation_size['state']} "
        f"priv={env.observation_size['privileged_state']}")

  from common import load_policy
  infer, ns = load_policy(os.path.abspath(args.policy))
  print(f"policy loaded, policy obs dim={ns}")

  def _slope_at(xy):
    o = 0.15
    pts = xy[None, :] + jp.array([[o, 0.0], [-o, 0.0], [0.0, o], [0.0, -o]])
    hs = env.terrain_height(pts)
    gx = (hs[0] - hs[1]) / (2.0 * o)
    gy = (hs[2] - hs[3]) / (2.0 * o)
    return jp.sqrt(gx * gx + gy * gy)

  def one(rng):
    state = env.reset(rng)
    spawn = {}
    diag = {}
    if args.diag:
      xy0 = state.data.qpos[:2]
      spawn = dict(h=env.terrain_height(xy0[None, :])[0], slope=_slope_at(xy0))
      diag = dict(
          fall_step=jp.full((), -1, jp.int32),
          fall_xy=jp.zeros((2,)),
          fall_h=jp.zeros(()), fall_slope=jp.zeros(()),
          fall_roll=jp.zeros(()), fall_pitch=jp.zeros(()),
          fall_z=jp.zeros(()), fall_vx=jp.zeros(()),
          fall_clr=jp.zeros((4,)),
          fall_zrel_body=jp.zeros(()), fall_zrel_feet=jp.zeros(()),
          fall_cause=jp.zeros((), jp.int32), fall_nc=jp.zeros(()),
          max_pitch=jp.zeros(()), max_roll=jp.zeros(()),
          min_z=jp.full((), 9.0))

    def body(carry, i):
      st, alive, n_alive, vx_sum, z_sum, _hist, _diag = carry
      act, _ = infer(st.obs, rng)
      st2 = env.step(st, act)
      d = st2.data
      # body-frame vx
      R = d.site_xmat[env._imu_site_id].reshape(3, 3)
      v = R.T @ d.qvel[:3]
      vxl = v[0]
      new_alive = alive & (~st2.done.astype(bool))
      # v6: 存活曲线 (每个控制步还活着的 env 数)
      hist = _hist.at[i].set(new_alive.astype(jp.float32))
      if args.diag:
        q = d.qpos[3:7]
        q = q / jp.maximum(jp.linalg.norm(q), 1e-8)
        w, x, y, z = q
        roll = jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = jp.arcsin(jp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        fell = alive & st2.done.astype(bool)
        fh = env.terrain_height(d.qpos[None, :2])[0]
        fsl = _slope_at(d.qpos[:2])
        foot_xy = d.site_xpos[env._feet_site_id][:, :2]
        fh_feet = env.terrain_height(foot_xy)
        fclr = d.site_xpos[env._feet_site_id][:, 2] - fh_feet
        # 终止原因拆解: 旧判据 z_rel_body 用躯干下方地形; 新候选 z_rel_feet
        # 用"触地足"下方地形中位数 (躯干压在箱顶/台阶棱上时不会误判)
        zrel_body = d.qpos[2] - fh
        contact = fclr < 0.06
        nc = jp.sum(contact.astype(jp.float32))
        # 与 env._is_healthy (v24) 口径一致: 支撑面 = 四足下方地形最小值
        fh_supp = jp.min(fh_feet)
        zrel_feet = d.qpos[2] - fh_supp
        _rad = jp.deg2rad(10.0)
        zlo = zrel_body <= 0.19
        zhi = zrel_body >= 0.65
        tilt = (jp.abs(q[1]) >= _rad) | (jp.abs(q[2]) >= _rad)
        cause = zlo.astype(jp.int32) + 2 * zhi.astype(jp.int32) + 4 * tilt.astype(jp.int32)
        _diag = dict(
            fall_step=jp.where(fell, i, _diag["fall_step"]),
            fall_xy=jp.where(fell, d.qpos[:2], _diag["fall_xy"]),
            fall_h=jp.where(fell, fh, _diag["fall_h"]),
            fall_slope=jp.where(fell, fsl, _diag["fall_slope"]),
            fall_roll=jp.where(fell, roll, _diag["fall_roll"]),
            fall_pitch=jp.where(fell, pitch, _diag["fall_pitch"]),
            fall_z=jp.where(fell, d.qpos[2], _diag["fall_z"]),
            fall_vx=jp.where(fell, vxl, _diag["fall_vx"]),
            fall_clr=jp.where(fell, fclr, _diag["fall_clr"]),
            fall_zrel_body=jp.where(fell, zrel_body, _diag["fall_zrel_body"]),
            fall_zrel_feet=jp.where(fell, zrel_feet, _diag["fall_zrel_feet"]),
            fall_cause=jp.where(fell, cause, _diag["fall_cause"]),
            fall_nc=jp.where(fell, nc, _diag["fall_nc"]),
            max_pitch=jp.where(alive, jp.maximum(_diag["max_pitch"], jp.abs(pitch)),
                               _diag["max_pitch"]),
            max_roll=jp.where(alive, jp.maximum(_diag["max_roll"], jp.abs(roll)),
                              _diag["max_roll"]),
            min_z=jp.where(alive, jp.minimum(_diag["min_z"], d.qpos[2]),
                           _diag["min_z"]))
      return (st2, new_alive, n_alive + new_alive.astype(jp.float32),
              vx_sum + jp.where(new_alive, vxl, 0.0),
              z_sum + jp.where(new_alive, d.qpos[2], 0.0), hist, _diag), None
    hist0 = jp.zeros((args.steps,))
    carry = (state, jp.array(True), jp.zeros(()), jp.zeros(()),
             jp.zeros(()), hist0, diag)
    carry, _ = jax.lax.scan(body, carry, jp.arange(args.steps),
                            length=args.steps)
    st, alive, n_alive, vx_sum, z_sum, hist, diag = carry
    # v6: 遮挡统计 (最后一步, 所有 env 平均)
    _sv = jp.array(1.0); _si = jp.array(1.0)
    if args.lidar:
      _sv = jp.mean(st.info["scan_stats"][0])
      _si = jp.mean(st.info["scan_stats"][1])
    return alive, n_alive, vx_sum, z_sum, _sv, _si, hist, diag, spawn

  rngs = jax.random.split(jax.random.PRNGKey(0), args.envs)
  (alive, n_alive, vx_sum, z_sum, sv, si, hist,
   diag, spawn) = jax.jit(jax.vmap(one))(rngs)
  alive = np.asarray(alive); n_alive = np.asarray(n_alive)
  sv = np.asarray(sv); si = np.asarray(si)
  hist = np.asarray(hist)      # (envs, steps)
  vx = np.asarray(vx_sum) / np.maximum(n_alive, 1)
  z = np.asarray(z_sum) / np.maximum(n_alive, 1)
  print("\n=== warp (训练物理) 结果 ===")
  print(f"  存活率 (跑满 {args.steps} 步) = {100*alive.mean():.1f}%  "
        f"({int(alive.sum())}/{args.envs})")
  print(f"  存活步数 = {n_alive.mean():.1f} ± {n_alive.std():.1f} "
        f"(中位 {np.median(n_alive):.0f})")
  ok = n_alive > 50
  if ok.any():
    print(f"  存活期机体系 vx = {vx[ok].mean():+.3f} m/s (cmd {args.cmd[0]:+.3f})")
    print(f"  存活期躯干 z = {z[ok].mean():.3f} m")
  if args.lidar:
    print(f"  scan 物理可见率 = {sv.mean():.3f}, 信息可用率 = {si.mean():.3f}")
  # v6: 存活曲线 (排除出生瞬态的"真行走"存活率)
  pts = [int(round(x / 0.02)) for x in (0.5, 1, 2, 3, 4, 6)]
  pts = [p for p in pts if 0 < p <= args.steps]
  if pts:
    print("  存活曲线: " + "  ".join(
        "%.1fs=%.1f%%" % (p * 0.02, 100.0 * hist[:, p - 1].mean()) for p in pts))
  print(f"  总耗时 {n_alive.sum()*0.02:.0f} 仿真秒")
  if args.diag:
    fs = np.asarray(diag["fall_step"])
    fell = fs >= 0
    print("\n=== 地形诊断 ===")
    print(f"  摔倒 env = {int(fell.sum())}/{args.envs} "
          f"({100.0*fell.mean():.1f}%)")
    if fell.any():
      t = fs[fell] * 0.02
      print("  摔倒时刻: 中位 %.2fs  p10 %.2fs  p90 %.2fs"
            % (np.median(t), np.quantile(t, .1), np.quantile(t, .9)))
      fh = np.asarray(diag["fall_h"])[fell]
      fsl = np.asarray(diag["fall_slope"])[fell]
      fr = np.abs(np.asarray(diag["fall_roll"])[fell])
      fp = np.abs(np.asarray(diag["fall_pitch"])[fell])
      fz = np.asarray(diag["fall_z"])[fell]
      fv = np.asarray(diag["fall_vx"])[fell]
      fc = np.asarray(diag["fall_clr"])[fell]
      print("  摔倒处地形高度 m: p10/50/90 = %s"
            % np.round(np.quantile(fh, [.1, .5, .9]), 3).tolist())
      print("  摔倒处局部坡度 tan: p10/50/90 = %s"
            % np.round(np.quantile(fsl, [.1, .5, .9]), 2).tolist())
      print("  摔倒时 |roll|/|pitch| = %.0f/%.0f deg (中位)"
            % (np.degrees(np.median(fr)), np.degrees(np.median(fp))))
      print("  摔倒时 z=%.3f m, vx=%+.2f m/s (中位)"
            % (np.median(fz), np.median(fv)))
      print("  摔倒时足端离地 m (中位): %s"
            % np.round(np.median(fc, axis=0), 3).tolist())
      fc_ = np.asarray(diag["fall_cause"])[fell].astype(int)
      zb = np.asarray(diag["fall_zrel_body"])[fell]
      zf = np.asarray(diag["fall_zrel_feet"])[fell]
      nc_ = np.asarray(diag["fall_nc"])[fell]
      names = {1: "仅z低", 2: "仅z高", 4: "仅tilt", 5: "z低+tilt",
               6: "z高+tilt", 7: "z低+z高+tilt"}
      print("  终止原因: " + "  ".join(
          "%s=%d" % (names.get(int(c), str(int(c))), int((fc_ == c).sum()))
          for c in np.unique(fc_)))
      print("  摔倒时 z_rel(躯干下方地形) 中位 %.3f m | "
            "z_rel(触地足下方地形) 中位 %.3f m | 触地足数 %.1f"
            % (np.median(zb), np.median(zf), np.median(nc_)))
      only_z = (fc_ & 5) == 1     # z低, 不含 tilt
      if only_z.any():
        ok_feet = (zf[only_z] > 0.19) & (zf[only_z] < 0.65)
        print("  z低终止中: 新判据下本应健康的比例 = %.1f%% "
              "(n=%d, 旧z_rel中位 %.3f, 新z_rel中位 %.3f)"
              % (100.0 * ok_feet.mean(), int(only_z.sum()),
                 np.median(zb[only_z]), np.median(zf[only_z])))
    alive_np = np.asarray(alive)
    def _grp(vals, edges, name):
      print("  按出生%s分组存活率:" % name)
      for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vals >= lo) & (vals < hi)
        if m.sum() > 0:
          print("    [%.2f,%.2f) n=%3d surv=%.1f%%"
                % (lo, hi, int(m.sum()), 100.0*alive_np[m].mean()))
    if spawn:
      _grp(np.asarray(spawn["h"]), [0, .01, .03, .06, .10, .20], "地形高度(m)")
      _grp(np.asarray(spawn["slope"]), [0, .05, .15, .3, .6, 2.0], "局部坡度(tan)")
      print("  每 env max|pitch|/max|roll| 中位: %.0f/%.0f deg" % (
          np.degrees(np.median(np.asarray(diag["max_pitch"]))),
          np.degrees(np.median(np.asarray(diag["max_roll"])))))
      print("  每 env 最低躯干 z 中位: %.3f m"
            % np.median(np.asarray(diag["min_z"])))
  return 0


if __name__ == "__main__":
  sys.exit(main())
