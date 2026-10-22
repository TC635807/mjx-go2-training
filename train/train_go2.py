#!/usr/bin/env python
"""训练 Go2 行走策略 (由 train_go1.py 移植), brax PPO + GPU (warp 后端)。

环境: envs/go2_walk.py (奖励/终止/obs 结构沿用 Go1 版)。
PPO 配置: 对齐 mujoco_playground 官方 locomotion (100M 步, 8192 envs,
(512,256,128) 网络, value 用 privileged_state)。参考仓库用 SB3 默认超参
(12 envs CPU), 我们保持 GPU 并行配置 —— 任务和环境已 1:1, 训练框架差异
本身就是本实验要验证的变量之一。

用法 (WSL, 从 RLcontroller_go1 目录):
  python -u -m train.train_go2 --num_timesteps 100000000
"""
import argparse
import functools
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict

from mujoco_playground import locomotion, wrapper

from envs.go2_walk import Go2Walk, default_config

# 注册自定义环境
locomotion.register_environment("Go2Walk", Go2Walk, default_config)

POLICY_DIR = os.path.join(os.path.dirname(__file__), "..", "policies")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def _check_ckpt_overwrite(ckpt_path, allow_overwrite):
  """拒绝往已有检查点的目录写 (brax 的 save 是 force 覆盖)。

  续训时步数从 0 重新计数, 新检查点会沿用 5/10/15/20M 这套名字; 若 logdir
  指向上一次训练, 就会静默覆盖掉它的证据。v22 之前没有任何闸门。
  """
  if not (os.path.isdir(ckpt_path) and os.listdir(ckpt_path)):
    return
  existing = sorted(os.listdir(ckpt_path))
  if not allow_overwrite:
    raise SystemExit(
        f"拒绝启动: {ckpt_path} 已有 {len(existing)} 个检查点 "
        f"({', '.join(existing[:6])}{'...' if len(existing) > 6 else ''})。\n"
        f"brax 的 checkpoint.save 会 force 覆盖同名步数目录 -> 续训会毁掉"
        f"已有证据。请换 --logdir (推荐), 或确认无价值后加 --allow_overwrite。\n"
        f"（Go1 的证据在 models/go1_v21_logs_frozen_backup/）")
  print(f"警告: --allow_overwrite 已开, 将覆盖 {ckpt_path} 下 {len(existing)} "
        f"个同名检查点目录: {', '.join(existing[:6])}")


def _check_restore(restore, step_offset, obs_state, obs_priv):
  """续训前自检: --restore 指向可读检查点, 且 obs 维度与本次一致。

  brax 对不存在的路径会抛 ValueError, 但"半个目录"或维度不匹配的失败信息
  不直观 —— 这里提前显式报出来。
  """
  if restore is None:
    return
  import json
  r = os.path.abspath(restore)
  if not os.path.isdir(r):
    raise SystemExit(f"--restore 路径不存在: {r}")
  cfg = os.path.join(r, "ppo_network_config.json")
  if not os.path.isfile(cfg):
    raise SystemExit(
        f"--restore 不像一个 brax 检查点 (缺 ppo_network_config.json): {r}")
  with open(cfg) as f:
    rc = json.load(f)
  obs = rc.get("observation_size", {})
  got_state = (obs.get("state") or {}).get("shape")
  got_priv = (obs.get("privileged_state") or {}).get("shape")
  nf = rc.get("network_factory_kwargs", {})
  print(f"续训自检: restore={r}")
  print(f"  检查点 obs: state={got_state} privileged_state={got_priv}  "
        f"layers={nf.get('policy_hidden_layer_sizes')}/"
        f"{nf.get('value_hidden_layer_sizes')}")
  print(f"  本次 obs:   state=[{obs_state}] privileged_state=[{obs_priv}]")
  mism = []
  if got_state is not None and list(got_state) != [obs_state]:
    mism.append(f"state {got_state} != [{obs_state}]")
  if got_priv is not None and list(got_priv) != [obs_priv]:
    mism.append(f"privileged_state {got_priv} != [{obs_priv}]")
  if mism:
    raise SystemExit(
        "--restore 的 obs 维度与本次训练不一致, 无法恢复:\n  "
        + "\n  ".join(mism)
        + "\n(维度不一致说明环境配置与产出该检查点的训练不同)")
  print("  维度一致 ✓")
  print("  注意: brax checkpoint **不含优化器状态与步数** -> Adam 动量从零"
        "重建, num_steps 从 0 记。这是续训必须降 lr 的原因 (§9.2/§22)")
  if step_offset:
    print(f"  step_offset={step_offset} (仅日志显示累计步数, 不改训练语义)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_timesteps", type=int, default=100_000_000)
    ap.add_argument("--num_envs", type=int, default=8192)
    ap.add_argument("--episode_length", type=int, default=750)
    ap.add_argument("--num_evals", type=int, default=20)
    ap.add_argument("--full_reset", action="store_true",
                    help="v29: brax AutoResetWrapper 用 full_reset=True —— 每 episode 调 env.reset "
                         "重新随机出生/地形采样 (默认 False 时每 env 整个训练只回放同一个缓存初始状态; "
                         "MujocoPlayground wrapper.py:124-134)。课程/随机出生必须开")
    ap.add_argument("--cmd_forward_prob", type=float, default=None,
                    help="v33: 混合命令分布 — 该概率采前向偏置范围 (low_forward/high_forward),"
                         " 其余采全范围。地形课程长训练建议 0.7")
    ap.add_argument("--cmd_low_forward", type=str, default=None,
                    help="v33: 前向偏置命令下界 (逗号分隔, 默认 0.25,0,-0.15)")
    ap.add_argument("--cmd_high_forward", type=str, default=None,
                    help="v33: 前向偏置命令上界 (逗号分隔, 默认 0.75,0,0.15)")
    ap.add_argument("--curriculum", action="store_true",
                    help="v30: per-env 地形课程 (sim/brax_curriculum.py)。地形 json 的 "
                         "spawn_entries 按难度升序; 每回合按位移/摔倒升降级")
    ap.add_argument("--curriculum_init_level", type=int, default=2,
                    help="v30: 初始 level 上界 U{0..该值}" )
    ap.add_argument("--curriculum_success_dist", type=float, default=3.0,
                    help="v30: 位移超过该值(且未摔)升级")
    ap.add_argument("--curriculum_min_dist", type=float, default=1.0,
                    help="v30: 位移低于该值(或摔倒)降级")
    ap.add_argument("--naconmax", type=int, default=None,
                    help="v29: 覆盖接触槽总数 (默认 32*4096=131072; 8GB 显存 + full_reset 建议 "
                         "24*2048=49152, 否则 warp EPA 缓冲区可能 OOM)")
    ap.add_argument("--njmax", type=int, default=None,
                    help="v29: 覆盖 constraint 上限 (默认 256; 显存紧时 192)")
    ap.add_argument("--num_resets_per_eval", type=int, default=0,
                    help="v29: 每次评测之间重置 eval env 的次数 (brax ppo.train 参数; 默认 0)")
    ap.add_argument("--restore", type=str, default=None,
                    help="从 checkpoint 目录恢复继续训练")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logdir", type=str, default=None)
    ap.add_argument("--allow_overwrite", action="store_true",
                    help="允许往非空 logdir 写检查点 (默认拒绝)。brax 的 "
                         "checkpoint.save 用 force=True 覆盖同名步数目录, 续训"
                         "时新检查点会沿用 5/10/15/20M 这套相对步数名 -> 若不换 "
                         "logdir 会**静默覆盖**上一版训练的证据 (v22 起加此闸门)")
    ap.add_argument("--step_offset", type=int, default=0,
                    help="续训起点步数 (仅**打印**用): 日志里的 num_steps 是本次"
                         "训练的步数, 加上它才是累计步数。不改训练语义")
    ap.add_argument("--save_name", type=str, default="go2_walk_policy.pkl")
    ap.add_argument("--sb3_opt", action="store_true",
                    help="v2: 对齐 SB3 优化制度 (minibatch 256→640 样本, "
                         "updates 4→8/批, entropy 0, discount 0.99)。"
                         "梯度密度 128/批 → 2048/批 (16×), 100M 步 ≈ "
                         "125 万次梯度更新 ≈ 参考仓库 5M 步 (SB3, 78 万次) 的 1.6×")
    ap.add_argument("--sb3_net", action="store_true",
                    help="v3: 网络缩到 (64,64) + value (64,64) (SB3 默认 "
                         "MlpPolicy 结构, ~1.5 万参数, 与参考仓库对齐)。"
                         "需配合 --sb3_opt (v2 教训: 优化制度是主变量)")
    ap.add_argument("--sb3_full", action="store_true",
                    help="v4: 完全仿照参考仓库 (SB3 默认) 的算法参数 —— "
                         "rollout 24576 (768envs×unroll32), minibatch 64, "
                         "10 epochs (每批 3840 次梯度更新, 密度 0.156/样本 "
                         "= SB3 精确值), entropy 0, discount 0.99, clip 0.2, "
                         "max_grad_norm 0.5, 无 obs 归一化, 无压缩高斯分布, "
                         "tanh 激活, 网络 (64,64)。总步数用 5M (参考默认)。"
                         "唯一残余差异: GAE 窗口 32 vs SB3 2048 (显存所限)")
    ap.add_argument("--sb3_envs", type=int, default=None,
                    help="覆盖 --sb3_full 档的 num_envs (默认 768 = Go1 逐位一致)。"
                         "minibatches 自动取 num_envs/2 以保持 SB3 的 minibatch"
                         "语义 (64 条/批) 与每 rollout 10 epochs 的密度不变。")
    ap.add_argument("--hard_contact", action="store_true",
                    help="v5: 物理对齐参考仓库 — 足端硬接触 (XML 原样 "
                         "solimp 0.015/1/0.031 condim=6) + Newton 100 迭代。"
                         "实测 warp 硬接触与 CPU 参照逐位一致 (diag_hard_contact.py); "
                         "软接触静平衡残差 324 vs 0.08 且会弹飞 (v1-v4 站立根因)")
    ap.add_argument("--norm_obs", action="store_true",
                    help="打开 brax 观测运行归一化 (v6 加过; v10 证明对 NaN "
                         "既非必要也非充分 —— NaN 真因是 njmax 溢出, 见实验"
                         "记录 §4.3。保留开关仅供对照)")
    # --- v11: 指令条件化 (每 episode 从 [cmd_low, cmd_high] 采样指令) ---
    ap.add_argument("--sample_cmd", action="store_true",
                    help="v11: 打开指令采样 (实现一个策略覆盖多方向)。"
                         "默认关 = 恒定指令 [0.5,0,0], 与 v1-v10 一致")
    ap.add_argument("--cmd_low", type=str, default=None,
                    help="指令下界, 逗号分隔 3 值 (vx,vy,wz), 如 '0,0,-1'")
    ap.add_argument("--cmd_high", type=str, default=None,
                    help="指令上界, 逗号分隔 3 值 (vx,vy,wz), 如 '0.5,0,1'")
    ap.add_argument("--fwd_turn", action="store_true",
                    help="v11 预设: 只做前进 + 转向 (vy=0, vx∈[0,0.5], "
                         "wz∈[-1,1])。等价于 --sample_cmd "
                         "--cmd_low 0,0,-1 --cmd_high 0.5,0,1")
    # --- 微调稳定性旋钮 (v11 教训: 直接热启动会被梯度打崩, 见实验记录) ---
    ap.add_argument("--lr", type=float, default=None,
                    help="覆盖学习率 (默认 SB3 档 3e-4)。热启动微调时 Adam 的"
                         "归一化步长会把收敛好的策略推走, 需显著调低 (如 1e-4/3e-5)")
    ap.add_argument("--entropy", type=float, default=None,
                    help="覆盖 entropy_cost (默认 0)。收敛后的策略 log_std 往往"
                         "缩得很小 → 微调时没有探索, 加一点熵可恢复探索")
    ap.add_argument("--layers", type=str, default=None,
                    help="覆盖 policy/value 隐层, 逗号分隔, 如 '128,128'。"
                         "SB3 档 (64,64) 是单指令容量; 指令条件化(多方向)建议加大")
    ap.add_argument("--body_frame", action="store_true",
                    help="v14: 指令解释为机体系(朝向)速度 (obs 的 base linvel 也"
                         "换成机体系)。转向必须用: global 语义下'边走边转'要求"
                         "机器人保持全局速度同时自转 = 持续横行, 不可实现")
    ap.add_argument("--tilt_deg", type=float, default=None,
                    help="覆盖健康终止的 roll/pitch 阈值 (默认 10 → 等效 ±20°)。"
                         "参考仓库的 ±20° 对直行够用, 但转弯时躯干正常俯仰就会"
                         "越界 (v14 实测: 所有终止都卡在 20° 边界, up_z 仍 0.99)")
    ap.add_argument("--healthy_z_ref", choices=["feet", "body"], default="feet",
                    help="v24: 健康终止的 z 基准。feet = 触地足下方地形高度均值 "
                         "(新默认, 修 boxes/台阶棱上躯干跨箱顶导致的假终止); "
                         "body = 旧版躯干正下方地形 (仅用于复现旧日志数字)")
    # --- v16: 防打滑/迈步奖励 (playground go1/joystick 同款配方) ---
    ap.add_argument("--natural_gait", action="store_true",
                    help="v16: 打开'真迈步'奖励 —— 接触相足端滑移惩罚 "
                         "(feet_slip) + 摆动抬脚高度塑形 (feet_height) + "
                         "空距阈值 1.0→0.2 并封顶 0.5s。参考仓库配方下"
                         "'贴地滑行'能拿满速度跟踪奖励, 是 v10-v15 滑行步态的"
                         "直接原因 (实验记录 §10); 这三项把解推向抬腿迈步")
    ap.add_argument("--dangle_penalty", action="store_true",
                    help="v17: 在 --natural_gait 基础上加'悬空过久'逐步惩罚。"
                         "feet_slip/feet_height 只在触地帧结算, 永不落地的腿"
                         "躲得掉 → v16 出现 FL 吊挂 14.7s (duty 0.01)。"
                         "这一项按 clip(air_time-0.8s) 每步扣分, 封顶 2.0")
    ap.add_argument("--slip_penalty", type=float, default=None,
                    help="覆盖 feet_slip 权重 (默认 -0.5)")
    ap.add_argument("--foot_height", type=float, default=None,
                    help="覆盖 feet_height 权重 (默认 -1.0)")
    ap.add_argument("--air_thresh", type=float, default=None,
                    help="覆盖 air_time_threshold (默认 0.2)")
    ap.add_argument("--dangle_w", type=float, default=None,
                    help="覆盖 feet_dangle 权重 (v17 默认 -0.5)")
    # --- v19: 地形感知 + 鲁棒性 ---
    ap.add_argument("--terrain", action="store_true",
                    help="v19: 载入地形场景 (凹凸 + 2 条 5 级阶梯, 总高差 0.30m)。"
                         "同时自动打开随机出生 —— 不随机 xy 的话 8192 环境全"
                         "挤在原点压平区, 永远见不到台阶 (见 §17.2)")
    ap.add_argument("--height_scan", action="store_true",
                    help="v19: 35 点 (7x5) 高度扫描进 obs (actor+critic 都拿)。"
                         "需配合 --terrain (平地档特征恒为 0, 无信息)")
    ap.add_argument("--terrain_xml", type=str, default=None,
                    help="覆盖地形场景 XML (默认 models/go2/scene_mjx_terrain.xml)。"
                         "例: --terrain_xml models/go2/scene_mjx_terrain_mix.xml "
                         "(由 sim/gen_terrain_go2_mix.py 生成)。需配合 --terrain")
    ap.add_argument("--no_random_init", action="store_true",
                    help="v19: 关掉随机出生 (调试用: 全部从原点出生, 便于与"
                         "v18 基线对照)")
    ap.add_argument("--random_init_range", type=float, default=None,
                    help="覆盖出生点 xy 范围 (默认 5.0m; 地形半边长 6m)")
    ap.add_argument("--push", action="store_true",
                    help="v19: 打开周期性外力推挤 (模拟真实扰动, 提升鲁棒性)")
    ap.add_argument("--push_vel", type=float, default=None,
                    help="覆盖推挤强度 (默认 0.5 m/s)")
    ap.add_argument("--push_interval", type=int, default=None,
                    help="覆盖推挤间隔 (默认 250 步 = 5s)")
    ap.add_argument("--obs_noise", action="store_true",
                    help="v19: 打开观测传感器噪声 (小幅值, 提升鲁棒性/sim2real)")
    ap.add_argument("--scan_noise", type=float, default=None,
                    help="覆盖高度扫描噪声 (默认 0.02m, 仅 actor)")
    # v5: 高度扫描几何 (默认 7x5, x[-0.3,0.9], y[-0.4,0.4])
    ap.add_argument("--scan_nx", type=int, default=None, help="v5: 前向采样点数 (默认 7)")
    ap.add_argument("--scan_ny", type=int, default=None, help="v5: 侧向采样点数 (默认 5)")
    ap.add_argument("--scan_x", type=float, nargs=2, default=None,
                    help="v5: 前向范围 x_min x_max (默认 -0.3 0.9)")
    ap.add_argument("--scan_y", type=float, nargs=2, default=None,
                    help="v5: 侧向范围 y_min y_max (默认 -0.4 0.4)")
    # --- v6: 激光雷达遮挡模型 (actor 感知通道) ---
    ap.add_argument("--lidar", action="store_true",
                    help="v6: 打开雷达遮挡模型 (垂直 FOV 盲区 + 地形自遮挡 + "
                         "机腿自遮挡 + 随机丢点); 未观测格点填 fill (默认 0)")
    ap.add_argument("--lidar_mount", type=float, nargs=3, default=None,
                    help="v6: 传感器安装位置 (机体系, m; 默认 0 0 0.032)")
    ap.add_argument("--lidar_vfov", type=float, nargs=2, default=None,
                    help="v6: 垂直视场 (rad; 默认 -0.5 0.2, 与部署 SimLidar 一致)")
    ap.add_argument("--lidar_dropout", type=float, default=None,
                    help="v6: 额外随机丢点率 (0-1; 默认 0)")
    ap.add_argument("--lidar_ray_samples", type=int, default=None,
                    help="v6: 地形自遮挡采样点数 (默认 4; <=1 关闭)")
    ap.add_argument("--no_lidar_self_occ", action="store_true",
                    help="v6: 关闭机腿自遮挡 (只留 FOV 盲区/地形遮挡/丢点)")
    ap.add_argument("--lidar_mount_jitter", type=float, default=None,
                    help="v6: 安装位置每步抖动半宽 (m, 三轴相同; 默认 0)")
    ap.add_argument("--lidar_vfov_jitter", type=float, default=None,
                    help="v6: vfov 下限每步抖动半宽 (rad; 默认 0)")
    ap.add_argument("--lidar_fill", type=float, default=None,
                    help="v6: 未观测格点在特征空间的填充值 (默认 0 = 标称平地)")
    ap.add_argument("--lidar_occ_recall", type=float, default=None,
                    help="v6: 高程图'记忆'概率: 物理被挡格点仍保留真值的概率 "
                         "(默认 0.75; 1=完全记忆, 0=只用当前可见格点/最坏情况)")
    ap.add_argument("--lidar_sector_prob", type=float, default=None,
                    help="v6: 每步整块扇区置为未观测的概率 (成片缺失; 默认 0)")
    ap.add_argument("--lidar_sector_width", type=float, nargs=2, default=None,
                    help="v6: 扇区宽度范围 (rad; 默认 0.6 2.4)")
    # --- v3: 速度估计误差 (腿式里程计的真实误差形态) ---
    ap.add_argument("--obs_vel_bias", type=float, default=None,
                    help="v3: obs 0:3 的每回合常值偏置半宽 (m/s, 逐轴 U(-b,b))。"
                         "实测 sim 内 LegOdometry 偏置约 0.07 m/s, 真机会更大")
    ap.add_argument("--obs_vel_drift", type=float, default=None,
                    help="v3: obs 0:3 的 OU 慢漂稳态 std (m/s, 默认关)")
    ap.add_argument("--obs_vel_tau", type=float, default=None,
                    help="v3: OU 相关时间 (s, 默认 2.0)")
    ap.add_argument("--obs_vel_delay", type=int, default=None,
                    help="v3: obs 0:3 最大延迟 (控制步; 50Hz -> 1 步=20ms, 默认关)")
    ap.add_argument("--obs_noise_linvel", type=float, default=None,
                    help="覆盖 obs_noise.lin_vel 白噪声 (默认 0.05 m/s; "
                         "legged_gym 用 0.5, 我们 v2 偏小)")
    # --- v19: 显式爬升激励 + 下降引导 ---
    ap.add_argument("--climb_reward", action="store_true",
                    help="v19: 打开相对地形的躯干高度奖励 (legged_gym 的 "
                         "base_height)。地形上防止躯干下沉/拖地 —— 爬台阶的"
                         "必要条件 (躯干压低就上不去)。注意它本身不是高度奖励,"
                         "而是站高维护项")
    ap.add_argument("--base_height_w", type=float, default=None,
                    help="覆盖 base_height 权重 (默认 -200.0)。**平方**误差语义, "
                         "量级与线性直觉差很多: 偏差 2cm -> -0.08, 5cm -> -0.5, "
                         "10cm -> -2.0 (实测 z_rel 中位数 0.276 vs 目标 0.28, "
                         "正常站立时该项 ≈ -0.0005, 几乎免费)")
    ap.add_argument("--soft_landing", action="store_true",
                    help="v19: 打开落地冲击惩罚 (下降引导)。按落地帧足端向下"
                         "速度超过阈值的部分平方惩罚, 抑制下台阶/下坡时砸地")
    ap.add_argument("--impact_w", type=float, default=None,
                    help="覆盖 feet_impact 权重 (默认 -2.0)。实测该信号很稀疏"
                         "(95%% 的时刻为 0, 只 0.39%% 的(足,步)对触发) —— "
                         "阈值 0.5m/s 不会误伤正常步态; 单次硬落地约 -0.7~-5")
    ap.add_argument("--impact_vel", type=float, default=None,
                    help="覆盖落地冲击速度阈值 (默认 0.5 m/s, 低于它不罚)")
    # --- v20: 对角步态引导 + 落地惩罚改按时间结算 ---
    ap.add_argument("--gait_phase", action="store_true",
                    help="v20: 打开对角步态相位奖励 (playground feet_phase 同款)。"
                         "v19 退化成'前/后成对'步态 (前腿占空比 13%%、"
                         "对角同步 13%%), 纯惩罚推不出步态, 需要显式相位引导")
    ap.add_argument("--gait_w", type=float, default=None,
                    help="覆盖 feet_phase 权重 (默认 2.0, playground spot 同值)")
    ap.add_argument("--gait_freq", type=float, default=None,
                    help="覆盖步频 Hz (默认 1.8; v19 实测主周期 0.52-0.56s≈1.9Hz)")
    ap.add_argument("--gait_freq_scaled", action="store_true",
                    help="v20: 步频随指令速度缩放 (站着不动不踏步)。默认关 = "
                         "恒定步频 (playground 做法)")
    ap.add_argument("--keep_foot_height", action="store_true",
                    help="v20: 不把 max_foot_height 联动到 gait_swing_height。"
                         "默认联动 (两者应一致, 否则 feet_height 与 feet_phase "
                         "对'抬多高'的要求打架)")
    ap.add_argument("--obs_history_frames", type=int, default=0,
                    help="v27: actor 观测历史帧数 (0=关; 推荐 2-3)。"
                         "历史内容 = base48+phase8=56 维/帧, 不含 scan; "
                         "obs 增加 (N-1)*56 维。部署侧环形缓冲实现。")
    ap.add_argument("--spawn_ring", action="store_true",
                    help="v26/R1: box geom 地形用环形出生 (外圈平地/朝向中心), "
                         "避免出生在竖面上; 半径由 boxgrid 范围自动算")
    ap.add_argument("--spawn_stairs_prob", type=float, default=None,
                    help="v28: 楼梯口出生概率 (0=关; 推荐 0.8)。需地形 json 里有 "
                         "spawn_entries (gen_terrain_stairs_bands.py 生成): 以该概率"
                         "在楼梯带入口前方出生并朝向楼梯 -> 保证每回合都在爬台阶")
    ap.add_argument("--ring_r", type=float, nargs=2, default=None,
                    help="hfield 地形环形出生半径 (m), 用于同协议公平对比")
    ap.add_argument("--boxgrid_json", type=str, default="",
                    help="v26/R1: box geom 地形解析高度表 json (真竖面); 配合 "
                         "--contact_mode clearance 使用")
    ap.add_argument("--contact_mode", choices=["sensor", "clearance"],
                    default="sensor",
                    help="足端接触判据: sensor=foot-floor 传感器 (hfield 地形); "
                         "clearance=几何净空 (box geom 地形)")
    ap.add_argument("--feet_edge_w", type=float, default=None,
                    help="v25/R3: 触地足落在'地形棱'上的惩罚权重 (默认 0=关)。"
                         "extreme-parkour feet_edge 同款; 建议先 -0.3, 扫 {-0.2,-0.3,-0.5}。"
                         "棱掩码 = 相邻 hfield 格高差>0.05m 再膨胀 1 格, 平滑地形不触发")
    ap.add_argument("--feet_edge_bg", action="store_true",
                    help="v28: feet_edge 的棱掩码叠加 boxgrid 高度表 (默认关)。"
                         "**只在宽踏面/粗格地形有用**: 0.4m 格 + 1 格踏面的楼梯上"
                         "掩码膨胀后覆盖 94%% 格点, 会退化成'别踩台阶' (stairs1 实测)")
    ap.add_argument("--feet_edge_thresh", type=float, default=None,
                    help="v28: 地形棱判定阈值 (m, 默认 0.08)。boxgeom 台阶地形"
                         "(hfield 全 0) 的棱从 boxgrid 高度表算, 0.08m 踏步差 == 阈值"
                         "不会被标 -> 台阶专项训练建议 0.06")
    # --- v28: 步态自然性 / 机体稳定性 (用户 2026-10-21 反馈) ---
    ap.add_argument("--hip_abduction", action="store_true",
                    help="v28: 打开髋外展/内收惩罚 (四髋偏离 home 的平方和)。"
                         "动机: 实测 hip_q mean 29-33°/p95 42-47°, hip 动作 p95 "
                         "1.2-1.5rad 顶到 ±1.047 限位, 且 v20/v24/v25a/v25b 一致、"
                         "平地上同样存在 -> 奖励面缺项, 不是地形/仿真问题。"
                         "权重 -0.5 时正常行走约 -0.35/步 (正奖励 ~+2.3/步)")
    ap.add_argument("--hip_abduction_w", type=float, default=None,
                    help="覆盖 hip_abduction 权重 (默认 -0.5, 负值=惩罚)。标定: "
                         "现状 Σ(q_hip-home)²≈0.7 -> -0.5 给 -0.35/步; 太小(-0.1)"
                         "没效果, 太大(-2)会压过正奖励把总奖励打到 0 (PPO 无梯度)")
    ap.add_argument("--hip_action_w", type=float, default=None,
                    help="v28b: 髋**动作目标**偏离 home 的平方和惩罚 (负=惩罚, 默认关)。"
                         "A1 负结果证明状态型 hip_abduction 在 3M 微调里无效 (髋角/动作"
                         "与 v25b 逐位相同), 需要直接对策略输出加压; 建议 -0.5")
    ap.add_argument("--clearance_w", type=float, default=None,
                    help="v29: 速度加权足端净空偏差权重 (mjlab feet_clearance, 建议 -1.0)")
    ap.add_argument("--clearance_target", type=float, default=None,
                    help="v29: 足端目标净空 (m, 默认 0.10; 跨 0.06 riser 需 >=0.083)")
    ap.add_argument("--progress_w", type=float, default=None,
                    help="v29: 世界系 +x 前进奖励权重 (R2: 1.0~1.5; 抗绕行/抗横移)")
    ap.add_argument("--lazy_stop_w", type=float, default=None,
                    help="v29: 防站桩惩罚权重 (负值; R2: -1.5)。cmd>0.1 且 |v|<0.2 时扣分")
    ap.add_argument("--termination_w", type=float, default=None,
                    help="v29: 终止惩罚权重 (负值, 如 -20; 需配合 --reward_clip_min 负值否则被 clip 掉)")
    ap.add_argument("--reward_clip_min", type=float, default=None,
                    help="v29: 覆盖总奖励下限 (默认 0.0; 台阶训练建议 -25 让终止/深穿透惩罚生效)")
    ap.add_argument("--action_acc_w", type=float, default=None,
                    help="v29: 动作二阶差分平滑权重 (PIE 'smoothness', 建议 -0.01)")
    ap.add_argument("--terminate_body_pen", type=float, default=None,
                    help="v29: 非足端碰撞体深穿透终止阈值 (m, 0=关; 台阶训练建议 0.02)。"
                         "PIE 用 illegal_contact(非足端接触力>10N) 终止, 这里用几何净空等价")
    ap.add_argument("--action_rate_w", type=float, default=None,
                    help="覆盖 action_rate 权重 (默认 -0.01; 动作平滑/防抽动)。"
                         "建议 -0.02~-0.03; 与 hip_abduction 联动看动作 p95")
    ap.add_argument("--xy_angvel_w", type=float, default=None,
                    help="覆盖 xy_angular_vel 权重 (默认 -0.05; 惩罚横滚/俯仰角速度"
                         "= 压制'走着晃'。不影响转向: 转向是 qvel[5] yaw)")
    ap.add_argument("--orientation_w", type=float, default=None,
                    help="覆盖 orientation 权重 (默认 -1.0; Σgravity_xy²)。"
                         "注意楼梯上躯干自然俯仰 10-20°, 不要为了'稳'把它压太狠")
    ap.add_argument("--torque_w", type=float, default=None,
                    help="覆盖 torque 权重 (默认 -0.0002)")
    ap.add_argument("--joint_acc_w", type=float, default=None,
                    help="覆盖 joint_acceleration 权重 (默认 -2.5e-7)")
    ap.add_argument("--value_layers", type=str, default=None,
                    help="覆盖 value/critic 网络隐层 (默认与 --layers 相同), 如 "
                         "'1024,512'。privileged_state 维度更高, 容量可单独加大")
    ap.add_argument("--swing_height", type=float, default=None,
                    help="v25/R2: 摆动相目标抬脚高度 (m), 默认 0.15。"
                         "boxes 诊断: 抬脚净空不够越过 0.09-0.15m 箱棱 -> 试 0.18/0.20。"
                         "默认联动 max_foot_height (可用 --keep_foot_height 关掉)")
    ap.add_argument("--body_contact_w", type=float, default=None,
                    help="v21: 非足端部件 (小腿/大腿/躯干) 撞地惩罚权重。"
                         "不加 = 0 (关闭)。修 §20 '后腿关节着地': 原 collision "
                         "项在 warp 后端是空实现 (无 data.contact), 膝盖撑地免费。"
                         "建议 -150 (实测标定: 坏步态平均贡献 -0.39, 与速度跟踪 "
                         "+1.0 同量级; 新增零梯度步最小)")
    ap.add_argument("--body_contact", action="store_true",
                    help="v21: 用推荐权重 -150 打开 body_contact 惩罚")
    ap.add_argument("--body_contact_max", type=float, default=None,
                    help="覆盖累计穿透深度上限 (m, 默认 0.02)。防止深穿透时"
                         "惩罚压过正奖励 -> 总奖励 clip 到 0 -> PPO 无梯度")
    # --- v22: 显式爬升激励 (修 §24 的激励缺口) ---
    ap.add_argument("--ascent", action="store_true",
                    help="v22: 打开「相对出生点的地形升高」奖励。原奖励面里没有"
                         "任何一项随「站得更高」增加 (base_height 是相对量, 速度"
                         "跟踪/相位/成本都与地形高度无关) -> 绕开楼梯与爬上去收益"
                         "完全相同。实测把 v21 策略放到楼梯脚下, 它走 2m 却横向"
                         "漂移 1.4m 绕开了楼梯 (最大爬升 4.7cm) = 激励问题")
    ap.add_argument("--ascent_w", type=float, default=None,
                    help="覆盖 ascent 权重 (默认 2.0)。**上界必须守住**: 这是"
                         "状态型奖励 (每步按当前高度给分), 在台顶站着不动可持续"
                         "收钱。实测套利算术: 走路每步 +1.776, 站着不动 +0.347, "
                         "站台顶 = 0.347 + W*0.30 -> W=4 给 +1.547 (88%% 走路, 险), "
                         "W>=8 给 +2.75 **超过走路 -> 策略会主动退化成站桩**。"
                         "(注意: argparse 的 help 里 %% 必须转义成 %%)")
    ap.add_argument("--ascent_cap", type=float, default=None,
                    help="覆盖升高上限 (m, 默认 0.35)")
    ap.add_argument("--ascent_gate", type=float, default=None,
                    help="v23: ascent 的行进门控速度 (m/s, 默认 0.2)。ascent 乘上 "
                         "clip(|v_body|/gate,0,1), 静止时归零 -> 不再有'站台顶不动"
                         "也收钱'的套利, 于是 W 可以提上去。设 0 关闭 (=v22 行为)")
    # --- v23: 左右镜像对称正则 ---
    ap.add_argument("--symmetry", action="store_true",
                    help="v23: 打开左右镜像对称正则 (修 §27 的永久不对称姿态: "
                         "后腿 hip 共模 -22 度, 左后膝多弯 0.5rad -> 小腿蹭地 66%%)。"
                         "形式 -W*EMA(镜像偏差); 用时间平均以避免对抗 trot 的左右反相")
    ap.add_argument("--symmetry_w", type=float, default=None,
                    help="覆盖对称权重 (默认 -0.4, 负值=惩罚)。按实测标定: "
                         "v22 的 EMA≈1.19 -> 0.48/步; v18(对称) EMA≈0.045 -> 0.018/步")
    ap.add_argument("--symmetry_beta", type=float, default=None,
                    help="覆盖 EMA 步长 (默认 0.01; 1/beta=100 步 ≈ 3.6 个步态周期)")
    ap.add_argument("--symmetry_cap", type=float, default=None,
                    help="覆盖对称项上限 (默认 2.0, 防负项压过正奖励把梯度打 0)")
    ap.add_argument("--trunk_height_ref", action="store_true",
                    help="v22 实验性对照 (**非默认**): base_height 参考点改成只用"
                         "躯干正下方 (默认是躯干+4足 5 点均值, 与 v21 一致)。"
                         "实测 mean5 的额外惩罚是随足端落点的振荡噪声 "
                         "(corr=+0.13, 均值 -0.034/步), 不是定向对抗爬升的力量 -> "
                         "本轮默认不改, 保持单变量")
    args = ap.parse_args()

    print("backend:", jax.default_backend())

    # v2 (--sb3_opt): 对齐参考仓库 (SB3 默认) 的优化制度。
    # v1 诊断 (实验记录 §4): 梯度密度 v1=128 次/批 vs SB3=3840 次/rollout,
    # 折算每样本更新次数差 200×。v2 只改优化制度四个参数, 网络结构/环境不动
    # (一次一个变量)。
    sb3_opt = args.sb3_opt
    sb3_net = getattr(args, "sb3_net", False)
    sb3_full = getattr(args, "sb3_full", False)

    # --- v2/v3 分层开关 (保留对照): sb3_opt = 优化制度, sb3_net = 网络 ---
    if sb3_full:
        # v4: 完全仿照参考仓库 (SB3 默认) 的算法参数。
        # SB3 默认: n_envs=12, n_steps=2048 → rollout 24576; batch=64;
        # n_epochs=10 → 每 rollout 3840 次梯度更新 (密度 0.156/样本);
        # gamma=0.99, clip_range=0.2, ent_coef=0, max_grad_norm=0.5,
        # lr=3e-4, MlpPolicy(64,64) tanh, 无 obs 归一化, 高斯无 squash。
        # brax 映射: num_envs=768 × unroll 32 = 24576 (batch_size 按 brax
        # 语义 = 每 minibatch 的 env 数 64; num_minibatches = 24576/64=384);
        # num_updates_per_batch=10 (epochs)。
        # 唯一残余差异: GAE 窗口 32 vs SB3 2048 (8GB 显存装不下 768×2048)。
        #
        # v9 关键修正 (brax 账目): brax 的 batch_size 是"每个 minibatch 的
        # 序列数", 每条序列 unroll_length 步 → minibatch 转移数 =
        # batch_size × unroll_length。v8 用 batch_size=64 → minibatch 2048
        # 条 (SB3 的 32 倍) → 每样本更新密度只有 SB3 的 1/32, 学得慢。
        # 正确映射: batch_size=2 (2 序列 × 32 步 = 64 条 = SB3 的 minibatch
        # 64), num_minibatches=384, updates=10 → 每训练步样本数
        # 24576、梯度更新 3840 次、密度 6.4 样本/更新 —— 与 SB3 完全一致,
        # 且保留 GAE 窗口 32. 总预算 203 训练步 = 5M 环境步 (= 参考仓库)。
        # Go2 追加 (v9 账目推广): num_envs 可覆盖。SB3 的 minibatch 语义是
        # "每批 64 条转移" = batch_size(2 序列)×unroll(32 步), 与 num_envs 无关;
        # 而 "每 rollout 10 个 epoch" 要求 minibatches = rollout/64
        # = num_envs×32/64 = num_envs/2。所以放大 num_envs 只改变每次迭代的
        # 规模 (更少的迭代 × 每次更多的梯度步), 每样本更新密度不变 ——
        # 这正是 SB3 的语义, 同时让 GPU 更吃饱。默认 768 保持 Go1 逐位一致。
        num_envs = int(getattr(args, "sb3_envs", 0) or 768)
        unroll_length = 32
        batch_size = 2
        minibatches = num_envs // batch_size
        updates_per_batch = 10
        entropy_cost = 0.0
        discounting = 0.99
        max_grad_norm = 0.5
        clipping_epsilon = 0.2
        normalize_observations = False
        policy_layers = (64, 64)
        value_layers = (64, 64)
        distribution_type = "normal"
        # brax network_factory 的 activation 是函数不是字符串 (SB3 "tanh")
        activation = __import__("jax").nn.tanh
        print(f"v9 sb3_full: num_envs={num_envs}, unroll={unroll_length}, "
              f"minibatch={batch_size}序列×{unroll_length}步="
              f"{batch_size*unroll_length}条 (=SB3 的 64), "
              f"updates/步={updates_per_batch*minibatches} 次, "
              f"密度 {unroll_length*batch_size/updates_per_batch:.1f} 样本/更新 "
              f"(=SB3 的 6.4)")
    else:
        if sb3_opt:
            minibatches = 256
            updates_per_batch = 8
            entropy_cost = 0.0
            discounting = 0.99
            print("v2 sb3_opt: minibatch=640样本, updates/批=2048, "
                  "entropy=0, discount=0.99")
        else:
            minibatches = 32
            updates_per_batch = 4
            entropy_cost = 1e-2
            discounting = 0.97

        # v3 (--sb3_net): 网络 (64,64) 对齐 SB3 默认 MlpPolicy (~1.5 万参数)。
        # 注意: brax 的 tanh_normal policy 头是 mean/log_std 双头, 输出 24 维
        # (vs SB3 同款), 参数量 ~2 万, 仍与参考仓库同量级。
        if sb3_net:
            policy_layers = (64, 64)
            value_layers = (64, 64)
        else:
            policy_layers = (512, 256, 128)
            value_layers = (512, 256, 128)
        num_envs = args.num_envs
        unroll_length = 20
        # brax 要求 batch_size × num_minibatches 能被 num_envs 整除
        # (train.py:339), 且 minibatch 转移数 = batch_size × unroll_length。
        # 取 batch_size = num_envs // num_minibatches, 使 rollout 恰好等于
        # num_envs × unroll_length, 与官方 locomotion 档语义一致
        # (v1: num_envs=8192, minibatches=32 → batch_size=256, 密度 1280 样本/更新)。
        # 注意默认 num_envs=8192 配 batch_size=64 会让 64×32=2048 不被 8192 整除,
        # brax 直接 assert 失败 —— 这是 v9 加 batch_size 变量时引入的回归。
        assert num_envs % minibatches == 0, (
            f"num_envs={num_envs} 必须能被 num_minibatches={minibatches} 整除")
        batch_size = num_envs // minibatches
        max_grad_norm = 1.0
        clipping_epsilon = 0.3
        normalize_observations = True
        distribution_type = "tanh_normal"
        # brax 的 activation 是函数不是字符串, 传 "silu" 会 TypeError
        activation = jax.nn.silu

    if args.layers:
        policy_layers = tuple(int(x) for x in args.layers.split(","))
        value_layers = policy_layers
    if args.value_layers:
        value_layers = tuple(int(x) for x in args.value_layers.split(","))
        print(f"v28 value_layers={value_layers} (policy={policy_layers})")

    env_cfg = default_config()
    env_cfg.episode_length = args.episode_length
    if args.cmd_forward_prob is not None:
        env_cfg.command_config.forward_prob = float(args.cmd_forward_prob)
        if args.cmd_low_forward is not None:
            env_cfg.command_config.low_forward = [
                float(v) for v in args.cmd_low_forward.split(",")]
        if args.cmd_high_forward is not None:
            env_cfg.command_config.high_forward = [
                float(v) for v in args.cmd_high_forward.split(",")]
        print(f"v33 command mix: forward_prob={env_cfg.command_config.forward_prob} "
              f"forward={env_cfg.command_config.low_forward}..{env_cfg.command_config.high_forward}")
    if args.curriculum:
        env_cfg.curriculum.enable = True
        env_cfg.curriculum.init_level_max = int(args.curriculum_init_level)
        env_cfg.curriculum.success_dist = float(args.curriculum_success_dist)
        env_cfg.curriculum.min_dist = float(args.curriculum_min_dist)
        print(f"v30 curriculum: init_max={args.curriculum_init_level} "
              f"success>{args.curriculum_success_dist}m min<{args.curriculum_min_dist}m")
    if args.naconmax is not None:
        env_cfg.naconmax = int(args.naconmax)
    if args.njmax is not None:
        env_cfg.njmax = int(args.njmax)
    # v5: 物理对齐参考仓库 (硬接触足端 + Newton 100 迭代)
    if getattr(args, "hard_contact", False):
        env_cfg.foot_soft_contact = False
        print(f"v5 hard_contact: 足端硬接触 (XML 原样) + Newton "
              f"{env_cfg.hard_contact_iters} 迭代 / ls "
              f"{env_cfg.hard_contact_ls_iters} (参考仓库物理)")
    # v6: 硬接触 + obs 归一化 (SB3 VecNormalize 缺失件)
    if getattr(args, "norm_obs", False):
        normalize_observations = True
        print("v6 norm_obs: 打开观测运行归一化 (SB3 语义)")

    # v11: 指令条件化 —— 每 episode 采样指令, 一个策略覆盖多方向
    if args.fwd_turn:
        args.sample_cmd = True
        args.cmd_low = args.cmd_low or "0,0,-1"
        args.cmd_high = args.cmd_high or "0.5,0,1"
    if args.sample_cmd:
        env_cfg.command_config.sample = True
        if args.cmd_low:
            env_cfg.command_config.low = [
                float(x) for x in args.cmd_low.split(",")]
        if args.cmd_high:
            env_cfg.command_config.high = [
                float(x) for x in args.cmd_high.split(",")]
        print(f"v11 指令条件化: 每 episode 采样 cmd ~ U("
              f"{list(env_cfg.command_config.low)}, "
              f"{list(env_cfg.command_config.high)})  "
              f"(fixed={list(env_cfg.command_config.fixed)} 仅用于评估/可视化)")
    if args.body_frame:
        env_cfg.command_config.frame = "body"
        print("v14 body_frame: 指令=机体系(朝向)速度, obs 的 base linevel 同步换"
              "机体系 (转向必需)")
    if args.tilt_deg is not None:
        env_cfg.healthy_roll_range = args.tilt_deg
        env_cfg.healthy_pitch_range = args.tilt_deg
        print(f"v15 tilt_deg={args.tilt_deg}: 终止阈值放宽到等效 ±"
              f"{2*args.tilt_deg:.0f}° (参考档 10 → ±20°)")
    # v24: 健康终止 z 基准 (feet = 触地足下方地形, 修 boxes 躯干跨箱顶假终止)
    env_cfg.healthy_z_ref = args.healthy_z_ref
    print(f"v24 healthy_z_ref = {args.healthy_z_ref}")
    if args.dangle_penalty:
        args.natural_gait = True
    # v16: 防打滑/迈步奖励 (playground go1/joystick 配方)
    if args.natural_gait:
        env_cfg.reward_config.scales.feet_slip = -0.5
        env_cfg.reward_config.scales.feet_height = -1.0
        env_cfg.reward_config.air_time_threshold = 0.2
        env_cfg.reward_config.air_time_max = 0.5
        print(f"v16 natural_gait: feet_slip={env_cfg.reward_config.scales.feet_slip}"
              f" (接触相足端速度² 惩罚), "
              f"feet_height={env_cfg.reward_config.scales.feet_height}"
              f" (摆动峰值抬到 {env_cfg.reward_config.max_foot_height}m), "
              f"air_time_threshold={env_cfg.reward_config.air_time_threshold}"
              f" 封顶 {env_cfg.reward_config.air_time_max}s")
    if args.dangle_penalty:
        env_cfg.reward_config.scales.feet_dangle = -0.5
        print(f"v17 dangle_penalty: feet_dangle="
              f"{env_cfg.reward_config.scales.feet_dangle} "
              f"(悬空超过 {env_cfg.reward_config.air_time_limit}s 逐步扣分, "
              f"封顶 2.0)")
    if args.slip_penalty is not None:
        env_cfg.reward_config.scales.feet_slip = args.slip_penalty
        print(f"覆盖 feet_slip={args.slip_penalty}")
    if args.foot_height is not None:
        env_cfg.reward_config.scales.feet_height = args.foot_height
        print(f"覆盖 feet_height={args.foot_height}")
    if args.air_thresh is not None:
        env_cfg.reward_config.air_time_threshold = args.air_thresh
        print(f"覆盖 air_time_threshold={args.air_thresh}")
    if args.dangle_w is not None:
        env_cfg.reward_config.scales.feet_dangle = args.dangle_w
        print(f"覆盖 feet_dangle={args.dangle_w}")

    # --- v19: 地形感知 + 鲁棒性 ---
    if args.height_scan and not args.terrain:
        # 平地档没有 hfield, 采样器不初始化; 高度特征会是常数 0 -> 白占 35 维。
        # 直接报错而不是静默降级, 避免训出一个"以为有地形输入"的假对照。
        raise SystemExit("--height_scan 需要 --terrain (平地上高度特征恒为 0)")
    if args.terrain:
        env_cfg.terrain = True
        # 地形训练必须随机出生: reset 默认把机器人放原点, 而地形在
        # flatten_origin 里把原点周围 0.5m 压平了 -> 8192 环境全在平地,
        # 见不到台阶, 学不会爬梯 (§17.2)。故 --terrain 隐式打开随机出生。
        if not args.no_random_init:
            env_cfg.random_init.enable = True
        if args.random_init_range is not None:
            env_cfg.random_init.xy_range = args.random_init_range
        if args.spawn_stairs_prob is not None:
            env_cfg.random_init.spawn_stairs_prob = float(args.spawn_stairs_prob)
            print(f"v28 spawn_stairs_prob={args.spawn_stairs_prob} "
                  f"(楼梯口出生; 需 terrain json 有 spawn_entries)")
        print(f"v19 terrain: 地形场景 + 随机出生 "
              f"(xy±{env_cfg.random_init.xy_range}m, yaw 随机, "
              f"z=地面+{env_cfg.random_init.spawn_z_offset}m, "
              f"出生速度±{env_cfg.random_init.random_vel})")
        if args.terrain_xml:
            env_cfg.terrain_xml = args.terrain_xml
            print(f"v19 terrain_xml: {args.terrain_xml}")
        if args.boxgrid_json:
            env_cfg.terrain_boxgrid_json = args.boxgrid_json
            env_cfg.contact_mode = args.contact_mode
            print(f"v26 R1: boxgrid_json={args.boxgrid_json}, "
                  f"contact_mode={args.contact_mode}")
        if args.spawn_ring:
            env_cfg.random_init.spawn_ring = True
            print("v26 R1: spawn_ring=True (外圈平地出生, 朝向场地中心)")
        if args.ring_r is not None:
            env_cfg.random_init.ring_r = tuple(args.ring_r)
    if args.no_random_init:
        env_cfg.random_init.enable = False
        print("v19 no_random_init: 全部从原点出生 (调试/基线对照)")
    if args.height_scan:
        env_cfg.height_scan.enable = True
        hs = env_cfg.height_scan
        print(f"v19 height_scan: {hs.nx}x{hs.ny}={hs.nx*hs.ny} 点, "
              f"x∈[{hs.x_min},{hs.x_max}] y∈[{hs.y_min},{hs.y_max}], "
              f"base={hs.base} scale={hs.scale} clip_m={hs.clip_m} "
              f"noise={hs.noise}m (仅 actor)  -> obs 48+{hs.nx*hs.ny}="
              f"{48+hs.nx*hs.ny} (actor) / {119+hs.nx*hs.ny} (critic)")
    if args.scan_noise is not None:
        env_cfg.height_scan.noise = args.scan_noise
        print(f"覆盖 height_scan.noise={args.scan_noise}")
    # v5: 高度扫描几何覆盖 (必须在 height_scan.enable 之后; 不依赖 --scan_noise)
    if args.height_scan and any(a is not None for a in (args.scan_nx, args.scan_ny, args.scan_x, args.scan_y)):
        hs = env_cfg.height_scan
        if args.scan_nx is not None:
            hs.nx = args.scan_nx
        if args.scan_ny is not None:
            hs.ny = args.scan_ny
        if args.scan_x is not None:
            hs.x_min, hs.x_max = args.scan_x
        if args.scan_y is not None:
            hs.y_min, hs.y_max = args.scan_y
        print(f"v5 height_scan 几何: {hs.nx}x{hs.ny}={hs.nx*hs.ny} 点, "
              f"x[{hs.x_min},{hs.x_max}] y[{hs.y_min},{hs.y_max}] "
              f"(间距 {(hs.x_max-hs.x_min)/(hs.nx-1):.3f} x {(hs.y_max-hs.y_min)/(hs.ny-1):.3f} m)"
              f" -> obs 48+{hs.nx*hs.ny}={48+hs.nx*hs.ny} (actor) / {119+hs.nx*hs.ny} (critic)")

    # v27: actor 观测历史帧
    if args.obs_history_frames and args.obs_history_frames > 0:
        env_cfg.obs_history.enable = True
        env_cfg.obs_history.n_frames = int(args.obs_history_frames)
        _n_phase_oh = 8 if (getattr(args, "gait_phase", False)
                            or getattr(args, "gait_w", None) is not None) else 0
        _n_hist = (int(args.obs_history_frames) - 1) * (48 + _n_phase_oh)
        print(f"v27 obs_history: n_frames={args.obs_history_frames}, "
              f"历史缓冲 {_n_hist} 维 ({_n_phase_oh}+48 每帧)")

    # v6: 雷达遮挡模型 (必须在 --height_scan 之后, 且需要 terrain)
    if args.lidar:
        if not args.height_scan:
            raise SystemExit("--lidar 需要 --height_scan (平地上扫描特征恒为 0)")
        ld = env_cfg.height_scan.lidar
        ld.enable = True
        if args.lidar_mount is not None:
            ld.mount = [float(v) for v in args.lidar_mount]
        if args.lidar_vfov is not None:
            ld.vfov = [float(v) for v in args.lidar_vfov]
        if args.lidar_dropout is not None:
            ld.dropout = args.lidar_dropout
        if args.lidar_ray_samples is not None:
            ld.n_ray_samples = args.lidar_ray_samples
        if args.no_lidar_self_occ:
            ld.self_occlusion = False
        if args.lidar_mount_jitter is not None:
            ld.mount_jitter = [args.lidar_mount_jitter] * 3
        if args.lidar_vfov_jitter is not None:
            ld.vfov_jitter = args.lidar_vfov_jitter
        if args.lidar_fill is not None:
            ld.fill = args.lidar_fill
        if args.lidar_occ_recall is not None:
            ld.occ_recall = args.lidar_occ_recall
        if args.lidar_sector_prob is not None:
            ld.sector_prob = args.lidar_sector_prob
        if args.lidar_sector_width is not None:
            ld.sector_width = [float(v) for v in args.lidar_sector_width]
        print(f"v6 lidar 遮挡模型: mount={list(ld.mount)} vfov={list(ld.vfov)} "
              f"ray_samples={ld.n_ray_samples} self_occ={ld.self_occlusion} "
              f"dropout={ld.dropout} occ_recall={ld.occ_recall} "
              f"sector={ld.sector_prob}/{list(ld.sector_width)} "
              f"jit_mount={list(ld.mount_jitter)} jit_vfov={ld.vfov_jitter} "
              f"fill={ld.fill} -> actor scan 有盲区/空洞, critic 仍为完美扫描")

    if args.push:
        env_cfg.push_config.enable = True
        if args.push_vel is not None:
            env_cfg.push_config.max_vel = args.push_vel
        if args.push_interval is not None:
            env_cfg.push_config.interval = args.push_interval
        print(f"v19 push: 每 {env_cfg.push_config.interval} 步推一次, "
              f"水平速度冲量 ±{env_cfg.push_config.max_vel} m/s")
    if args.obs_noise:
        env_cfg.obs_noise.enable = True
        on = env_cfg.obs_noise
        print(f"v19 obs_noise: lin_vel±{on.lin_vel} ang_vel±{on.ang_vel} "
              f"gravity±{on.gravity} dof_pos±{on.dof_pos} "
              f"dof_vel±{on.dof_vel} (仅 actor; critic 无噪)")
        if args.obs_noise_linvel is not None:
            env_cfg.obs_noise.lin_vel = args.obs_noise_linvel
            print(f"v3 obs_noise.lin_vel = {args.obs_noise_linvel} m/s "
                  f"(覆盖默认 0.05)")
        # v3: 速度估计误差模型 (独立于 --obs_noise; 但它本来就该配合打开)
        if args.obs_vel_bias is not None:
            env_cfg.obs_noise.lin_vel_bias = args.obs_vel_bias
        if args.obs_vel_drift is not None:
            env_cfg.obs_noise.lin_vel_drift_sigma = args.obs_vel_drift
        if args.obs_vel_tau is not None:
            env_cfg.obs_noise.lin_vel_drift_tau = args.obs_vel_tau
        if args.obs_vel_delay is not None:
            env_cfg.obs_noise.lin_vel_delay_steps = args.obs_vel_delay
        _vb = float(env_cfg.obs_noise.lin_vel_bias)
        _vd = float(env_cfg.obs_noise.lin_vel_drift_sigma)
        _vl = int(env_cfg.obs_noise.lin_vel_delay_steps)
        if _vb > 0.0 or _vd > 0.0 or _vl > 0:
            print(f"v3 速度估计误差: bias±{_vb} m/s, OU std={_vd} m/s "
                  f"tau={env_cfg.obs_noise.lin_vel_drift_tau}s, "
                  f"delay 0..{_vl} 步 ({_vl*20}ms)  (仅 actor obs 0:3)")

    # --- v19: 显式爬升激励 + 下降引导 ---
    # 权重来自实测标定 (sim/probe_v19_reward_scale.py(已删除, 结论见实验记录)), 不是猜的:
    #   base_height 是**平方**误差 -> 线性直觉会差一个数量级。
    #   实测正常站立时平方误差 p50=1.9e-5 (≈免费), 摔倒/拖地时才大。
    #   -200 下: 偏差 2cm=-0.08, 5cm=-0.5, 10cm=-2.0; 每步均值 -0.88
    #   (-2000 会给到每步 -8.8, 是 v18 每步奖励 2.46 的 3.6 倍 -> 会把总奖励
    #    打到 0, PPO 无梯度, 正是 v1-v7 的失败模式, 所以不用)
    if args.climb_reward:
        env_cfg.reward_config.scales.base_height = -200.0
    if args.base_height_w is not None:
        env_cfg.reward_config.scales.base_height = args.base_height_w
        args.climb_reward = True
    if env_cfg.reward_config.scales.get("base_height", 0.0) != 0.0:
        print(f"v19 climb_reward: base_height="
              f"{env_cfg.reward_config.scales.base_height} "
              f"(相对足下地形, 目标站高 "
              f"{env_cfg.reward_config.base_height_target}m; **平方**误差: "
              f"偏差 2cm=-0.08, 5cm=-0.5, 10cm=-2.0; "
              f"正常站立实测每步≈-0.0005)")
    if args.soft_landing:
        env_cfg.reward_config.scales.feet_impact = -2.0
    if args.impact_w is not None:
        env_cfg.reward_config.scales.feet_impact = args.impact_w
    if args.impact_vel is not None:
        env_cfg.reward_config.impact_vel_limit = args.impact_vel
    if env_cfg.reward_config.scales.get("feet_impact", 0.0) != 0.0:
        _iw = env_cfg.reward_config.scales.feet_impact
        _lim = env_cfg.reward_config.impact_vel_limit
        print(f"v19 soft_landing: feet_impact={_iw}, 阈值 {_lim} m/s "
              f"(落地 1.0m/s={_iw*(1.0-_lim)**2:.2f}, "
              f"1.5m/s={_iw*(1.5-_lim)**2:.2f}, "
              f"2.0m/s={_iw*(2.0-_lim)**2:.2f} /足; "
              f"实测正常步态 95% 的时刻不触发)")
        print(f"v20 注: feet_impact 已改为**按时间结算** (净空 < "
              f"{env_cfg.reward_config.impact_height}m 时每步考核向下速度), "
              f"不再是'只在落地帧' —— 修 §18.10.3 的激励漏洞")
        print("v20 注: feet_height 已改为**摆动相均值**结算 (同源漏洞)")

    # --- v20: 对角步态相位引导 ---
    if args.gait_phase:
        env_cfg.reward_config.scales.feet_phase = 2.0
    if args.gait_w is not None:
        env_cfg.reward_config.scales.feet_phase = args.gait_w
    if args.gait_freq is not None:
        env_cfg.reward_config.gait_freq = args.gait_freq
    if args.gait_freq_scaled:
        env_cfg.reward_config.gait_freq_scaled_by_cmd = True
    if args.swing_height is not None:
        env_cfg.reward_config.gait_swing_height = float(args.swing_height)
        print(f"v25 R2: gait_swing_height={args.swing_height} "
              f"(默认 0.15; max_foot_height 会在下面联动)")
    if args.feet_edge_w is not None:
        env_cfg.reward_config.scales.feet_edge = float(args.feet_edge_w)
        print(f"v25 R3: feet_edge={args.feet_edge_w} "
              f"(触地足落在地形棱上的惩罚; 阈值 "
              f"{env_cfg.reward_config.feet_edge_thresh}m)")
    if env_cfg.reward_config.scales.get("feet_phase", 0.0) != 0.0:
        rc = env_cfg.reward_config
        # **必须联动**: feet_phase 要求抬到 gait_swing_height, feet_height 罚
        # 偏离 max_foot_height。两者不一致时策略被两个目标撕扯, 谁都不满足。
        if not args.keep_foot_height:
            rc.max_foot_height = rc.gait_swing_height
        print(f"v20 gait_phase: feet_phase={rc.scales.feet_phase} "
              f"(对角 trot 相位 [0,pi,pi,0], 步频 {rc.gait_freq}Hz"
              f"{', 随指令缩放' if rc.gait_freq_scaled_by_cmd else ''}, "
              f"swing_height={rc.gait_swing_height}, sigma={rc.gait_phase_sigma})")
        print(f"     奖励 = exp(-Σ(足端净空 - rz(相位))²/{rc.gait_phase_sigma})")
        print(f"     联动: max_foot_height={rc.max_foot_height} "
              f"(feet_height 的抬脚目标, 与 swing_height 一致"
              f"{'' if not args.keep_foot_height else ' —— 被 --keep_foot_height 关掉'})")
        print(f"     区分度自检: 理想trot=1.0000, 全贴地≈0.34, 全悬≈0.19 "
              f"(gap≈0.66; 若 gap<0.15 奖励面太平, 推不动步态)")

    # --- v22: 显式爬升激励 (修 §24 的激励缺口) ---
    if args.ascent:
        env_cfg.reward_config.scales.ascent = 2.0
    if args.ascent_w is not None:
        env_cfg.reward_config.scales.ascent = args.ascent_w
    if args.ascent_cap is not None:
        env_cfg.reward_config.ascent_cap = args.ascent_cap
    if args.ascent_gate is not None:
        env_cfg.reward_config.ascent_gate = args.ascent_gate
    if env_cfg.reward_config.scales.get("ascent", 0.0) != 0.0:
        rc = env_cfg.reward_config
        if not args.terrain:
            raise SystemExit(
                "--ascent 需要 --terrain (平地上地形高度恒为 0, 该项无信息)")
        print(f"v22 ascent: ascent={rc.scales.ascent} "
              f"(每步 += W * clip(h_now - h_spawn, 0, {rc.ascent_cap}))")
        print(f"     为什么需要它: 原奖励面**没有任何一项**随「站得更高」增加 —— "
              f"base_height 是相对量 (躯干离足下地面多高), 地形上爬升不改变它; "
              f"速度跟踪/相位/各项 cost 都与地形高度无关。")
        print(f"     -> 绕开楼梯与爬上去收益**完全相同**, 而绕行更容易 -> 策略学会绕行。")
        print(f"     实测依据: 把 v21 策略放在楼梯脚下正对楼梯, 走 2.0m 却横向漂移"
              f" 1.4m 绕开, 最大爬升 4.7cm (sim/diag_v22_climb_test.py)")
        print(f"     标定 (sim/probe_v22_reward_terms.py 实测每步均值): "
              f"linear_vel_tracking=+1.76, feet_phase=+1.59, base_height=-0.30")
        _gate = env_cfg.reward_config.ascent_gate
        if _gate > 0.0:
            print(f"     v23 行进门控: ascent *= clip(|v_body|/{_gate},0,1) -> "
                  f"静止/站台顶不动 = 0 (站桩套利消失), 故 W 可提到 "
                  f"{env_cfg.reward_config.scales.ascent}; "
                  f"无门控时 W>=4 就有站桩风险 "
                  f"(走路 +1.776 vs 站台顶 0.347+W*0.30)")
        else:
            print(f"     反套利上界 (无门控 = v22 行为): 走路每步 +1.776, "
                  f"站着不动 +0.347, 站台顶=0.347+W*0.30 -> W=4 给 +1.547 "
                  f"(88%, 险); W>=8 超过走路 -> 会退化成站桩")
        print(f"     信号稀疏性: 升高 >2cm 只占 1.8% 的步 (>10cm 占 0.0%) -> "
              f"每步均值小, 但梯度方向明确")
    # base_height 参考点: 默认 mean5 = 与 v21 完全一致 -> v22 是单变量实验。
    # trunk 仅供单独对照 (实测它**不是**爬升失败的原因, 见 _cost_base_height)。
    if args.trunk_height_ref:
        env_cfg.reward_config.base_height_ref = "trunk"
        print("v22 base_height_ref=trunk (**实验性对照, 非默认**): 参考面 = 躯干"
              "正下方一点。实测量化: mean5 的额外惩罚是随足端落点的振荡噪声 "
              "(corr=+0.13, 均值 -0.034/步, 峰值 -0.083 = 走路 4.7%), 不是定向"
              "对抗爬升的力量 —— 故本轮默认不改它, 只加 ascent (单变量)")
    else:
        print("v22 base_height_ref=mean5 (默认): 参考面 = 躯干+4足 5 点均值, "
              "**与 v21 一致** -> 本轮唯一变量是 ascent")

    # --- v21: 非足端部件撞地惩罚 (修 §20 "后腿关节着地") ---
    # --- v23: 左右镜像对称正则 ---
    if args.symmetry:
        env_cfg.reward_config.scales.symmetry = -0.4
    if args.symmetry_w is not None:
        env_cfg.reward_config.scales.symmetry = args.symmetry_w
    if args.symmetry_beta is not None:
        env_cfg.reward_config.symmetry_beta = args.symmetry_beta
    if args.symmetry_cap is not None:
        env_cfg.reward_config.symmetry_cap = args.symmetry_cap
    if env_cfg.reward_config.scales.get("symmetry", 0.0) != 0.0:
        print(f"v23 symmetry: symmetry={env_cfg.reward_config.scales.symmetry} "
              f"beta={env_cfg.reward_config.symmetry_beta} "
              f"cap={env_cfg.reward_config.symmetry_cap} "
              f"(每步 -= W * EMA(镜像偏差), EMA 时间常数 "
              f"{1.0/env_cfg.reward_config.symmetry_beta:.0f} 步)")
    if args.body_contact:
        env_cfg.reward_config.scales.body_contact = -150.0
    if args.body_contact_w is not None:
        env_cfg.reward_config.scales.body_contact = args.body_contact_w
    if args.body_contact_max is not None:
        env_cfg.reward_config.body_contact_max = args.body_contact_max
    if env_cfg.reward_config.scales.get("body_contact", 0.0) != 0.0:
        _bw = env_cfg.reward_config.scales.body_contact
        _bm = env_cfg.reward_config.body_contact_margin
        _bx = env_cfg.reward_config.body_contact_max
        print(f"v21 body_contact: body_contact={_bw} "
              f"(非足端碰撞体 = 小腿+大腿+躯干, 共 28 个几何)")
        print(f"     形式 = -{abs(_bw)} * min(Σ relu(margin−净空), {_bx}m), "
              f"margin={_bm}m")
        print(f"     margin=0 的含义 (**实测定标**): 平地沉降后的标称站姿, "
              f"28 个几何净空全为正 (最小 +2.2mm)")
        print(f"       -> 正常姿态惩罚严格=0, 不会推歪姿势; 只有真穿透才扣分")
        print(f"     标定依据 (sim/calib_penalty_form.py(已删除, 结论见实验记录), MuJoCo 接触判定):")
        print(f"       大腿区触地率  v18(好)= 0.0%  v20(坏)= 6.6%  <- 干净判据")
        print(f"       后腿穿透深度  v18=0.00055m  v20=0.00305m  (5.6x)")
        print(f"       选线性不选平方: 区分度 4.07x vs 2.22x, 且浅穿透有梯度")
        print(f"     上限 {_bx}m: 深穿透封顶, 避免惩罚压过正奖励把总奖励打 0")
        print(f"       (总奖励 clip(·,0,·) -> 梯度归零, 是 v1-v7 的失败模式)")

    # --- v28: 步态自然性 / 机体稳定性 (用户 2026-10-21 反馈: 四髋摆动极大、走路晃) ---
    if args.hip_abduction:
        env_cfg.reward_config.scales.hip_abduction = -0.5
    if args.hip_abduction_w is not None:
        env_cfg.reward_config.scales.hip_abduction = args.hip_abduction_w
    if args.hip_action_w is not None:
        env_cfg.reward_config.scales.hip_action = args.hip_action_w
    if args.action_rate_w is not None:
        env_cfg.reward_config.scales.action_rate = args.action_rate_w
    if args.action_acc_w is not None:
        env_cfg.reward_config.scales.action_acc = args.action_acc_w
    if args.terminate_body_pen is not None:
        env_cfg.terminate_body_pen = float(args.terminate_body_pen)
        print(f"v29 terminate_body_pen={args.terminate_body_pen}m "
              f"(非足端净空 < -该值 -> 终止; PIE illegal_contact 等价)")
    if args.xy_angvel_w is not None:
        env_cfg.reward_config.scales.xy_angular_vel = args.xy_angvel_w
    if args.orientation_w is not None:
        env_cfg.reward_config.scales.orientation = args.orientation_w
    if args.torque_w is not None:
        env_cfg.reward_config.scales.torque = args.torque_w
    if args.joint_acc_w is not None:
        env_cfg.reward_config.scales.joint_acceleration = args.joint_acc_w
    if args.clearance_w is not None:
        env_cfg.reward_config.scales.feet_clearance = args.clearance_w
    if args.clearance_target is not None:
        env_cfg.reward_config.feet_clearance_target = args.clearance_target
    if args.progress_w is not None:
        env_cfg.reward_config.scales.forward_progress = args.progress_w
    if args.lazy_stop_w is not None:
        env_cfg.reward_config.scales.lazy_stop = args.lazy_stop_w
    if args.termination_w is not None:
        env_cfg.reward_config.scales.termination = args.termination_w
    if args.reward_clip_min is not None:
        env_cfg.reward_clip_min = float(args.reward_clip_min)
    _hipa = float(env_cfg.reward_config.scales.get("hip_action", 0.0))
    if _hipa != 0.0:
        print(f"v28b hip_action: {_hipa} (四髋动作目标偏离 home 平方和; "
              f"cap={env_cfg.reward_config.get('hip_action_cap', 1.2)}) "
              f"-> 现状 Σ≈0.6 -> {_hipa*0.6:+.2f}/步 (直接梯度, 不经动力学)")
    _hipw = float(env_cfg.reward_config.scales.get("hip_abduction", 0.0))
    if args.feet_edge_bg:
        env_cfg.reward_config.feet_edge_use_boxgrid = True
        print("v28 feet_edge_bg: 棱掩码叠加 boxgrid 高度表 (宽踏面地形才开)")
    if _hipw != 0.0:
        print(f"v28 hip_abduction: {_hipw} (四髋偏离 home 平方和, 返回>=0, 权重为负)")
        print(f"     动机: 实测 hip_q mean 29-33°/p95 42-47°, hip 动作 p95 1.2-1.5rad "
              f"顶到 ±1.047 限位; v20/v24/v25a/v25b 四个检查点完全一致 -> 奖励缺项")
        print(f"     量级: 现状 Σ(q_hip-home)²≈0.7 -> {_hipw*0.7:+.2f}/步 "
              f"(正奖励 ~+2.3/步); 静止标称姿态严格为 0 (home 髋角 ∓0.1rad)")
        print(f"     反套利: 权重过大会把总奖励压到 0 (PPO 梯度归零, v1-v7 失败模式)")
    print(f"v28 reward 覆盖: action_rate={env_cfg.reward_config.scales.action_rate} "
          f"xy_angular_vel={env_cfg.reward_config.scales.xy_angular_vel} "
          f"orientation={env_cfg.reward_config.scales.orientation} "
          f"torque={env_cfg.reward_config.scales.torque} "
          f"joint_acc={env_cfg.reward_config.scales.joint_acceleration} "
          f"feet_edge_thresh={env_cfg.reward_config.get('feet_edge_thresh', 0.08)}")

    # 地形/高度扫描会改 obs 维度 -> 网络输入层形状必须随之改变。brax 从
    # env.observation_size 自动推断 (mjx_env.py 用 jax.eval_shape(reset)),
    # 这里只做一次显式核对, 避免"以为改了其实没生效"。
    if (args.progress_w or args.lazy_stop_w or args.termination_w
            or args.reward_clip_min is not None or args.clearance_w is not None):
        print(f"v29 stairs signals: progress={env_cfg.reward_config.scales.get('forward_progress')} "
              f"lazy_stop={env_cfg.reward_config.scales.get('lazy_stop')} "
              f"termination={env_cfg.reward_config.scales.get('termination')} "
              f"clearance={env_cfg.reward_config.scales.get('feet_clearance')}"
              f"/{env_cfg.reward_config.get('feet_clearance_target')} "
              f"reward_clip_min={env_cfg.reward_clip_min}")
    _probe = env_cfg.height_scan.enable
    _n_scan = env_cfg.height_scan.nx * env_cfg.height_scan.ny if _probe else 0
    # v20: 相位 (cos,sin) x 4 足 = 8 维, 仅在开相位奖励时进 obs
    _n_phase = 8 if env_cfg.reward_config.scales.get("feet_phase", 0.0) != 0.0 else 0
    _n_hist_exp = int(getattr(args, "obs_history_frames", 0) or 0)
    _n_hist_exp = max(0, _n_hist_exp - 1) * (48 + _n_phase)
    print(f"预期 obs 维度: state={48 + _n_scan + _n_phase + _n_hist_exp}  "
          f"privileged_state={119 + _n_scan + _n_phase}"
          f"{'  (含相位 8 维)' if _n_phase else ''}")

    _lr = args.lr if args.lr is not None else 3e-4
    _ent = args.entropy if args.entropy is not None else entropy_cost
    print(f"优化: lr={_lr} entropy_cost={_ent} clip={clipping_epsilon} "
          f"grad_clip={max_grad_norm} layers={policy_layers} "
          f"restore={args.restore}")

    logdir = args.logdir or os.path.join(LOG_DIR, "walk")
    # orbax 保存要求绝对路径 (相对路径在首次 checkpoint 时抛
    # "Checkpoint path should be absolute")
    logdir = os.path.abspath(logdir)
    ckpt_path = os.path.join(logdir, "checkpoints")

    # --- 启动前安全检查 (放在 env/locomotion.load 之前: 拒绝时不该已占 GPU) ---
    _check_ckpt_overwrite(ckpt_path, args.allow_overwrite)
    _check_restore(args.restore, args.step_offset,
                   48 + _n_scan + _n_phase + _n_hist_exp,
                   119 + _n_scan + _n_phase)

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    env = locomotion.load("Go2Walk", config=env_cfg)
    print(f"env: {type(env).__name__} action_size={env.action_size}")

    ppo_params = config_dict.create(
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1.0,
        episode_length=args.episode_length,
        normalize_observations=normalize_observations,
        action_repeat=1,
        unroll_length=unroll_length,
        num_minibatches=minibatches,
        num_updates_per_batch=updates_per_batch,
        discounting=discounting,
        learning_rate=args.lr if args.lr is not None else 3e-4,
        entropy_cost=(args.entropy if args.entropy is not None
                      else entropy_cost),
        num_envs=num_envs,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        clipping_epsilon=clipping_epsilon,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=policy_layers,
            value_hidden_layer_sizes=value_layers,
            policy_obs_key="state",
            value_obs_key="privileged_state",
            distribution_type=distribution_type,
            activation=activation,
        ),
    )

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )

    if args.curriculum:
        from sim.brax_curriculum import wrap_for_curriculum_training
        _wrap_fn = functools.partial(
            wrap_for_curriculum_training,
            init_level_max=int(args.curriculum_init_level),
            success_dist=float(args.curriculum_success_dist),
            min_dist=float(args.curriculum_min_dist),
            max_level=int(getattr(env, "_cur_max", 5)))
    else:
        _wrap_fn = functools.partial(
            wrapper.wrap_for_brax_training, full_reset=bool(args.full_reset))
    print(f"v29 reset: full_reset={bool(args.full_reset)} "
          f"num_resets_per_eval={args.num_resets_per_eval}")
    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        num_resets_per_eval=int(args.num_resets_per_eval),
        seed=args.seed,
        save_checkpoint_path=ckpt_path,
        restore_checkpoint_path=args.restore,
        wrap_env_fn=_wrap_fn,
        num_eval_envs=128,
    )

    times = [time.monotonic()]

    # eval_reward 最高的 checkpoint → checkpoints/best/
    best_reward = -float("inf")
    best_step = 0
    best_ckpt_dir = os.path.join(ckpt_path, "best")

    def _m(metrics, *names):
        for n in names:
            if n in metrics:
                return float(metrics[n])
        return float("nan")

    # 续训时 num_steps 从 0 重新计数, 日志里同时打印累计步数 (step_offset + 
    # num_steps) 和相对步数, 避免把 25M 读成 5M (§22 的坑)。
    _off = args.step_offset

    def progress(num_steps, metrics):
        nonlocal best_reward, best_step
        times.append(time.monotonic())
        fps = num_steps / (times[-1] - times[0])
        loss = _m(metrics, "losses/total_loss", "total_loss")
        ent = _m(metrics, "losses/entropy", "entropy")
        pl = _m(metrics, "losses/policy_loss", "policy_loss")
        vl = _m(metrics, "losses/value_loss", "value_loss")
        tag = f"{num_steps}(累计{_off + num_steps})" if _off else f"{num_steps}"
        if "eval/episode_reward" in metrics:
            r = float(metrics["eval/episode_reward"])
            if r > best_reward:
                best_reward = r
                best_step = num_steps
                src = os.path.join(ckpt_path, f"{num_steps:012d}")
                if os.path.isdir(src):
                    if os.path.isdir(best_ckpt_dir):
                        shutil.rmtree(best_ckpt_dir)
                    shutil.copytree(src, best_ckpt_dir)
                    print(f"  [best] eval_reward={r:.3f} @ {tag} → best/",
                          flush=True)
            # v6: 雷达遮挡统计 (eval/episode_obs/scan_vis|scan_info 是**整回合求和**,
            # 除以 ep_len 才是"平均每步的可见格点比例")
            _sv = _m(metrics, "eval/episode_obs/scan_vis")
            _si = _m(metrics, "eval/episode_obs/scan_info")
            _el = metrics.get("eval/avg_episode_length", float("nan"))
            _lidar = (f" vis={_sv / max(_el, 1.0):.2f} info={_si / max(_el, 1.0):.2f}"
                      if _sv == _sv else "")
            print(f"{tag}: eval_reward={r:.3f} "
                  f"ep_len={metrics.get('eval/avg_episode_length', float('nan')):.1f} "
                  f"| loss={loss:.4f} pi={pl:.4f} vf={vl:.4f} ent={ent:.2f} "
                  f"fps={fps:.0f}{_lidar}", flush=True)
        else:
            print(f"{num_steps}: loss={loss:.4f} pi={pl:.4f} vf={vl:.4f} "
                  f"ent={ent:.2f} fps={fps:.0f}", flush=True)

    print("开始训练...")
    make_inference_fn, params, _ = train_fn(
        environment=env, progress_fn=progress
    )
    print("训练完成")

    import pickle
    save_path = os.path.join(POLICY_DIR, args.save_name)
    with open(save_path, "wb") as f:
        pickle.dump(params, f)
    print(f"已保存策略: {save_path}")
    print(f"best eval_reward={best_reward:.3f} @ {best_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())