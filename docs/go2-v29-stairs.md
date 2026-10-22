# Go2 v29: 台阶能力专项 —— 开源调研与修复实验 (2026-10-21 早)

用户要求: 专注解决 (1) 上台阶成功率低; (2) 策略绕开台阶 (reward hacking)。并调研
开源实现怎么做。本文件记录调研结论、我们 v28 失败的补查根因、v29 的改动与实验。

## 0. TL;DR

开源 (legged_gym / IsaacLab / mjlab / mjlab-PIE / MuJoCo Playground) 的台阶能力做法:

1. **架构与我们的想象不同**: 它们也是"**一张共享地形 + 每个 env 一个 spawn origin**
   指向地形里的一块瓦片"，并不是每 env 一份独立地形。课程 = 重算 origin。
   我们的"单张共享地形"架构与开源同构，缺的是瓦片难度梯度 + origin 课程簿记。
2. **台阶参数**: legged_gym 台阶 riser = 0.05 + 0.18*difficulty (10 行, 0.05..0.23m),
   step_width 0.31m; mjlab/PIE 的 Go2 台阶: step_height_range=(0.03, 0.25),
   step_width=0.30, 10 行 x 20 列; mjlab 台阶专项集从 **0.02m** 起
   (easy 0.02-0.05 / moderate 0.05-0.08 / hard 0.08-0.10)。官方 Go2 的公开能力上限
   约 0.10m (37% 站高)，我们 v28 直接上 0.06~0.22 是难度错位。
3. **课程晋升公式** (legged_gym legged_robot.py): 位移 > 瓦片长/2 升一级;
   位移 < 0.5*指令速度*episode 时长 降一级; 满级随机回落。mjlab 改进: 用
   commanded distance*0.8、最小 1m 防漂移、早摔补时、首轮冻结。
4. **防绕行**: 每 env 出生在瓦片中心/入口、朝向固定; 台阶地形把可走面积填满;
   命令是前向偏置 (PIE obstacle 地形: 80% 直行/20% 小 yaw, vx 0.2-1.5);
   另有 termination 惩罚 / lazy_stop 防站桩。我们的 v28 缺这些。
5. **奖励** (PIE Go2 stairs 完整配置, 源码): track body planar vel 1.5 (sigma 0.25),
   track yaw 0.5, lin_vel_z -1, ang_vel_xy -0.05, **orientation -5**,
   joint_acc -2.5e-7, power -2e-5, **collision -10** (非足端接触力>0.1 计数),
   action_rate -0.01, **smoothness(action_acc) -0.01**, joint_pos_limits -10,
   **foot_gait +0.5**, **foot_slip -0.25**, **hip_deviation_l2 -0.5**,
   **base_height +0.2** (target 0.27, 支撑足力门控); 终止: bad_orientation 55 度、
   **illegal_contact (非足端接触力>10N) 终止**。**没有 ascent 奖励** —— 爬升由
   "前向命令 + 地形课程"逼出来。
6. **抬腿高度**: 跨过 h=0.06m riser 需要足心相对下一级踏面 >=0.082m (足半径 0.022m)。
   我们 v28 的 feet_height 是"摆动相平均净空、落地帧结算"，feet_phase 的固定 trot
   轨迹又在落地前把脚拉回 0 -> 均值达标但跨棱瞬间不够。mjlab/unitree 用**每步、
   速度加权、相对地形**的 feet_clearance (-1.0, target 0.10); Barkour 注释明确
   "feet_air_time does not encourage high clearances"。

**我们 v28 失败的补查根因 (新增, 本轮实测/读码)**:
* brax/MuJoCo Playground wrapper 默认 full_reset=False: **每个 env 整个训练只回放
  训练开始时缓存的同一个初始状态** (playground/_src/wrapper.py L121-169);
  brax num_resets_per_eval 默认 0。spawn_entries 只有 14 个点时，等于只有 14 种出生。
* bands2 的 box 场 ±4.4m 但 hfield ±6m -> **1.6m 宽、占 46% 面积的平地外环**，
  spawn_entries (x=±4.6) 正好在环上; 转身即可在平地上"前进"。
* 命令是机体系、wz∈[-1.5,1.5] -> 转身面朝下坡走也算完成速度指令。
* 终止仅等效 ±35°，termination 权重 0，无 lazy_stop -> 台阶前停 1.5s 是理性解。
* CPU 实测 (v25b, bands_u006 统一 0.06m, 8s): 只前进 0.61m (指令 3.2m), 横向漂移
  0.80m, max|yaw| 67 度, 爬升 0.038m -> "斜着找路 + 抬腿不够"同时存在。

## 1. 开源依据 (源码/文档 URL)

| 来源 | 关键内容 |
|---|---|
| legged_gym utils/terrain.py | stairs: step_height=0.05+0.18*difficulty, step_width=0.31, proportions 台阶上/下 0.35/0.25; 每 tile 8x8m, 10 行 x 20 列 |
| legged_gym envs/base/legged_robot.py | _update_terrain_curriculum: 位移>tile/2 升, <0.5*cmd*T 降, 满级随机回落; reset 时重算 env_origins |
| legged_gym envs/base/legged_robot_config.py | terrain_proportions 与 curriculum/max_init_terrain_level |
| IsaacLab terrains/terrain_generator.py | difficulty = (row+eta)/num_rows*(upper-lower)+lower, 每行难度线性 |
| parkour_mjlab (Go2 PIE) src/tasks/pie/terrains.py | num_rows=10, num_cols=20, stairs_up/down step_height_range=(0.03,0.25), step_width=0.30, proportion 45%/45%+flat 10% |
| parkour_mjlab .../config/go2/env_cfgs.py | 上面第 5 条的完整奖励表 + 终止条件 + TerrainAwareVelocityCommand (obstacle yaw 80% 直行, vx 0.2-1.5) + 20s episode + 4096 env |
| parkour_mjlab .../mdp/curriculums.py | terrain_levels_vel: distance>size/2 且未终止升级; 终止或进度不足一半降级 |
| PIE paper arXiv 2408.13740 | 隐式-显式学习的视觉跑酷; 台阶/障碍课程 + foot clearance 目标辅助监督 |
| mujoco_playground _src/wrapper.py L86-134 | full_reset 语义: 默认缓存 first_state 回放; full_reset=True 才每回合真 reset (并保留 AutoResetWrapper_preserve_info 供课程) |
| mjlab 文档 terrains | 地形生成器/terrain entity 与课程接口 |
| extreme-parkour (a1_parkour_config) | 台阶/跑酷奖励与地形配置 (由 R2 报告汇总) |

## 2. v29 已实施改动

训练栈: mjx warp + brax PPO, 单卡 RTX 5060 8GB。

(a) **每回合真重采样**: train_go2.py 新增 --full_reset / --num_resets_per_eval,
    run 脚本 FULL_RESET=1 / RESETS_PER_EVAL=4。代价实测: fps 3300-4300 -> **1412**
    (每步要向量化算所有 env 的 reset)，但换来每 episode 新出生/新随机。
(b) **地形**: 新 sim/gen_terrain_stairs_wide.py —— 一条**通宽楼梯**:
    入口平道 3 格 (1.2m, 助跑) -> 上行 6 级 -> 顶部平台 8 格 -> 下行 5 级, 全程通宽,
    两端无平走廊; **hfield ext = field = 4.4m (无外环, 掉了就是掉下去)**;
    spawn 在入口平道内 (x=-3.8,y=0,yaw=0, 朝向 +x), 0.6-0.75m 助跑。
    6 个难度阶段: riser 0.02 / 0.03 / 0.05 / 0.08 / 0.11 / 0.14 m
    (对应 climb 0.12 / 0.18 / 0.30 / 0.48 / 0.66 / 0.84 m)。
(c) **命令**: 前向偏置, 禁止掉头: cmd vx∈[0.3,0.8], vy=0, wz∈[0,0.2]
    (PIE obstacle: 80% 直行/20% 小 yaw)。
(d) **奖励/终止** (envs/go2_walk.py, 全部默认 0 关闭, CLI 打开):
    forward_progress +1.5 (世界系 +x 速度裁剪到 [0,cmd_x], 抗绕行/抗横移);
    lazy_stop -1.5 (cmd>0.1 且 |v|<0.2 时扣 clip(0.2-|v|,0,0.2));
    termination -20 (需 reward_clip_min=-25 才不被 clip 掉);
    orientation -5 (PIE 同款); action_acc -0.01 (二阶平滑); feet_phase 2.0->0.5;
    hip_abduction -0.3 / hip_action -0.3 (台阶阶段先放松髋约束);
    terminate_body_pen=0.02 (非足端净空 < -2cm 即终止, PIE illegal_contact 几何等价)。
(e) 评测: sim/eval_stairs_climb.py 输出 max 爬升/p50/p90/分级比例/vx (存活率不能
    当台阶指标, 站着不动也 100% 存活)。

## 3. 实验计划 (GPU 串行)

| 阶段 | 地形 (riser) | 起点 | 步数/lr | 目的 |
|---|---|---|---|---|
| v29a-s0 | wide_h020 (0.02) | v25b | 2M / 2e-5 | 学会抬脚跨第一级/多级 (0.12m) |
| v29a-s1 | wide_h030 (0.03) | s0 best | 1.5M | 泛化到 0.18m |
| v29a-s2 | wide_h050 (0.05) | s1 best | 1.5M | 0.30m |
| v29a-s3 | wide_h080 (0.08) | s2 best | 1.5M | 0.48m (逼近能力上限) |
| v29a-s4 | wide_h110/140 | s3 best | 1.5M | 极限档 |
| v29b-cons | hyb2 (混合) | 最好的台阶模型 | 2M / 1.5e-5 | 保住 hyb2/stg1/部署能力 |

验收 (R2 建议): 0.06m 级: >=80% episode 至少上 1 级 (climb>=0.06), >=50% 上 2 级
(>=0.12), >=20% 到顶; 同时 CPU 平地 100% 存活、髋指标不明显变差、hyb2 不退化。


* **v29b-lanes 结果 (4.19M 步, 09:47 完成)**: eval_reward -535 -> -408 -> -352 -> -313,
  ep_len 481-553 (隔墙/高难度导致终止偏多)。爬升阶梯对比:

  | 地形 | v25b | v29s2 | v29b-lanes |
  |---|---|---|---|
  | wide_h030 (0.03) climb / 上1级 / 上2级 | 0.033 / 67% / 4.0% | 0.040 / 90% / 6.5% | 0.041 / 91% / 9.6% |
  | wide_h050 (0.05) climb / 上1级 / 存活 | 0.036 / 29% / 88% | 0.043 / 48% / 63% | 0.044 / 47% / **79%** |
  | wide_h080 (0.08) climb | 0.016 | 0.018 | 0.021 |
  | wide_h110 (0.11) climb | 0.010 | 0.012 | 0.014 |

  -> 混合难度赛道把 0.03 的收益保住 (上 1 级 91%, 上 2 级 9.6%), 0.05 存活率从 63%
  回到 79%; 但 0.08/0.11 仍基本爬不上 (climb 0.02 左右, 只相当于 1/4 级) —— 与外部
  证据一致: Go2 公开能力上限约 0.10m, 0.14m 已是 52% 站高, 需更长训练/轨迹先验。
* **v29c-cons (hyb2 巩固, 09:50 启动)**: 从 v29b-lanes best 在 hyb2 混合地形上
  2.5M/lr1.5e-5, 命令与 v25b 分布一致 (ring 出生), 保留 lazy_stop/termination/
  clearance(-0.5)/orientation(-2.5)/action_acc, 关掉 progress (混合地形无固定前向)。
  目的: 把台阶能力带回可部署的混合地形, 保住 hyb2/stg1/平地。

## 4. 当前进展

* 2026-10-21 08:34 v29 smoke (0.3M 步, 0.02m): eval_reward -788 -> -664, ep_len
  667/750 (不再开局掉落); 爬升与 v25b 持平 (0.016->0.017 @0.02; 0.032->0.033 @0.03),
  说明仅 0.3M 步还不足以改变行为, 需完整 stage。
* **v29a-s0 (0.02m, 实际 2.95M 步, 08:51 完成)**: eval_reward -812 -> -457 -> ... -> +172,
  ep_len 663 -> 696/750, fps 2500-3460 (full_reset 约 25% 代价)。

  爬升评测 (128env x 300步, spawn_stairs=1.0):

  | 地形(riser) | 策略 | 存活 | climb_mean | p90 | 上1级% | 上2级% | yaw p90 | vx |
  |---|---|---|---|---|---|---|---|---|
  | wide_h020 (0.02) | v25b | 96.9 | 0.017 | 0.028 | 46.0 | 1.6 | 27.3 | .265 |
  | wide_h020 | v29s0 | 96.9 | 0.020 | 0.032 | 58.9 | 0.8 | 21.0 | .275 |
  | wide_h030 (0.03) | v25b | 96.9 | 0.032 | 0.046 | 66.9 | 1.6 | 39.1 | .247 |
  | wide_h030 | v29s0 | 96.9 | 0.038 | 0.054 | 79.0 | 4.8 | 25.4 | .250 |
  | wide_h050 (0.05) | v25b | 86.7 | 0.037 | 0.055 | 25.2 | 0.0 | 62.4 | .205 |
  | wide_h050 | v29s0 | 71.1 | 0.039 | 0.065 | 37.4 | 1.1 | 55.6 | .197 |

  -> P1 的防绕行/防掉头明显有效 (yaw p90 27-62 度 -> 21-56 度, 上 1 级比例 +13~17pp);
  但仍然只能上 1 级 (上 2 级 <5%), 0.05m 上存活率还下降。说明缺"连续抬脚/跨多级"能力
  -> 进入 P2: mjlab 速度加权 feet_clearance (-1.0, target 0.10) + 逐阶段加高 riser。

* **v29a-s1 (0.03m + mjlab 速度加权 feet_clearance -1.0/0.10 + swing 0.18, 2.5M 计划,
  实际在 1.97M 处因 8GB 显存 OOM 中断; best=1.97M 评估)**: eval_reward -169 -> -102 ->
  +34 -> +211, ep_len 686-711。

  爬升评测 (128env x 300步, spawn_stairs=1.0):

  | 地形(riser) | 策略 | 存活 | climb_mean | p90 | 上1级% | 上2级% | yaw p90 | vx |
  |---|---|---|---|---|---|---|---|---|
  | wide_h030 (0.03) | v25b | 97.7 | 0.034 | 0.048 | 71.2 | 2.4 | 38.0 | .249 |
  | wide_h030 | v29s1 | 97.7 | **0.046** | 0.061 | **93.6** | **17.6** | **20.8** | .262 |
  | wide_h050 (0.05) | v25b | 84.4 | 0.036 | 0.056 | 25.9 | 0.0 | 61.1 | .202 |
  | wide_h050 | v29s1 | 60.2 | **0.053** | 0.083 | **63.6** | 3.9 | 59.9 | .199 |
  | wide_h080 (0.08) | v25b | 64.1 | 0.016 | 0.027 | 0.0 | 0.0 | 80.5 | .178 |
  | wide_h080 | v29s1 | 34.4 | 0.023 | 0.038 | 0.0 | 0.0 | 82.3 | .168 |

  -> **台阶专项信号开始生效**: 0.03m 上 1 级 71%->94% (p90 爬升 0.061 = 2 级),
  上 2 级 2.4%->17.6%, 掉头角度 p90 38->21 度; 但 0.05m 上存活率下降 (60%),
  0.08m 仍不行 -> 继续逐级课程 (S2 0.05, S3 0.08), 并需更多训练量。

* **v29a-s2 (0.05m, 3.1M 步, 09:25 完成)**: eval_reward -200 -> -127 -> ... -> -5.9,
  ep_len 653-679; 显存配置改为 ENVS=2048/naconmax=49152/njmax=192/XLA 0.35 后稳定。

  | 地形(riser) | 策略 | 存活 | climb_mean | p90 | 上1级% | 上2级% | yaw p90 | vx |
  |---|---|---|---|---|---|---|---|---|
  | wide_h030 (0.03) | v25b | 97.7 | 0.033 | 0.051 | 67.2 | 4.0 | 40.0 | .251 |
  | wide_h030 | v29s2 | 96.9 | 0.040 | 0.054 | 89.5 | 6.5 | 21.0 | .245 |
  | wide_h050 (0.05) | v25b | 88.3 | 0.036 | 0.055 | 29.2 | 0.9 | 62.7 | .205 |
  | wide_h050 | v29s2 | 63.3 | 0.043 | 0.068 | 48.1 | 0.0 | 52.8 | .193 |
  | wide_h080 (0.08) | v25b | 59.4 | 0.016 | 0.030 | 0.0 | 0.0 | 82.6 | .178 |
  | wide_h080 | v29s2 | 57.8 | 0.018 | 0.030 | 0.0 | 0.0 | 81.1 | .180 |

  -> 0.05m 上开始能上 1 级 (29%->48%), 但**出现遗忘**: 0.03 的上 2 级从 17.6% 回落到 6.5%
  (S1 模型 93.6%/17.6% vs S2 模型 89.5%/6.5%)。单阶段串行课程会互相覆盖。
* **v29b-lanes (混合难度赛道, 进行中, 09:29 启动)**: 新地形模拟 legged_gym 的"一张地形
  多难度瓦片": 4 条互相隔离的楼梯赛道 (riser 0.03/0.05/0.08/0.11, 宽 2.0/2.0/2.0/1.6m,
  三条 0.5m 高隔墙), 每条入口 3 格平道 + spawn_entries; 2560/2048 个 env 同时覆盖 4 个
  难度, 从 S2 best 热启动 3.5M, 其余信号同 P1/P2 (clearance -1.0, progress 1.5,
  lazy -1.5, term -20, orient -5)。目标: 同时保住 0.03 并提升 0.05-0.11, 消除遗忘。
  首评 (1.05M): eval_reward -535 -> -408, ep_len 481/750 (隔墙+高难度导致终止偏多,
  继续观察)。

* OOM 处理: 8GB 显存下 full_reset + 2560 env + naconmax=131072 会 OOM (warp EPA
  缓冲 55MB 分配失败, 发生在 2.4M 步附近)。S2 起改为 ENVS=2048 / naconmax=49152 /
  njmax=192 / XLA 显存比例 0.35, 并把 naconmax/njmax 做成 CLI (--naconmax/--njmax)。

## 6. v29c 最终验收 (2026-10-21 10:05-10:15)

训练链: v25b -> wide_h020 (3M) -> wide_h030+feet_clearance (2M) -> wide_h050 (3.1M)
-> lanes1 混合难度 0.03/0.05/0.08/0.11 (4.2M) -> hyb2 巩固 (3.1M) = **v29c**。

### 6.1 台阶爬升阶梯 (warp, 128env x 300步=6s, spawn_stairs=1.0, cmd 0.4 m/s, 带遮挡 lidar)

| 地形 (riser) | 策略 | 存活% | climb 均值 | p90 | 上1级% | 上2级% | yaw p90 | vx |
|---|---|---|---|---|---|---|---|---|
| wide_h030 (0.03) | v25b | 98.4 | 0.033 | 0.051 | 65.1 | 1.6 | 37.6 | .254 |
| wide_h030 | **v29c** | 96.9 | **0.040** | 0.057 | **83.1** | **12.1** | 31.0 | .247 |
| wide_h050 (0.05) | v25b | 86.7 | 0.034 | 0.056 | 19.8 | 0.0 | 58.5 | .203 |
| wide_h050 | **v29c** | 82.0 | **0.042** | 0.062 | **38.1** | **1.9** | 49.3 | .193 |
| wide_h080 (0.08) | v25b | 66.4 | 0.018 | 0.036 | 0.0 | 0.0 | 82.4 | .183 |
| wide_h080 | v29c | 68.0 | 0.021 | 0.037 | 0.0 | 0.0 | 80.1 | .186 |
| wide_h110 (0.11) | v25b | 62.5 | 0.010 | 0.021 | 0.0 | 0.0 | 90.1 | .187 |
| wide_h110 | v29c | 78.1 | 0.016 | 0.031 | 0.0 | 0.0 | 87.8 | .209 |
| lanes1 (混合 0.03-0.11) | v25b | 61.7 | 0.034 | 0.055 | 59.5 | 7.6 | 80.5 | .162 |
| lanes1 | **v29c** | 46.1 | **0.045** | 0.066 | **83.1** | **22.0** | 76.7 | .165 |

要点: 0.03m 上 1 级 65%->83%, 上 2 级 1.6%->12%; 0.05m 上 1 级 20%->38%;
混合赛道 1 级 60%->83%, 2 级 7.6%->22%。0.08m 以上仍基本爬不上 (climb≈0.02),
与外部证据一致 (Go2 公开能力上限 ~0.10m; 0.14m=52% 站高)。绕行/掉头明显改善
(yaw p90 全面下降 5-15 度; 0.03 由 37.6->31.0, 0.05 由 58.5->49.3)。

### 6.2 部署地形 warp 矩阵 (384env x 6s, 12 个标准地形协议)

| 地形 | v25b 存活% | v29c 存活% |
|---|---|---|
| hyb2 | 91.1 | **92.2** |
| stg1 (台阶+金字塔) | 79.4 | **82.6** |
| bg200 (0.06/0.11 台阶) | 83.6 | **87.8** |
| hyb1 | 92.4 | 91.9 |
| rubble | 85.2 | **94.5** |
| wide_h030 | 97.7 | 98.2 |
| wide_h050 | 85.9 | 79.9 |
| wide_h080 | 62.0 | **70.1** |

### 6.3 部署 CPU 闭环 (24 seeds x 15s, profile go2_v29c)

| 地形 | 存活% | v 均值 | 位移 | v25b 存活% |
|---|---|---|---|---|
| flat | 100.0 | 0.335 | 4.69 | 100.0 |
| hyb2 | 87.5 | 0.272 | 3.40 | 75.0 |
| stg1 | 70.8 | 0.241 | 2.66 | 62.5 |
| big1 | 91.7 | 0.282 | 3.48 | 91.7 |
| rubble | 95.8 | 0.321 | 4.35 | 62.5 |
| **合计** | **89.2** | 0.29 | 3.71 | 76.2 |

髋 q p95 = [43.8, 41.0, 37.7, 41.8] 度 (v25b [47.1, 45.3, 43.3, 45.6]);
卡死事件 0.97/seed, 严格卡死 0; hyb2 取证: 缝夹 0/16, 1 次 d_other, 位移 3.74m, v 0.29。

### 6.4 结论

1. v29 修复了训练栈的 **full_reset=False (每 env 整个训练回放同一初始状态)** 与
   地形/命令/奖励面的四个绕行漏洞; 台阶能力从"完全不会"到"0.03-0.05m 可以稳定上 1 级、
   偶尔 2 级", 但仍未达到 0.08m 以上。
2. 部署侧综合鲁棒性显著提升 (CPU 76.2% -> 89.2%, warp 多地形 +1~+9pp), 髋摆幅下降,
   flat 仍 100% -> **v29c 已设为默认交付档**, v25b 保留回退。
3. 仍未解决: 0.08-0.11m riser 的多级攀爬。下一步 (按证据):
   (a) per-env 地形课程 (playground wrapper full_reset=True 已开, 还差
       AutoResetWrapper_preserve_info + reset(rng, level) 的 ~120 行课程簿记,
       把 10 级难度瓦片放在同一张地形里);
   (b) 更长训练/更大网络 (4.2M 混合赛道仍在上 2 级 22%, 有继续提升空间);
   (c) 若仍不够, 引入 reference motion / WTW 式带净空参数的摆动轨迹先验 (训练侧轨迹
       生成器输出关节目标 + 残差, 部署接口不变)。
