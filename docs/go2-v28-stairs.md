# Go2 v28: 台阶专项 + 步态自然性 —— 实验记录 (2026-10-21, 进行中)

用户三条反馈 (turn 60/62):
1. 上台阶成功率还是比较低;
2. 动作不自然, 整体四个髋关节摆动幅度极大导致鲁棒性不佳、走路晃来晃去;
3. 经常发生脚卡到地面里后完全动不了。
并补充: 卡死怀疑"楔在瓦片竖缝/台阶立面", 注意性能损耗; v28 可试"只有台阶的环境",
可适当增大模型, 注意机体稳定性; 训练前把问题想透 (允许用 Agent Team 与联网调研)。

本文件记录: 三条反馈的根因实测 → 修复方案 → 训练实验 (A1 负结果 / A1b / B / C)。
部署侧修复记录见 mjx-go2-mujoco2ros2/docs/go2-sim-fidelity.md 第 5 节。

## 0. TL;DR

| 反馈 | 根因 (实测) | 修复 | 状态 |
|---|---|---|---|
| 上台阶成功率低 | 训练侧弱项 (warp: stg1 79.9% / bg200 83.6% / hyb2 92.7%); 台阶在 hyb2 只占 2/9 patch, 随机出生遇到台阶样本比例低; 且 hyb2/stg1 有 0.22m 单级立面 (82% 站高, 超出 Go2 公开能力 0.10m) | 新建 bands 台阶地形 (7 条全长楼梯带, 覆盖 86.4%, spawn_entries 楼梯口出生) 做 stairs-only 专项训练 | A1b 中 -> B |
| 四髋摆动极大/走路晃 | 奖励面缺项: hip_q mean 29-33 度, hip 动作 p95 1.2-1.5rad 顶到 +-1.047 限位; v20/v24/v25a/v25b 一致、平地同样存在。A1 负结果: v25b + hip_abduction(-0.5) 微调 3M, 髋角/髋动作与 v25b 逐位相同 -> 状态惩罚被步态吸引子淹没 | 新增 hip_action 惩罚 (直接惩罚策略输出, 梯度不经动力学); A1b 实测 hip_q 30.96->28.88 度 (-7%), 有效但幅度小; B/C 继续加大权重 | A1b 完成 (小幅) -> B/C |
| 脚卡地面动不了 | T1 接触取证: 3/3 非出生期严格卡死 = 球足被相邻精确尺寸倾斜瓦片的缝夹住 (持续穿透 2.1-2.5cm, 法向力 1.3-1.7kN); 合成实验: 精确瓦片接缝 79% 嵌入 -> 瓦片 x1.05 重叠 0% | 部署导出器瓦片尺寸 x1.05 (数量/性能不变) + alpha=0.01 (CPU MuJoCo mj_ray 跳过 alpha=0 的 geom) + 视觉 hfield group 3->1 | 已修 (见 sim-fidelity 第 5 节) |

## 1. 诊断证据 (T1: 卡死力学取证, task-7)

脚本: mjx-go2-mujoco2ros2/scripts/stuck_forensics.py (可复用 --scene/--profile/--seeds/
--duration/--spawn/--phase; 输出 JSON + 每事件接触窗口 + PNG)。

严格卡死判据: 连续 >=1.2s 基座平面位移 <2cm (cmd=0.4, tilt<30 度, 未摔倒, 过 2.5s warmup);
缝夹 (pinch) 判据: 同一足球同时有 2 个接触、世界法向水平分量 |n_xy|>0.7 且对立
(dot<-0.6)、两侧法向力均 >30N。

修复前 (v25b, 16 seeds x 15s, ring 出生):

| 地形 | 摔倒 | 非出生期严格卡死 | 缝夹 seeds | 缝夹 s/seed | 前进 m |
|---|---|---|---|---|---|
| flat | 0/16 | 0 | 0/16 | 0 | 4.84 |
| hyb2 | 0/16 | 2 (~0.5 事件/min) | 6/16 | 1.04 | 4.03 (-17%) |
| stg1 | 2/16 | 0 | 3/16 | 1.54 | 3.34 (-31%) |

卡死机制 (逐例接触证据):
* 3/3 非出生期严格卡死 = 球足被两块相邻碰撞盒的近竖直侧面夹在缝里: 持续穿透
  2.1-2.5cm、单接触法向力 0.67-0.74kN (整足 1.3-1.7kN, 正常站立约 40N/足)、
  被卡足 xy 位移 <=8mm、逃逸后数值弹射 5m。3 例中 2 例在隐藏瓦片缝 (tc_*),
  1 例在真竖面方块缝 (bx_*)。
* 1 例 (b) 小腿/机身顶在 0.22m 高 bx 立面上 (不是球足被楔)。
* stg1 无瓦片也有 3/16 缝夹 -> 离散 box + 软接触本身会夹球足, 瓦片缝把平滑地形
  也变成 0.2m 缝网, 显著放大问题。

合成最小实验 (2kg 足型球 r=22mm, 落接缝中点): 精确尺寸瓦片 52%(合成)/79%(真实
hyb2 瓦片区) 落点持续嵌入; 瓦片 xy x1.02->9-16%; x1.05->0%; 连续三角面->0%。
-> 修复 = 瓦片重叠 (只改 size, 几何数量/性能不变)。

评测口径附带 bug: uniform 出生只按中心点 raycast, 四足出生即嵌瓦片 2-7.8cm
(6/16 假卡死, 前进 -54%)。修复: 出生高度取四足偏移点下方地形最高点 +2cm
(scripts/eval_terrain_matrix.py::_ground_safe)。

## 2. 诊断证据 (T2: 碰撞表示 A/B, task-8)

候选: C0 现状; C1 重叠瓦片 1.02/1.05/1.10 (alpha=0 或 0.01=ray 可见); C2 连续三角
棱柱 mesh (1908 个); C3 真 hfield 碰撞。

| 项 (hyb2 ring 24×15s) | C0 | C1×1.05 | C1×1.10 | C2 mesh |
|---|---|---|---|---|
| 存活 | 70.8% | **75.0%** | 87.5% | 75.0% |
| 前进 / v | 3.32m/.267 | 3.53/.289 | 3.80/.297 | 3.62 |
| 缝贯穿率 (1526 缝点) | **7.27%** (深 28mm) | **0.26%** (10mm) | 0% | 0% |
| 严格卡死 (T1 口径) | 2 (含瓦缝 seed13) | 0-7 (边界 seed) | 0 | 17 |
| mj_step 中位 | 0.094ms | 0.072ms | 0.085ms | 0.185ms |
| ngeom / ncon / 内存 | 793/22/45.4MB | 同 | 同 | 2086/24/50.5MB |

结论: **推荐 C1 重叠 1.05** (性能零代价, 缝消除, 四项指标方向一致不差);
C2 mesh 几何最干净但加载 +330ms、整机不更优、严格卡死反而更差 -> 不采用;
C3 真 hfield 否决 (球-hfield 返回 1~6 个**同点重复接触**, ncon 峰值 17, 存活 45.8%)。

**重要附带发现**: CPU MuJoCo 3.12 的 mj_ray **跳过 rgba alpha==0 的 geom** ->
旧部署场景的隐藏瓦片对 raycast 不可见, uniform 出生地面探测 mean -22.3mm / 37.5%>2cm;
最终修复为: 瓦片保持 alpha=0 (扫描看视觉 hfield 面, 不看碰撞包络), 视觉 hfield
group 3->1 (环境组掩码 raycast 可见), 出生高度取四足偏移点地形最高点 +2cm。

## 3. 外部调研 (T3: v28 方案, task-9; 报告 /tmp/t3_v28/research_v28.md, 60KB)

要点 (均有 URL):
* 台阶能力外部上限: 官方 Go2 配方 (unitree_rl_mjlab) riser <=0.10m=37% 站高; ANYmal
  (站高 0.55m) 0.05-0.23m; 换算 Go2 42%=0.113m/55%=0.147m -> **训练主轴 riser 到
  0.14m, 0.18/0.22 只作极限档, 不设通过门**。
* hip_abduction 与 extreme-parkour 的 hip_pos=-0.5 同构; 但 **A1 实测证明在 v25b
  热启动 3M 微调下无效** (见 §4) -> 需要直接动作惩罚 (A1b) 或从零训练。
* 网络: actor 1024-512 保持 (v21 的 2048-1024 实测 -0.6pp / fps -33% / 推理 4x)。
* 课程: 先 stairs-only (B) 后 hyb2 巩固 (C); 不要在混合地形里单纯提高台阶比例
  (v7/v8/v16/v19/v22 五次负结果); stairs-only 后必须 hyb2 巩固 (否则重演
  big1-only 的部署偏航, v26/v27 教训)。
* 奖励预算: step() 里 reward=clip(total,0,10000), v25b 的 pre-clip 均值 +1.03 /
  clip_active 21.3% (128env x 300步实测); 新负项要 cap, 且监控 clip 比例。

## 4. A1: v25b 热启动 + 步态 pack 微调 3M —— 负结果 (2026-10-21 01:27 完成)

配置: v25b@2.05M 热启动, hyb2 3M, lr 2e-5, hip_abduction=-0.5 (cap1.2),
symmetry -0.4->-0.6, action_rate -0.01->-0.02, 其余与 v25b 完全一致。

训练: eval_reward 1008->1107 (+9.8%), ep_len 629->605。
warp 地形 (384env x 6s, 与 v25b 同进程): hyb2 91.4 vs 92.7 / stg1 80.7 vs 79.9 /
bg200 83.3 vs 83.6 / hyb1 90.9 vs 90.1 / rubble 87.5 vs 87.8 / bands1 77.1 vs 75.5 /
bands_u006 82.8 vs 81.0 / bands_u010 61.7 vs 66.7 / u014 76.3 vs 77.9 / u018 75.8 vs 82.3 /
u022 75.3 vs 81.8 -> **地形能力基本不变** (高台阶略降), 无回归。

**但髋指标完全没变** (probe 128env x 300步, hyb2, 同口径):
hip_q mean 30.34°(A1) vs 30.96°(v25b), p95 33.9 vs 34.4, hip 动作 p95 0.64 vs 0.65。
-> 结论: hip_abduction 状态惩罚的梯度穿过 500Hz 动力学+PD 后被稳定步态吸引子淹没;
3M/2e-5 微调无法改变步态风格 (与 doc §27 "永久不对称姿态是收敛吸引子、加步数修不好"
同源)。A1 可作为 B 的 warm start (地形不回归), 但不解决反馈 #2。

## 5. A1b: 直接髋动作惩罚 (2026-10-21 01:53 完成) —— 小幅正结果

改动 (envs/go2_walk.py): 新增 hip_action cost = min(Σ(a_hip-home)², cap1.2),
权重为负; 直接惩罚策略输出的髋目标 (梯度不经动力学)。
配置: A1@3.19M 热启动, hyb2 4M, lr 3e-5, entropy 0.003,
hip_abduction -0.8, hip_action -0.5, symmetry -0.6, action_rate -0.02。
训练: eval_reward 739 -> 823 (+11%), ep_len 646 -> 598。

髋指标 (probe 128env x 300步, hyb2):

| 指标 | v25b | A1 (状态惩罚) | A1b (直接动作惩罚) | 目标 |
|---|---|---|---|---|
| hip_q mean | 30.96 度 | 30.34 | **28.88** (-2.1 度) | <=20 度 |
| hip_q p95 | 34.4 度 | 33.9 | **32.2** | <=32 度 |
| hip 动作 p95 | 0.65 rad | 0.64 | **0.61** | <=1.05 |

-> 直接动作惩罚**有效但幅度小** (-7%)。推断: (1) 4M/3e-5 仍偏短;
(2) 权重 -0.5 只占正奖励 ~2.6 的 12%; (3) 宽站姿可能是策略的稳定解。
B 阶段把 hip_action 提到 -0.8 并在新地形上继续训练 (顺带验证"换地形能否打破
步态吸引子")。

部署侧 CPU 平地闭环 (24 seeds x 15s, scripts/eval_terrain_matrix.py, 同一出生序列):

| 指标 | v25b | v28a1b | 变化 |
|---|---|---|---|
| 存活 / v / dist | 100% / 0.330 / 4.74m | 100% / 0.358 / 5.15m | 略快 |
| tilt_p95 | 6.1 度 | 6.6 度 | +0.5 |
| hip_q p95 (FR/FL/RR/RL) | 47.1/45.3/43.3/45.6 度 | 44.5/40.0/39.8/44.7 度 | -2.6/-5.3/-3.5/-0.9 |
| hip 动作 p95 | 1.47/1.19/1.23/1.37 rad | 1.38/1.13/1.16/1.39 | 略降 |

-> 部署 CPU 口径也确认髋幅度下降 (p95 降 1~5 度), 平地不退化。但距离"自然步态"
仍有差距; 需要更长训练/更强权重或从零重训 (v28 时间预算内先交付台阶能力)。

warp 地形 (384env x 6s, 与 v25b 同进程): hyb2 90.1 vs 92.4 / stg1 78.1 vs 80.7 /
bg200 81.0 vs 83.1 / hyb1 91.4 vs 92.4 / rubble 84.6 vs 86.5 / bands1 67.4 vs 71.4 /
u010 63.3 vs 66.4 / u014 76.0 vs 74.0 / u018 72.9 vs 76.8
-> 地形略降 1-4pp (髋自由度被限制的代价), 可接受; 作为 B 的起点。

## 6. 台阶专项地形: bands (sim/gen_terrain_stairs_bands.py)

7 条沿 ±x 的全长楼梯带铺满 22x22 格 (±4.4m), 带宽 4/4/3/3/3/3/2, 覆盖 **86.4%**
(stairs1 只有 37.2%); 单级 riser 0.06~0.22m, 踏面 0.4m; 每条带 = 入口平->逐级上->
顶部平台->逐级下->出口平; 相邻带高度剖面不同->带间竖壁, 强迫沿带爬。
json 带 **spawn_entries** (14 个楼梯口: 两端 x=±4.6, y=带中心, yaw 朝带内), env 新增
random_init.spawn_stairs_prob (以该概率在入口前方出生并朝向楼梯)。
同族评测场: bands_u006/u010/u014/u018/u022 (所有带同一 riser 高度的高度阶梯)。

### 6.1 **B 负结果 + bands2 修正** (2026-10-21 02:33)

B (bands1, 8M, spawn_stairs 0.8, 前向指令, ascent 2.0) 训练完成但**没学会爬台阶**:
8 个检查点在 warp climb 评测里 max 爬升 **0.007~0.015 m** (v25b 0.010m), 全部 128 env
没有一个爬过 0.10m。eval_reward 轨迹 635(step0 热启动) -> 537 -> 572 -> 617 -> 627,
即"先退化再恢复", 但 best 仍是 step0 -> 训练在**优化另一种行为**。

轨迹取证 (jitted 单 env dump): 入口出生后机器人走 0.3-0.4m 到第一个 riser 前,
然后**转身沿 y 方向走**。原因: 原 bands 剖面两端各有 1~2 格平地 (profile 以 0 开头/结尾),
22x22 格场的 x 两端因此形成 **0.4~0.8m 宽的平走廊 (沿 y)**; 策略发现"转身沿走廊走"
同样能拿速度跟踪奖励, 且不用克服 riser -> 学会绕开台阶 (典型 reward hacking)。

**bands2 修正** (生成器已改, 默认 --no_flat_ends):
* 剖面去掉两端平格: 第一条/最后一条格子高度 = h (入口即 riser, 不爬无法前进);
  长度约束变为 2*n_up + plat - 1 = 22 -> platform 格数重配 (A 11, B 13, C 15, D/E 17, F/G 19);
* 场地覆盖率 100% (484 个 box), 无任何平地走廊;
* spawn_entries 位置不变 (x=±4.6), spawn_stairs_prob=1.0 (B2 只用楼梯口出生);
* B2 训练配置: 6M, ascent 4.0 (行进门控), hip_abduction -0.8, hip_action -0.5,
  symmetry -0.6, action_rate -0.02, 前向指令 vx∈[0.2,0.9]。

### 6.2 **B2 也失败** (4.1M 时提前评估, 2026-10-21 03:07)

B2 (bands2, 4.1M) eval_reward 440(step0) -> 460 -> 436 (不再上升)。
用专门的爬升评测 (sim/eval_stairs_climb.py, 128env x 300步, 出口在楼梯口, 量
max base_z - 初始 z) 对比:

| 策略 | 存活 | 平均最大爬升 | p50 | p90 | 爬过 0.10m 比例 | vx |
|---|---|---|---|---|---|---|
| v25b | 78.1% | 0.009 m | 0.006 | 0.020 | 0% | 0.10 |
| A1b | 47.7% | 0.014 m | 0.011 | 0.028 | 0% | 0.12 |
| B2@4.1M | 54.7% | 0.015 m | 0.013 | 0.030 | 0% | 0.11 |

**没有任何策略能爬上 0.06m 的单级 riser** (bands2 第一条带就是 0.06m)。轨迹取证:
入口出生 -> 走 0.3-0.4m 到 riser 前 -> 转向/沿边行走 -> 多数在 1.6-2.4s 内摔倒
(脚/小腿蹭到相邻带的立面)。这不是"走廊绕行"了 (走廊已去除), 而是**策略完全不会
抬腿上台阶**: 平地步态的实际抬脚净空不足 (目标 0.15m 的 gait_swing_height 只是
奖励塑形, 实际摆动净空估计 0.05-0.08m < 需要的 0.083m), 且转向/蹭墙导致摔倒。

结论: 在现有奖励/动作空间/1024-512 网络 + 3-8M 步预算下, **learn stairs 没有成功**;
需要至少 (a) 显式抬脚高度/台阶边缘课程 (从 0.03-0.06m 更矮台阶开始),
(b) 更大的网络/更长训练, (c) 或改用 reference motion / 教师学生蒸馏。

### 6.3 D: 统一 0.06m 矮台阶 + swing 0.20 (3M, 2026-10-21 03:22)

最后尝试: bands_e2 (所有带统一 riser 0.06m, 无平端) 从 v25b 训 3M,
swing_height 0.20 (台阶专项重试; 混合地形上曾证伪)。eval_reward 486->498->510 (升)。

| 策略 (bands_e2, 128env x 300步) | 存活 | 平均最大爬升 | p50 | p90 | >0.10m | vx |
|---|---|---|---|---|---|---|
| v25b | 64.8% | 0.032 m | 0.031 | 0.062 | 1.2% | 0.157 |
| v28d (3M) | 64.8% | **0.042 m** | 0.035 | **0.080** | **2.4%** | 0.154 |

-> 在**统一 0.06m** 矮台阶上, D 的爬升比 v25b +31% (p90 0.08m = 1.3 级), 但**仍无法
稳定爬多级**(走完 6 级需 0.36m); 且 D 只在矮台阶训练, 未做 hyb2 巩固, 不能直接交付。
这条曲线说明: 0.06m 单级已接近当前策略能力边缘 (实际抬脚净空不足), 台阶能力提升
需要更系统的抬脚高度训练/课程, 不是几次短训能解决的。

## 6.4 本轮台阶结论与下一步建议

* 台阶能力 (反馈 #1) **本轮未解决**: 8M stairs-only (bands1/bands2) 均未学会爬 0.06m;
  bands1 的失败已被证明是地形缺陷 (平走廊 reward hacking) 并可修, bands2/D 的失败
  指向**策略抬脚净空不足 + 蹭墙摔倒**。
* 建议下一步 (按性价比): (1) 先做抬脚高度诊断/训练 (feet_height 目标 0.20-0.25 +
  实际净空统计, 台阶从 0.03m 起步逐级加高); (2) 更大网络 (1536-768) + 10-20M 步;
  (3) 若仍不行, 采用 reference motion (unitree_rl_gym Go2 walk) 或教师-学生蒸馏;
  (4) 评估时用 sim/eval_stairs_climb.py 的爬升分布, 不要只看存活率 (站立不动也能
  100% 存活)。
* 已知上限: hyb2/stg1 的 0.22m 单级 = 82% 站高, 超出 Go2 公开能力 (官方 0.10m),
  即使训练成功也只能作极限档, 不应作为通过门。

## 7. 训练计划 (GPU 串行, gpu_guard.sh)

| 阶段 | 内容 | 起点 | 步数/lr | 状态 |
|---|---|---|---|---|
| A1 | hyb2 步态 pack 微调 | v25b | 3M / 2e-5 | 完成 (负结果: 髋不变) |
| A1b | 直接髋动作惩罚 | A1@3.19M | 4M / 3e-5 | 完成 (髋 -7%, 小幅正结果) |
| B | stairs-only bands1 专项 | A1b best | 8M / 3e-5 | 完成 (负结果: 学会走平走廊绕台阶) |
| B2 | stairs-only **bands2** (无平走廊) | A1b best | 4.1M 提前停 | 完成 (负结果: 仍不会爬 0.06m) |
| D | stairs-only bands_e2 (uniform 0.06) + swing 0.20 | v25b | 3M / 3e-5 | 完成 (爬升 +31% 但仍不能多级) |
| C | hyb2 巩固 | A1b best | 2.5M / 1.5e-5 | 进行中 (交付候选: 保 hyb2 能力 + 髋 -7%) |

B2 的额外配方: --spawn_stairs_prob 1.0, ascent 4.0, 前向偏置指令
cmd_low=0.2,-0.3,-1.0 / cmd_high=0.9,0.3,1.0, hip_abduction -0.8, hip_action -0.5,
symmetry -0.6, action_rate -0.02, swing_height 0.15 (R2 已证伪 0.20, 不重试)。

验收双轨:
* warp 矩阵 (probe_sweep_params.py, 384env x 300步, spawn_stairs 0.8, nominal lidar):
  hyb2/stg1/bg200/hyb1/rubble/bands*/bands_u* 与 v25b 同进程对比;
* 部署 CPU 闭环 (scripts/eval_terrain_matrix.py + scripts/stuck_forensics.py):
  平地 24seed x 15s 的 hip_q/hip_act/tilt/vx/duty; hyb2/stg1 的存活/前进/严格卡死/pinch。

## 8. 复现命令

    # 生成 bands 地形 (默认无平端; 旧布局加 --flat_ends)
    $PY sim/gen_terrain_stairs_bands.py --prefix bands2
    # A1
    TAG=v28a1_pack SRC=models/go2_v25b_cons_logs/checkpoints/000002048000 \
      SCENE=models/go2/scene_mjx_terrain_hyb2.xml BOXGRID=models/go2/assets/hyb2.json \
      CONTACT=clearance RING=1 STEPS=3000000 LR=2e-5 HIPW=-0.5 SYMM=-0.6 ACTW=-0.02 \
      SWING=0.15 bash sim/run_go2_v28.sh
    # 评测 (warp 同进程多策略/多地形)
    $PY sim/probe_sweep_params.py --ckpts "v25b=...,v28a1=..." \
      --terrains "hyb2=...,bands1=..." --boxgrid_map "hyb2=...,bands1=..." \
      --contact_mode clearance --spawn_ring --spawn_stairs_prob 0.8 \
      --envs 384 --steps 300 --spawn random --cmd 0.4 0 0 --foot-contact hard \
      --healthy_z_ref feet --scan_nx 17 --scan_ny 11 --scan_x -0.4 1.2 \
      --scan_y -0.5 0.5 --lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 \
      --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25 \
      --out logs/v28_sweep.tsv
    # 部署侧验收
    $PY scripts/eval_terrain_matrix.py --terrains hyb2,stg1,big1,rubble,rough_only \
      --seeds 24 --duration 15 --jobs 3 --out logs/matrix.json
    $PY scripts/stuck_forensics.py --scene hyb2 --spawn ring --seeds 0:16 --duration 15

## 9. 最终结果 (2026-10-21 03:46) —— 诚实结论

### 9.1 v28c (A1b -> hyb2 巩固 2.5M) 与 v25b 同口径对比

warp 地形矩阵 (384env x 300步, 台阶地形 spawn_stairs=1.0, 其余 ring):

| 地形 | v25b | v28c | 差 |
|---|---|---|---|
| hyb2 | 91.9% | 89.6% | -2.3 |
| stg1 | 79.4% | 76.6% | -2.8 |
| bg200 | 84.4% | 78.6% | -5.8 |
| hyb1 | 91.9% | 90.4% | -1.5 |
| rubble | 87.2% | 85.2% | -2.0 |
| bands2 (0.06~0.22) | 60.2% | 47.4% | -12.8 |
| bands_e2 (uniform 0.06) | 68.5% | 52.6% | -15.9 |

爬升能力 (128env x 300步, spawn_stairs=1.0):

| 地形 | 策略 | 存活 | 平均最大爬升 | p90 | >0.10m |
|---|---|---|---|---|---|
| bands2 | v25b | 61.7% | 0.013m | 0.026 | 0% |
| bands2 | v28c | 46.9% | 0.016m | 0.030 | 0% |
| bands_e2 | v25b | 71.1% | 0.034m | 0.061 | 2.2% |
| bands_e2 | v28c | 49.2% | **0.048m** | **0.089** | **4.8%** |

髋指标 (warp probe 128env x 300步, hyb2): v25b 31.1°/34.7°/0.64rad ->
v28c **28.2°/31.9°/0.60rad** (-9%); 部署 CPU 平地 24seed: hip_q p95
47.1/45.3/43.3/45.6 -> 44.2/43.5/40.9/42.5 度, flat 100% 存活, v 0.362 (v25b 0.330)。

### 9.2 结论

| 反馈 | 结果 |
|---|---|
| #3 脚卡死 | **已修复** (部署侧碰撞几何, 与策略无关): hyb2 pinch 6/16->0~1/16, 严格卡死 2->1 (剩余为 0.22m 立面挡机身, 属策略/训练问题); 性能零代价; tests 12/12。详见 mjx-go2-mujoco2ros2/docs/go2-sim-fidelity.md §5 |
| #2 髋摆/走路晃 | **部分改善**: A1b 的 hip_action 直接惩罚有效 (髋 -7%), v28c 巩固后 -9%; 但代价是地形鲁棒性 (bg200 -6pp, 台阶地形 -13~16pp) -> **不作为默认交付** (默认仍 v25b; 实验档 go2_v28a1b/go2_v28c) |
| #1 上台阶 | **未解决** (负结果): 3 次 stairs-only 训练 (bands1 8M / bands2 4.1M / bands_e2 3M) 均未学会稳定爬 0.06m; 根因链已查清 (bands1 平走廊 reward hacking -> bands2 无走廊后仍不会抬脚; 实际抬脚净空不足 + 蹭带墙摔倒)。所需下一步见 6.4 |

### 9.3 本轮新增可复用资产

* 部署侧: 瓦片 x1.05 重叠 + raycast 可见性修复 + 24 个地形 (含 bands/stairs) +
  scripts/stuck_forensics.py + scripts/collision_bench.py + scripts/eval_terrain_matrix.py
* 训练侧: sim/gen_terrain_stairs_bands.py (无平端楼梯带 + spawn_entries),
  envs/go2_walk.py 新增 hip_abduction/hip_action/feet_edge_bg/spawn_stairs_prob,
  sim/eval_stairs_climb.py (爬升分布评测), sim/run_go2_v28.sh (v28 训练脚本)
* 实验记录: 本文件 + 部署仓 docs/go2-sim-fidelity.md §5

