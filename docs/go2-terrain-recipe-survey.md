# Go2 地形行走配方调研: 高度扫描 obs / 地形课程 / 奖励 —— 面向 MJX+brax 的离散障碍改进

> 日期: 2026-10-19 · 调研人: terrain-research
> 范围: 只做 CPU/网络侧调研 (读源码与论文), 不改训练/环境代码, 不占 GPU。
> 来源快照: 下表的 GitHub 文件行号取自 2026-10-19 抓取的 `master`/`main` 分支; 上游代码会变,
> 引用时以文件路径为准、行号仅作导航。
> 服务对象: [mjx-go1-getup](../README.md) 的 Go2 地形策略 (obs 243 = 48 本体 + 187 扫描 + 8 相位,
> MLP 128x128 tanh, brax PPO 0.14.2, 2560 env, 动作=12 维绝对关节位置, 50 Hz)。

---

## 0. TL;DR (先给结论)

1. **观测规格不是短板**: 本项目 17x11=187 点 / 0.1 m 间距 / 1.6x1.0 m (x 偏前视 [-0.4,1.2]) 与
   legged_gym、IsaacLab、unitree 官方 Go2 完全同规格; 特征形式 (base_z - h - 标称站高, clip, x5)
   也等价于 legged_gym 的 `clip(root_z-0.5-h, ±1)*5`。**不要**再把扫描加密/放大当成第一杠杆。
2. **离散障碍的差距在"几何保真 + 课程 + 边缘/净空奖励 + 训练规模"**, 不在 obs 维度:
   - 开源 discrete 地形**要么是 1–2 m 大块** (legged_gym, 难度 0.05→0.23 m), **要么是连续 0.45 m 网格**
     (IsaacLab / mjlab)。**没有**开源配方直接用"0.4 m 格 + 50% 空心 + 8–15 cm 高"这种孤立小方块场,
     本项目的 boxes 比公开"discrete"地形更难 (每个箱子只有 0.4×0.4 m 可落面, 一半是坑)。
   - 两个权威实现都明确指出 **hfield 的台阶立面不是垂直的**: Rudin 2022 说 hfield 不可能有竖面,
     因此把低分辨率 hfield 转 trimesh 并"扶正"竖面; Miki 2022 更直接: 用 hfield 表示楼梯时
     "stair risers modeled by a height map are not perfectly vertical; we observed that the policy
     exploited these non-vertical edges in simulation, resulting in poor sim-to-real transfer",
     所以他们**用 box 几何搭楼梯**。本项目 boxes 地形 n=128 → 0.094 m/格, 0.08–0.15 m 台阶被抹成
     **50–58° 的斜坡**, 与"踩棱被绊"的诊断完全吻合。
3. **"踢立面"的显式惩罚确实存在**: extreme-parkour 同时开了 `feet_stumble = -1` (水平接触力 > 4×竖直力)
   和 **`feet_edge = -1`** (落点在"竖面脚点"掩码上, 掩码按 0.05 m 膨胀, 且仅 terrain_level>3 生效)。
   legged_gym 有 stumble 的形式但默认权重 **-0.0 (关)**; IsaacLab 默认没有 stumble/edge/clearance。
   → 这是最值得直接抄的一条。
4. **训练规模对比**: 本项目 v6a@7.37M / v7 +8M / v8 +10M 级; 开源 Go2/ANYmal rough 配方是
   **4096 env × 24 step × 1500 iter ≈ 147M** (legged_gym / IsaacLab), unitree mjlab Go2 设到
   `max_iterations=10001` (≈240M env-step @1024 env, 更多 @4096), 网络统一 **512-256-128 ELU**
   (IsaacLab 只有 flat 任务用 128x128x3)。本项目 128x128 是"平地档"的规格。
5. **可达上限估计 (推断, 非实测)**: 在"真竖面/box 几何 + 10 级课程 + 147M 级样本 + 512-256-128 +
   边缘/净空奖励"的忠实配方下, 同类 0.08–0.15 m、0.4 m 网格离散方块场 **85–92% 存活**是合理目标;
   只靠"加 boxes 面积 + 现有 128x128/7–17M"大概率停在 75–80%。**排序见 §6**:
   R1 几何保真 > R2 抬脚净空/落地 > R3 边缘+绊倒惩罚 > R4 逐 env 地形课程 > R5 网络/训练量。

---

## 1. 调研对象与来源

| 简称 | 仓库 | 本报告用到的关键文件 |
|---|---|---|
| legged_gym | <https://github.com/leggedrobotics/legged_gym> | `envs/base/legged_robot_config.py`, `envs/base/legged_robot.py`, `utils/terrain.py` |
| IsaacLab | <https://github.com/isaac-sim/IsaacLab> | `isaaclab/terrains/config/rough.py`, `.../locomotion/velocity/velocity_env_cfg.py`, `.../velocity/mdp/curriculums.py`, `.../config/anymal_c/agents/rsl_rl_ppo_cfg.py` |
| mjlab | <https://github.com/mujocolab/mjlab> | `src/mjlab/terrains/config.py`, `terrains/heightfield_terrains.py`, `terrains/primitive_terrains.py`, `src/mjlab/tasks/velocity/velocity_env_cfg.py` |
| unitree_rl_mjlab (官方 Go2, MuJoCo Warp) | <https://github.com/unitreerobotics/unitree_rl_mjlab> | `src/tasks/velocity/velocity_env_cfg.py`, `.../config/go2/env_cfgs.py`, `.../config/go2/rl_cfg.py`, `.../mdp/rewards.py` |
| unitree_rl_gym (官方 Go2, legged_gym 版) | <https://github.com/unitreerobotics/unitree_rl_gym> | `legged_gym/envs/go2/go2_config.py` |
| walk-these-ways (WTW) | <https://github.com/Improbable-AI/walk-these-ways> | `go1_gym/envs/go1/go1_config.py`, `go1_gym/envs/base/legged_robot_config.py` |
| extreme-parkour | <https://github.com/chengxuxin/extreme-parkour> | `legged_gym/envs/base/legged_robot_config.py`, `.../base/legged_robot.py`, `.../utils/terrain.py` |
| mujoco_playground | <https://github.com/google-deepmind/mujoco_playground> | `_src/locomotion/go1/joystick.py` |
| 论文 | Rudin et al. 2022 *Learning to Walk in Minutes*; Miki et al. 2022 *Learning robust perceptive locomotion for quadrupedal robots in the wild* | <https://arxiv.org/abs/2109.11978>, <https://arxiv.org/abs/2201.08117> |

---

## 2. Q1 —— 高度扫描型 (非深度) 策略的 obs 设计

### 2.1 对照表

| 实现 | 扫描点数 | 范围 / 间距 | 坐标系 | 特征 | 噪声 | actor 有足端/接触? | 历史帧 | 网络 |
|---|---|---|---|---|---|---|---|---|
| **legged_gym** (ANYmal-C rough) | **187** (17×11) | x∈[-0.8,0.8], y∈[-0.5,0.5], 0.1 m | base yaw | `clip(root_z-0.5-h, ±1)*5`; 每点取相邻 3 格 `min` | 0.1 m ×5 (均匀) | 否 (critic 也无) | 否 | 512-256-128 ELU |
| **IsaacLab** (ANYmal-C rough) | **187** | size=[1.6,1.0], 0.1 m | base yaw, raycast | `base_z - hit_z`, clip ±1 | U(-0.1, 0.1) | 否 | 否 | 512-256-128 ELU |
| **unitree_rl_mjlab Go2** | **187** | size=(1.6,1.0), 0.1 m, 射线 max 5 m | base_link, yaw | `frame_z - hit_z`, scale 1/5 | ±0.1 | 否; **critic 有** foot_height/air_time/contact/contact_force | 否 | 512-256-128 ELU + obs 归一化 |
| **unitree_rl_gym Go2** | **187** | 同 legged_gym | base yaw | 同 legged_gym | 0.1×5 | 否 | 否 | 512-256-128 ELU |
| **walk-these-ways Go1** | **关** (`measure_heights=False`) | — | — | — | — | 否 | 否 | 512-256-128 |
| **extreme-parkour** | **132** (12×11) | x∈[-0.45,1.2], y∈[-0.75,0.75], **0.15 m** | base | `root_z-0.3-h` | — | **是** (4 触地标志拼进 proprio) | **10 帧** proprio (n_proprio=53 → +530 维) | teacher MLP / student 深度 CNN |
| **mujoco_playground Go1** | **无 extero** | — | — | — | — | actor 否 (critic 有 contact/feet_vel/air_time) | 否 (纯本体) | 训练脚本另配 (非本表重点) |
| **本项目 v7/v8** | **187** | x∈[-0.4,1.2], y∈[-0.5,0.5], 0.1 m | base yaw | `clip(base_z-h-0.268, ±0.35)*5` | **0.02 m** (+ 雷达遮挡模型) | actor 否 / **critic 有** contact+air_time+foot_linvel | 否 | **128x128 tanh** |

来源:
[legged_gym 配置 L45-66](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py#L45-L66)、
[取高与 min-of-3 L761-812](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L761-L812)、
[obs 拼接 L209-226](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L209-L226)、
[IsaacLab rough 地形](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/config/rough.py)、
[IsaacLab 扫描器与 height_scan obs](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L66-L73)、
[unitree Go2 扫描器 + obs/critic 项](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/velocity_env_cfg.py)、
[unitree Go2 PPO 512-256-128](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/go2/rl_cfg.py)、
[WTW 关闭高度扫描](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/go1/go1_config.py#L61-L70)、
[extreme-parkour obs/history/scan 配置 L40-60, L150-200](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py#L40-L60)、
[playground 奖励/无 extero](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/joystick.py#L52-L95)。

### 2.2 逐项解读 (对离散障碍有意义的部分)

1. **扫描规格**: 三家主流 (legged_gym / IsaacLab / 官方 Go2) 都是 **1.6×1.0 m @ 0.1 m = 187 点**,
   0.1 m 间距对 0.3 m 台阶能采到 3 个点。extreme-parkour 反而用 **0.15 m 间距** (132 点)、
   y 方向扩到 ±0.75 m —— 它靠深度相机做学生策略, 高度扫描只是教师特权通道, 所以横向铺更宽。
   **结论: 本项目 0.1 m 间距不落后, 甚至比 EP 更密。**
2. **特征形式**: 全部是"传感器/base 高度 − 地形高度"的**相对量**, 再乘 5 后 clip。
   legged_gym 用 ANYmal 标称站高 0.5 作 offset; 本项目用 Go2 标称 0.268 —— 等价。
   legged_gym 每点取相邻 3 格 (2×2 的 3 个角) 最小值 (保守, 避免把悬空边看成平地), 本项目是单点解析采样;
   差别很小, 但"取 min"在台阶边缘会让观测偏高 (更保守), 属于可选细节。
3. **噪声**: legged_gym 高度噪声 0.1 m、IsaacLab U(-0.1,0.1)、unitree 0.1; 本项目 scan 噪声 **0.02 m**
   (但另有一套"雷达遮挡/盲区/丢点/记忆"模型)。就噪声幅度而言本项目偏小 5 倍; 若出现 sim2real
   高程图噪声问题, 可把 0.02 → 0.05 再配合遮挡模型, 但这属于后话。
4. **足端位置/接触进 actor obs**: **开源都不这么做**。legged_gym / IsaacLab / 官方 Go2 的 actor 只有
   本体 proprio + 扫描; 官方 Go2 把 `foot_height / foot_air_time / foot_contact / foot_contact_forces`
   只放进 **critic** (非对称 AC)。本项目已经是对称性训练 + critic 特权通道 (含 contact/air_time/
   foot_linvel), 与官方一致。→ 本项目 [v7 报告 §9.3](go2-v7-terrain.md) 里"obs 加足端位置"的候选**优先级应下调**。
5. **历史帧**: 只有 extreme-parkour 显式用 10 帧 proprio 历史 (`history_len=10`, 含 4 个触地标志);
   它这么做是为了让 **student (深度相机)** 从本体历史里推断接触/高程。纯高度扫描 + 非对称 AC
   的 legged_gym/IsaacLab/官方 Go2 都**不用**历史帧。→ 历史帧是"高成本、非必要"项。
6. **网络**: 四个 Go2/ANYmal rough 配方**全部 512-256-128 ELU** (legged_gym 默认、IsaacLab ANYmal
   rough、unitree_rl_gym、unitree_rl_mjlab Go2); IsaacLab 只有 **flat** 任务用 `[128,128,128]`。
   本项目 128×128 相当于"平地档"网络容量。

### 2.3 对本项目的判断

- **obs 已达标**, 不是 boxes 72% 的主因; 继续堆扫描点数/范围收益很小 (与 §2.1 的四家一致)。
- 真正与开源不同的地方: ① 网络/训练量小一个档; ② 没有历史帧 (但非必要); ③ 扫描噪声偏小;
  ④ 特征 clip ±0.35 m 比 legged_gym 的 ±1 m 窄 —— 本项目地形最高 0.30 m, ±0.35 足够,
  仅在"摔倒/骑在棱上"时饱和, 不影响正常行走。
- 参考实现位置: 本项目扫描配置 [envs/go2_walk.py L369-439](../envs/go2_walk.py#L369-L439),
  actor/critic 拼接 [envs/go2_walk.py L1552-1623](../envs/go2_walk.py#L1552-L1623)。

---

## 3. Q2 —— 地形难度递增与"离散障碍专项"课程

### 3.1 对照表

| 实现 | patch | 网格 | 难度映射 | 课程规则 | 离散/方块地形参数 |
|---|---|---|---|---|---|
| **legged_gym** | 8×8 m | **10 行 (难度) × 20 列 (类型)** | `difficulty = i / num_rows` (0, 0.1 … 0.9) | 走够 **patch 半边长 4 m** → level+1; 走不到 `0.5×cmd×T` → level−1; 满级后随机回落到某级 (防遗忘); `max_init_terrain_level=5` | **discrete 占 20%**: 20 条 1–2 m 矩形块, 高 **0.05 + 0.2·d** (0.05→0.23 m)。另有 stairs 60% (0.05+0.18d, 宽 0.31 m)、slope 20% |
| **IsaacLab** | 8×8 m | 10×20, horizontal 0.1 m | 同上 | 同上 (`terrain_levels_vel`) | **boxes 占 20%**: `MeshRandomGridTerrainCfg(grid_width=0.45, grid_height_range=(0.05,0.2))`; stairs 40% (0.05–0.23 m, 宽 0.3 m); rough 20% (0.02–0.10) |
| **mjlab ROUGH** (官方 Go2 用) | 8×8 m | 10×20, `curriculum=True`, `max_init=5` | 插值 difficulty | 同上 | flat 20 / stairs 20 (**0–0.1 m**) / inverted stairs 20 (0–0.1) / slopes 20 (0–1.0) / rough 10 / wave 10 |
| **mjlab STAIRS_TERRAINS_CFG** (专项) | 8×8 m | 10×4 | 多难度混合 | 同上 | flat 25% + easy stairs 35% (0.02–0.05) + moderate 25% (0.05–0.08) + challenging 15% (0.08–0.10) → **"专项"也是多难度混合, 不是单一种类** |
| **mjlab box_random_grid** | — | — | — | — | `grid_width=0.4`, `height 0–0.3`, **真 box geom**, 可 `merge_similar_heights` 合并相邻同高格减 geom |
| **mjlab hf_discrete_obstacles** | — | — | — | — | 40 个障碍, 宽 0.3–1.0 m, 高 0.05–0.3 m, **0.1 m hfield**, 高度在 ±h/2, ±h 中随机 (坑+凸混合) |
| **unitree_rl_gym Go2** | 8×8 m | 10×20 | 同 legged_gym | 同 legged_gym | 同 legged_gym (Go2 直接套用 ANYmal 的地形超参, 只改 `base_height_target=0.25`) |
| **extreme-parkour** | **18×4 m 走廊** | **10 行 × 40 列** | `difficulty = i/(num_rows-1)`; 非课程时随机 U(-0.2,1.2) | 同 legged_gym 的距离课程 + **goal 课程** (每 patch 8 个 goal, 到达即推进) | 5 种 parkour 各 20%: 台阶 `0.1+0.35d`, 栏架高 `0.1+0.1d ~ 0.15+0.25d`, 间隙 `0.1+0.3d`, 平台 `0.1+0.4d`; **horizontal_scale=0.05 m** |
| **本项目** | 4×4 m ×9 patch | 3×3, 行=难度 0.3/0.6/1.0, 列=类型 | 行固定 3 档 | **无 per-env 课程**: 随机出生 xy±5 m + 随机 yaw, 全体共享一张 hfield | boxes 0.4 m 格 **50% 空心**, 高 (0.05+0.07d)/行 → v8 0.078/0.101/0.132 m; 评测 boxes_only 最高 0.113 m |

来源:
[legged_gym 课程与难度公式 L85-140](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/terrain.py#L85-L140)、
[legged_gym 课程更新 L421-441](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L421-L441)、
[IsaacLab ROUGH_TERRAINS_CFG](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/config/rough.py)、
[IsaacLab terrain_levels_vel](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py)、
[mjlab 地形预设与 ROUGH/STAIRS 配置](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/terrains/config.py)、
[mjlab hfield 离散障碍 L689-800](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/terrains/heightfield_terrains.py#L689-L800)、
[mjlab box_random_grid L423+](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/terrains/primitive_terrains.py#L423)、
[unitree Go2 用 ROUGH + curriculum](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/go2/env_cfgs.py)、
[extreme-parkour 地形类型与难度 L150-200](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py#L150-L200)、
[EP parkour 地形生成 L152-315](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/utils/terrain.py#L152-L315)。

### 3.2 关键观察

1. **课程是"逐 env"的**: 每个环境自己维护 `terrain_level`, 按"走够距离就升级 / 走不够就降级"自动调节,
   满级后**随机回落到低难度** (legged_gym L437-440 原文: *"Robots that solve the last level are sent to a
   random one"*) —— 这是防遗忘的关键细节。本项目是"全体共享一张图 + 全图随机出生", 相当于**没有课程**:
   约 2/3 的环境出生在 difficulty 0.6/1.0 的 patch 上。
2. **"离散障碍专项"在开源里长什么样**:
   - legged_gym 的 discrete 是 **1–2 m 的大块** (人可在上面站), 不是 0.4 m 小方块;
   - IsaacLab 的 boxes 是 **0.45 m 连续网格** (每格都有高度, 高度 0.05–0.2 随机), 没有"空心";
   - mjlab 的 box_random_grid 也是 **0.4 m 连续网格** (0–0.3 m), 只是它用真 box geom;
   - 唯一"孤立柱/坑"的是 mjlab `hf_discrete_obstacles`: 40 个 0.3–1.0 m 宽、0.05–0.3 m 高的
     **hfield 柱体**, 且高度取 ±h/2、±h (坑和凸混合)。
   → **本项目的"0.4 m 格 + 50% 空心 + 0.08–0.15 m"比上述所有公开"discrete"都更稀疏、更碎**:
   每个箱子只有 0.4×0.4 m 的可落面 (脚掌有效接触半径仅 ~0.023 m), 相邻箱子之间是 0 m 地面,
   一旦落点偏出箱面几厘米就是"半个脚在棱上"。
3. **专项训练不等于"只训一种地形"**: mjlab 的 STAIRS_TERRAINS_CFG (唯一的"专项集") 仍是
   flat 25% + 三档台阶 75%。EP 的 parkour 课程也是 5 类混合各 20%。本项目 v7 用 4/9 面积给 boxes
   已经接近"专项"上限了, 实测只 +2~7pp 且回退 —— 与开源经验一致: **面积不是杠杆, 难度分配才是**。

### 3.3 对本项目的判断

- 现成生成器 [sim/gen_terrain_go2_mix.py L243-262](../sim/gen_terrain_go2_mix.py#L243-L262) 的
  **行 = difficulty 0.3/0.6/1.0**, 并且 `layout_meta` 已记录每个 patch 的 `row/type/difficulty/x0..y1`
  (L298-302) —— 这意味着**逐 env 课程只差 reset 里"按 level 选 patch 出生"这一步**, 不需要重做地形。
- 缺的不是"更多 boxes 面积", 而是: ① 让每个 env 先易后难; ② 在低难度阶段就能反复练"踩棱不摔";
  ③ 满级后随机回落防遗忘。

---

## 4. Q3 —— 奖励/约束: 净空、落脚、踢立面

### 4.1 对照表 (只列与离散障碍相关的项)

| 项 | 形式 | 权重 (实现) | 出处 |
|---|---|---|---|
| **feet_air_time** | `Σ(air_time - threshold)·first_contact`, 指令 >0.1 才生效 | legged_gym **1.0** (thr 由 `(air_time-0.5)` 硬编码); IsaacLab **0.125** (thr 0.5); playground **0.1** | [legged_gym L882-893](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L882-L893) |
| **feet_stumble** (踢立面) | `any(‖F_xy‖ > k·|F_z|)` 二值惩罚, 判"脚撞到竖面" | legged_gym: **k=5, 权重 -0.0 (默认关)**; extreme-parkour: **k=4, 权重 -1** | [legged_gym L895-898](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L895-L898) + [配置 L144](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py#L144); [EP L1272-1276](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot.py#L1272-L1276) |
| **feet_edge** (落在竖面脚点) | 落点 (x,y) 查 `x_edge_mask` → 若该格是"竖面起始格"且脚在接触, 每脚 +1 | **-1**, 且 `(terrain_levels > 3)` 才生效 | [EP L1278-1286](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot.py#L1278-L1286) |
| 上述 edge_mask 构造 | 把 hfield 转 trimesh 时, 沿 x 方向相邻格高差 > `slope_threshold` 的格标记为竖面脚点, 再用 **`edge_width_thresh=0.05 m`** 的竖条结构膨胀 (`binary_dilation`) | — | [EP terrain L85-91](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/utils/terrain.py#L85-L91), [L879-943](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/utils/terrain.py#L879-L943) |
| **foot_clearance** (摆动净空) | ① 相对**局部地形**的高度误差 × 脚水平速度: `Σ|foot_h - target|·‖v_xy‖`; ② 绝对 z 版: `Σ|foot_z - 0.10|` | mjlab 核心 **-1.0** (target 0.10, terrain-relative); unitree Go2 **-1.0** (target 0.10, 绝对 z); playground **-2.0** (max_foot_height 0.1) | [mjlab feet_clearance L241-268](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/mdp/rewards.py#L241-L268); [unitree 版 L163-185](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/mdp/rewards.py#L163-L185); [playground L520-530](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/joystick.py#L520-L530) |
| **feet_swing_height** (落点前峰值) | 摆动期记录峰值, 在 first_contact 时罚 `(peak/target - 1)²` | mjlab 提供类, weight 由任务定 | [mjlab L271+](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/mdp/rewards.py#L271) |
| **feet_slip** (触地滑移) | `Σ‖v_xy‖²·contact` | unitree Go2 **-0.25**; playground **-0.1**; 本项目 **-0.5** (v16 natural_gait 起) | [unitree L267-294](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/mdp/rewards.py#L267-L294) |
| **soft_landing** (落地冲击) | `Σ‖F‖·first_contact` | unitree Go2 **-1e-3** | [unitree L297-323](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/mdp/rewards.py#L297-L323) |
| 碰撞/非法接触 | 非足端触地惩罚或终止 | legged_gym `collision=-1`, `max_contact_force=100` (ANYmal-C rough 覆盖为 500); IsaacLab `undesired_contacts=-1` (thigh, thr 1 N) + base 非法接触终止; EP `collision=-10`, `max_contact_force=40`; unitree Go2 非足端触地 **>10 N 直接终止** | 见上表链接 |
| 姿态/机身高度 | `(base_h - target)²` 等 | legged_gym 默认 -0.0; EP 用 `base_height_target=1`; 本项目 `base_height=0` (v22 起只作诊断), `ascent` 显式爬升 | [本项目 envs/go2_walk.py L196-231](../envs/go2_walk.py#L196-L231) |

### 4.2 "踢到台阶立面"的显式项 —— 重点

- **只有 extreme-parkour 同时开了两项**: `feet_stumble=-1` (力方向判据, 无需地形图) 和
  `feet_edge=-1` (落点判据, 需要离线算一张 edge mask)。两项都是**二值**, 每步每脚最多 +1,
  所以 -1 的量级与 legged_gym 的 `collision=-1` 一致, 不会压死主奖励。
- `feet_edge` 的工程要点 (可直接照搬):
  1. 地形生成时顺手算 mask: 对 hfield 每个格, 若 `h[x+1] - h[x] > slope_threshold` (即"上台阶的竖面脚点"),
     标 1; 再沿 x 用宽 `edge_width_thresh=0.05 m` 的竖条 `binary_dilation` 膨胀 → 覆盖"脚掌半径"。
  2. 运行时用脚的 world (x,y) 除以 `horizontal_scale` 取整查 mask, 与 `contact` 相与。
  3. **门控**: 只在 `terrain_level > 3` 时启用 (EP 做法) —— 初级地形不罚, 避免早期学不会走路。
- legged_gym 的 stumble **默认是关的** (`feet_stumble=-0.0`), 说明"力判据"单独用效果不稳定
  (正常蹬地/上坡也会出现水平力); EP 把阈值从 5 降到 4 并配 edge 项后才开。**两者一起用**更稳。
- IsaacLab 的 velocity 模板里**没有** stumble / edge / clearance 任何一项, 只有 air_time 0.125 +
  undesired_contacts -1。所以"官方 IsaacLab 配方"并不覆盖本项目的失效模式 —— 这正是要参考
  legged_gym 作者另一条线 (EP) 与 mjlab (clearance) 的原因。

### 4.3 对本项目的判断

- 本项目已有 `feet_slip / feet_height / feet_impact / feet_phase / body_contact`, 缺的正是
  **"踢立面"的方向判据 (stumble)** 与 **"落在棱上"的落点判据 (edge)**; 另外 `max_foot_height`
  与 `gait_swing_height` 目前联动在 **0.15 m**, 而评测 boxes 最高 0.113 m、诊断摔倒处高 0.085 m ——
  摆动净空与障碍高度几乎相等, 属于"刚好不够"。
- 本项目奖励配置位置: [envs/go2_walk.py L112-231](../envs/go2_walk.py#L112-L231);
  参考实现在 [playground 同款 feet_slip/feet_height](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/joystick.py#L415-L432)。

---

## 5. Q4 —— 纯 height-scan + MLP 在 0.1–0.15 m 离散障碍上的证据与上限

### 5.1 直接证据

1. **Miki et al. 2022 (Science Robotics, ANYmal, 高度采样 + MLP teacher/student)**:
   - 固定高度台阶实验 (12 cm → 36.5 cm, 每档 10 次): **纯本体感觉基线在 20 cm 就开始频繁卡住**
     ("success rate dropped at 20 cm step height when the front legs started frequently getting stuck");
     **带高度扫描的策略稳定通过到 30.5 cm**; 32 cm 以上策略会"犹豫不走" (学到那是物理极限)。
   - 障碍课程: 20 cm 平台 / 17 cm 高 29 cm 深楼梯 / 20 cm 立方块, 带扫描策略"无协助平滑通过",
     基线 "卡在全部三种障碍上"。
   - 关键工程结论 (与本项目直接相关): *"We use boxes to form the stairs, because stair risers modeled by
     a height map are not perfectly vertical; we observed that the policy exploited these non-vertical
     edges in simulation, resulting in poor sim-to-real transfer."*
   → 这是"**高度扫描 + MLP 足以处理 ~0.2 m 级离散障碍**"的最强公开证据, 但前提是**几何是竖面 (box)**。
   来源: <https://arxiv.org/abs/2201.08117> (Fig. 4 与 §"Experiments"), 项目页
   <https://leggedrobotics.github.io/rl-perceptiveloco/>。
2. **Rudin et al. 2022 (legged_gym 的论文, ANYmal, 高度采样 108 点 + MLP 512-256-128)**:
   - 课程把台阶和随机障碍高度从 **5 cm 逐步升到 20 cm**; 成功率实验 (Fig. 5):
     **楼梯到 0.2 m 约 100%**; **离散障碍"成功率随难度持续下降"** (原文: *"Randomized obstacles seem to
     be more demanding, with the success rate decreasing steadily"*), 且"报告的障碍高度翻倍,
     因为相邻障碍可以一正一负" (峰峰值最大 0.4 m); 坡度 25° 以上上不去。
   - 训练量: 4096 robots × 24 steps × 1500 updates ≈ **147M env-step**, 单卡 <20 min (A6000);
     观测 = 本体 + **108 个地形高度采样** (论文数; 当前仓库默认配置是 187 点, 属版本差异,
     两者都在 0.1 m 级)。
   - 关键工程结论: *"A height field ... it is impossible to get vertical surfaces. In order to get a steep
     slope approximating a vertical step, we need high resolution, which degrades the simulation
     performance. Instead, we convert a low-resolution height field to a triangle mesh and correct the
     vertical surfaces."*
   → 说明"竖面几何"在 legged_gym 是**显式处理**过的, 不是 hfield 天然给的。
   来源: <https://arxiv.org/abs/2109.11978> (Fig. 2/5, 附录)。
3. **extreme-parkour**: 用 **0.05 m** 地形分辨率 + hfield→trimesh 的**竖面扶正** (`slope_treshold=1.5`) + 深度相机学生, 台阶 0.1→0.45 m、
   栏架 0.1→0.4 m 级; 高度扫描只作为教师特权通道 (132 点 @0.15 m)。
   → 说明"更高/更陡的离散障碍"这条路要靠**更细的几何 + 深度**, 但它的奖励项 (stumble/edge) 和
   课程机制可以搬到高度扫描策略上。来源: <https://github.com/chengxuxin/extreme-parkour>。

### 5.2 本项目 boxes 的难度定位

- 站立高度 0.268 m 的 Go2 上, 评测 boxes 最高 **0.113 m ≈ 42% 站高**, 训练 v8 最高 0.132 m ≈ 49% 站高;
  Miki 的 30.5 cm 台阶对应 ANYmal ~0.55–0.6 m 站高 ≈ **50–55%**。**相对高度处于同一量级**,
  但本项目是 **0.4 m 孤立小方块 (50% 空心)**, 每个箱子只够放一只脚, 落脚容错远小于 ANYmal 的
  连续台阶/20 cm 立方块 (20 cm 立方块也比 0.4 m 格宽)。
- 本项目实测: boxes 72%(nominal), 摔倒 96% 是 tilt, 摔倒处地形高 p50 0.085 m / 局部坡度 p50 0.24;
  v7 加 boxes 面积只 +2~7pp 且回退。→ 与 Rudin "离散障碍比楼梯难得多、成功率持续下降"一致。

### 5.3 上限估计 (明确标注: 推断, 非实测)

在**忠实配方** (真竖面/box 几何 + 逐 env 10 级课程 + ~150M 样本 + 512-256-128 + stumble/edge/clearance)
下, 同类"0.4 m 格、50% 空心、0.08–0.15 m"方块场, 合理目标是 **85–92% 存活**; 现有配方
(128x128 / 7–17M / hfield 斜边 / 无 edge/clearance / 无课程) 的**现实上限约 75–80%**。
关键因素排序 (按证据强度):

1. **竖面几何保真** (Rudin/Miki 两篇都点名; 本项目 0.094 m 格把 0.113 m 台阶抹成 50° 坡);
2. **抬脚净空 vs 障碍高度** (本项目 0.15 m 摆动 vs 0.113 m 障碍, "刚好不够"; Miki 靠提前抬腿);
3. **边缘/绊倒的显式惩罚** (EP 同时开两项; 本项目完全没有);
4. **逐 env 课程 + 满级回落** (四个开源实现的共同做法; 本项目没有);
5. **网络容量 + 样本量** (全部开源 rough 配方 512-256-128 + 147M–1B; 本项目 128x128 + 7–17M);
6. 扫描分辨率/范围 (已达标, 排最后)。

---

## 6. 建议 (按推荐优先级排序: 先做"便宜且对症", 大改排在确认前几步不够之后; 每条可直接落到本项目文件)

> 成本按本项目实测吞吐 ~4.3–4.9k env-step/s 估算: 10M ≈ 35–40 min, 40M ≈ 2.3–2.6 h。
> 所有"预期收益"都是**推断**, 必须用同进程同批 384 env × 6 s 口径验证 (见
> [docs/go2-v7-terrain.md §9.1](go2-v7-terrain.md))。

### R1 (最高) 让离散障碍拥有"垂直立面": box 几何替代 hfield 方块

- **改动点**:
  - `sim/gen_terrain_go2_mix.py`: boxes patch 不再写进 hfield, 而是输出 **box geom 列表**
    (0.4 m 格, 50% 保留, 每 box 高度 = 该格随机高度, box 底面落在 h=0)。参考 mjlab
    `BoxRandomGridTerrainCfg` ([L423+](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/terrains/primitive_terrains.py#L423)),
    0.4 m 格正是它的默认 `grid_width`; 还提供 `merge_similar_heights` 把相邻同高格合并减 geom。
  - MJCF 场景: 以 v7 的 4 个 boxes patch 为例 ≈ 4×100×50% = **200 个 box geom** (v8 是 3 个 patch ≈150), 加一个 hfield 只负责
    smooth/rough/slope patch (boxes 区域 hfield 置 0)。
  - 高度采样: boxes 区域改为**解析式** `boxes_height(x,y)` (查格 + 顶面高度), 与现有
    `terrain_height()` 并列; scan 语义不变 (仍是"base_z - h - 0.268")。
  - 评测/CPU 侧同一份 MJCF 即可 (box geom 是 MuJoCo 一等公民, CPU/warp 都有)。
  - 若不想一次到位: **最低成本的 80% 版本**是把 boxes 图分辨率从 n=128 → **n=256**
    (0.047 m/格; 0.113 m 台阶的等效坡从 50° 变 67°), 只改生成命令
    (`--n 256`), 重新训练。证据: Rudin 论文"竖面需要高分辨率, 但会牺牲性能"。
- **预期收益**: boxes **+5~10pp** (推断); 同时消除"策略利用斜边"导致的 sim2real 落差,
  CPU/warp 接触一致性也会改善 (本项目已知的 warp↔CPU hfield 接触分歧)。
- **成本**: 中高。生成器 + env 高度查询改动; **不能续训**, 需从零 ~40M (2.3–2.6 h) 或先用
  n=256 的便宜版本 (只重生成 + 从 v6a 续训 10M)。
- **风险**: warp 下 200 个静态 box 的 broadphase/接触开销 (先跑一个 300 步 perf probe);
  box 竖面比斜边"更硬", 若净空/edge 奖励不同步上, 短期可能先变差 (建议 R2/R3 一起上);
  纯 box 地形不覆盖"箱子之间 0 m 地面"到 hfield 的过渡 (用 hfield=0 保证一致)。
- **参考**: [mjlab box 地形](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/terrains/primitive_terrains.py#L423)、
  [Rudin §"terrain resolution"](https://arxiv.org/abs/2109.11978)、
  [Miki "boxes for stairs"](https://arxiv.org/abs/2201.08117)。

### R2 抬脚净空"高于障碍" + 落地变软 (奖励/配置级, 最便宜)

- **改动点**:
  - `gait_swing_height`/`max_foot_height`: **0.15 → 0.20 m** (Go2 站高 0.268, 0.20 仍在可达范围);
    或做**地形自适应**: `target = 0.15 + clip(max(scan 前视区) - 0.268 + 0.02, 0, 0.08)`, 让看到箱子时
    只抬到"箱顶 + 2 cm", 平地上不浪费。
  - 新增 `feet_clearance`: 复用 mjlab 的"相对局部地形 + 脚水平速度加权"形式
    ([L241-268](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/mdp/rewards.py#L241-L268)),
    target **0.10–0.12 m**, 权重 **-0.5 ~ -1.0** (本项目已有 `_cost_feet_height`, 只需改成
    "相对地形"而非绝对高度, 并只在摆动期结算)。
  - 新增 `soft_landing = -1e-3` (unitree Go2 同款) 防"砸地"。
  - 与 `feet_phase` 的联动保持不变 (相位奖励的目标高度同步到新 swing height)。
- **预期收益**: boxes **+3~6pp** (推断), 直接对症"抬脚净空不足以清过箱棱"。
- **成本**: 低。纯配置+奖励函数, 从 v6a 续训 8–10M (~40 min), 可多组权重并行扫 (同进程)。
- **风险**: 摆动变长 → 步频/速度下降、`feet_impact` 变大; 用 soft_landing 和 0.5 s 固定相位周期
  对冲; 若速度指令跟踪掉 >5%, 回退到 0.18 m。
- **参考**: [playground clearance/feet_height](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/joystick.py#L520-L540)、
  [unitree Go2 foot_clearance/soft_landing 配置](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/velocity_env_cfg.py)。

### R3 抄 extreme-parkour 的 `feet_stumble` + `feet_edge` (专治"踢立面/落在棱上")

- **改动点**:
  - `_cost_stumble`: `any(‖F_xy‖ > 4·|F_z|)` × **-1** (用现有足端接触力; 阈值取 EP 的 4, 不是 legged_gym 的 5)。
  - `_cost_feet_edge`: 生成地形时算一张 **edge mask** (逐格: 沿 x/y 高差 > 阈值的"竖面脚点", 按
    `edge_width_thresh=0.05 m` 膨胀), 存进 `terrain_*.json` 的 meta; 运行时脚 (x,y) 查表 × contact × **-1**。
  - **门控**: 只在 `terrain_level > N` 或该 patch difficulty ≥ 0.6 时生效 (EP 的 `terrain_levels > 3`;
    本项目可先用"所在 patch difficulty ≥ 0.6"近似, 与 R4 天然配套)。
  - 加诊断日志: 每 episode 的 `feet_at_edge` 次数 (EP 也记这个)。
- **预期收益**: boxes/stairs **+2~5pp** (推断), 且学到"抬脚过棱/落脚离棱"而不是"硬刚"。
- **成本**: 低。地形 mask 离线 numpy 一次算完; 奖励两项 ~30 行; 续训 8–10M (~40 min)。
- **风险**: 正常 toe-off/上坡蹬地也带水平力 → 过度惩罚会让步态保守、速度下降; 先用权重 -0.3 起,
  扫 {-0.3, -0.5, -1.0}; edge 惩罚可能诱导"绕开箱子", 需与 ascent / 速度跟踪一起看。
- **参考**: [EP `_reward_feet_stumble`/`_reward_feet_edge` L1272-1286](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot.py#L1272-L1286)、
  [EP edge mask 构造](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/utils/terrain.py#L85-L91)。

### R4 逐 env 地形课程 (用现成的"行=难度"布局, 不重做地形)

- **改动点**:
  - `info["terrain_level"]` ∈ {0,1,2} 对应现成 3 行 difficulty **0.3 / 0.6 / 1.0**;
    reset 时: `level` 决定 patch 行, 列在 3 个类型里按固定或随机选, 再在 patch 内随机 xy/yaw
    (保留 `random_init.spawn_z_offset` 与 slope 过滤)。
  - 升级/降级照抄 legged_gym: **走过 > patch 半边长 (2 m) → level+1**;
    **走不到 `0.5×cmd×T` → level−1**; **满级后随机回落到 0/1** 防遗忘
    ([L421-441](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L421-L441))。
  - 增加 `logs` 里 level 分布直方图 (诊断课程是否饱和)。
  - 评测口径**不改**: eval 仍随机出生全图 (训练课程只影响训练)。
  - **多种子**: 每个课程阶段换一次 `--seed` 重新生成 hfield (现有 `--seed` 参数), 避免背地图。
- **预期收益**: 高难度地形 **+3~8pp** (推断); 更重要的是**可以在不放 4/9 面积给 boxes 的前提下**
  让高难度 boxes 被充分训练 → 缓解 v7 式 mix/stairs 回退; 与开源四家做法一致。
- **成本**: 中。只改 reset 的出生逻辑 (jax 里 per-env level + `where` 即可, 不动物理), 重训 10–20M
  (~1–1.5 h) 或从 v6a 续训。评测脚本需保留"非课程随机出生"路径以便对比。
- **风险**: 2560 env / 9 patch → 每 patch ~280 env, 密度足够; 但课程早期全部 env 挤在 easy 行,
  easy 行只有 1/3 面积 (16 m²) → 出生点重叠 (可用 patch 内 jitter + 出生 slope 过滤);
  升级判据 (距离) 在随机 yaw 下会偏 (建议用 |Δpose| 或"到达 patch 边界"更稳)。
- **参考**: [legged_gym 课程实现](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L421-L441)、
  [IsaacLab terrain_levels_vel](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py)、
  [mjlab STAIRS_TERRAINS_CFG "专项也是多难度混合"](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/terrains/config.py)。

### R5 (可选, 最后) 网络 128x128 → 512-256-128 + 30–50M 步

- **改动点**: actor/critic `[128,128]` tanh → **`[512,256,128]` ELU** (四个开源 rough 配方一致);
  训练步数从 7–17M 提到 30–50M; 注意 brax PPO 的 batch = envs × unroll 也要同比放大
  (参考 legged_gym batch≈98k–200k 样本)。
- **预期收益**: **+2~5pp** (推断); 128x128 对 243 维 obs (其中 187 维是空间扫描) 表达力偏紧。
- **成本**: 中。改网络配置 + 2–4 h 训练; 需要重跑同进程对比。
- **风险**: 大网络 + 小 batch 容易不稳 (brax PPO 默认超参未针对 512 宽调); 收益可能被 R1–R4 覆盖,
  所以排最后。
- **参考**: [legged_gym PPO L202-234](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py#L202-L234)、
  [IsaacLab ANYmal rough 512-256-128 (flat 才 128x128x3)](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/agents/rsl_rl_ppo_cfg.py)、
  [unitree Go2 512-256-128](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/go2/rl_cfg.py)。

### 6.1 不建议做 (省时间)

- **再加扫描点数/范围**: 已与四家开源同规格 (§2), EP 甚至更疏; 收益最低。
- **继续把 boxes 面积加大**: v7 实测 +2~7pp 且回退; 开源"专项"也是多难度混合, 不是单点加码。
- **actor obs 加足端位置/接触**: 开源 actor 都不加; critic 已有特权通道, 与官方 Go2 一致。
- **actor obs 加 10 帧历史**: 只有 EP 为深度学生加; 成本高 (obs 翻数倍), 收益不确定。
- **换地形生成框架/上深度相机**: 超出本项目交付路径 (MJX/brax + 高度扫描), 除非 R1–R4 用尽。

---

## 7. 参考链接汇总 (可点击)

- legged_gym: <https://github.com/leggedrobotics/legged_gym>
  - 地形/课程/难度公式: [utils/terrain.py L85-140](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/terrain.py#L85-L140)
  - 课程更新: [envs/base/legged_robot.py L421-441](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L421-L441)
  - 高度采样与 obs: [L761-812](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L761-L812), [L209-226](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L209-L226)
  - stumble: [L895-898](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py#L895-L898)
  - 网络/PPO: [config L202-234](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py#L202-L234)
- IsaacLab: <https://github.com/isaac-sim/IsaacLab>
  - ROUGH_TERRAINS_CFG: [terrains/config/rough.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/config/rough.py)
  - 1.6x1.0@0.1 扫描 + obs + 奖励: [velocity_env_cfg.py L66-73, L134-139, L234-263](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
  - 课程: [mdp/curriculums.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py)
  - ANYmal rough 网络: [config/anymal_c/agents/rsl_rl_ppo_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/agents/rsl_rl_ppo_cfg.py)
- mjlab: <https://github.com/mujocolab/mjlab>
  - 地形预设 (ROUGH/STAIRS/ALL): [src/mjlab/terrains/config.py](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/terrains/config.py)
  - hfield 离散障碍: [heightfield_terrains.py L689-800](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/terrains/heightfield_terrains.py#L689-L800)
  - 0.4 m box 网格: [primitive_terrains.py L423+](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/terrains/primitive_terrains.py#L423)
  - 相对地形净空奖励: [tasks/velocity/mdp/rewards.py L241-268](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/mdp/rewards.py#L241-L268)
- unitree_rl_mjlab (Go2 官方): <https://github.com/unitreerobotics/unitree_rl_mjlab>
  - obs/critic/奖励: [src/tasks/velocity/velocity_env_cfg.py](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/velocity_env_cfg.py)
  - Go2 特化: [config/go2/env_cfgs.py](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/go2/env_cfgs.py)
  - 网络: [config/go2/rl_cfg.py](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/go2/rl_cfg.py)
- unitree_rl_gym (Go2 legged_gym 版): [go2_config.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)
- walk-these-ways: [go1_config.py (measure_heights=False)](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/go1/go1_config.py#L61-L70)
- extreme-parkour: <https://github.com/chengxuxin/extreme-parkour>
  - obs/history/扫描/地形类型: [legged_robot_config.py L40-60, L150-200](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py#L150-L200)
  - stumble/edge 奖励: [legged_robot.py L1272-1286](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot.py#L1272-L1286)
  - edge mask: [utils/terrain.py L85-91, L879-943](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/utils/terrain.py#L85-L91)
- mujoco_playground: [go1/joystick.py](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/joystick.py)
- 论文: [Rudin 2022](https://arxiv.org/abs/2109.11978)、[Miki 2022](https://arxiv.org/abs/2201.08117)、
  [项目页](https://leggedrobotics.github.io/rl-perceptiveloco/)
- 本项目相关文档: [地形现状调研](go2-terrain-survey.md)、[鲁棒性计划](go2-terrain-robustness-plan.md)、
  [v7 boxes 专项报告](go2-v7-terrain.md)、[交付/评测](go2-delivery.md)、[评测框架](go2-eval-harness.md)

---

## 8. 局限与待验证

1. **收益数字全部是推断**: 没有在本项目里做 A/B; 落地时必须用"同进程、同批 384 env × 300 步、
   随机出生、nominal 遮挡"口径 ([docs/go2-v7-terrain.md §9.1](go2-v7-terrain.md) 已说明跨进程
   绝对值可差 4pp)。
2. **R1 的 warp 开销未实测**: 200 个静态 box 在 MuJoCo Warp 下的 broadphase/接触成本需要 300 步
   perf probe 验证; 若过慢, 退回"n=256 hfield"版本。
3. **R1 改变任务定义**: 竖面会让"踩棱"从"骑斜坡"变成"真撞墙", 短期可能更差; 必须先上 R2/R3。
4. **open-source 数字与 Go2 不完全可比**: legged_gym/IsaacLab 的主体是 ANYmal (~50 kg, 腿长 ~0.55 m,
   IsaacLab 用 PhysX trimesh), 官方 Go2 配方的台阶高度被压到 0–0.1 m (mjlab ROUGH) —— 说明
   "官方 Go2 也是按小台阶配的"; 本项目的 0.113–0.15 m boxes 已在其上, 属于要自己探的上限区。
5. **未覆盖**: 深度相机/视觉方案 (extreme-parkour 学生、Barkour)、LSTM/Transformer 历史编码、
   域随机化细节; 这些与"仅高度扫描 + MLP"的交付路径无关, 若后续要上视觉再调研。
