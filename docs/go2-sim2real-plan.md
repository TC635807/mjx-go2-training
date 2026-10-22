# Go2 sim2real 训练准备（v1）

> 2026-10-17。目标：在 `mjx-go1-getup` 这套 MJX/warp + brax PPO 训练栈上**重训 Go2** 的
> 走路 + 起身策略，训练完的策略要直接落进 `/home/tc63/mujoco/mjx-go2-mujoco2ros2`
> （MuJoCo ↔ ROS 2 / DDS 双向力矩闭环，50 Hz 策略 / 500 Hz 电机层）。
>
> 本文所有"训练侧事实"都以两个仓库的源码为准，凡引用都给了文件:行。带 ★ 的是我本轮实测出来的数字。

---

## 0. 一页纸

**Go1 已经证明的事**（照抄）：动作锚定名义站姿比增量式好训（0% → 80%）；判据口径比策略本身
更能左右成功率（`up_z` 0.99 vs 0.95 差一个数量级）；交还时**不要混合**两个策略的动作；
奖励分项必须实测标定；"先量化再改"。

**Go1 没做好、Go2 必须补的**（★ = 本轮实测）：

| # | 问题 | 证据 | Go2 要做的 |
|---|---|---|---|
| 1 | 动作空间退化成 bang-bang | ★ 下发目标 89% 顶在关节限位上，原始 `|a|` 中位 12.7（clip=8），站着时关节速度 4 rad/s | 动作**按关节行程归一化 + tanh**，不再用全局 clip |
| 2 | 训练判据 ≠ 部署判据 | ★ 末帧占用：训练口径 96.1% vs 部署口径 64.8% | 训练/评估/部署共用**同一个** `stand_ok()` |
| 3 | 地形是"完美等距真值高度" | 部署侧只能给点云（★ 平地 35 格里只有 9 格可见） | 训练侧**虚拟雷达 → 高程图**，与部署 `perception.py` 同语义 |
| 4 | 时序理想（零延迟、无丢帧） | 部署实测 p95 抖动 4.7 ms，曾有 ~20 ms 陈旧观测 | 延迟队列 + 观测延迟 + 一阶执行器滞后 |
| 5 | 执行器是理想 PD | 真机有力矩-转速包络、增益误差 | DC 电机包络 + kp/kd 随机化 |
| 6 | 出界虚空吃掉 ~10% 样本 | ★ `|xy|>5.5` 占 10.4% env-step | 出生范围留 ≥1.8 m 余量 / 加边界 |
| 7 | 爬台阶激励只占 1% | ★ `ascent` = +0.035/步（总 +3.48） | 势能差形式 + 地形课程 + 出生分布 |
| 8 | 换机型=换 home/范围/接触 | Go1 策略直接跑 Go2：1.24 m/12 s（Go1 是 5.0 m） | 全部按 Go2 重标 |

**结论**：Go2 不是"把 XML 换掉重跑一遍"，而是**在同一个奖励/训练框架上补三件部署侧已经
量化的东西**：感知（点云）、时序（延迟）、执行器（包络）。这三件在 `mjx-go2-mujoco2ros2`
里**都已经有参考实现**（`perception.py` / `domain_gap.py` / `config/domain_gap.json`），
训练侧只要"镜像"它们。

---

## 0.1 决策记录（2026-10-17，已确认）

| # | 问题 | 决定 | 直接后果 |
|---|---|---|---|
| 1 | 传感器 | **Go2 自带 Unitree 4D LiDAR L1** | 感知参数按 L1 实测填（§3.C）：**360°×90°、11 Hz、有效 21600 点/s（≈1964 点/帧）、竖直 180 Hz、精度 ±2.0 cm、近处盲区 0.05 m、30 m@90%**。框架 `perception` 段的 `vfov=[-0.5,0.2]` **必须改**（那是窄视场假设） |
| 2 | 算力 | **机器人算力盒（Jetson Orin，官方 40–100 TOPS）** | 策略 128×128 前向 <1 ms 无压力；**真正要 benchmark 的是点云→高程图**（~2000 点 × 11 Hz）。`scripts/bench_io.py` 的拆账要在算力盒上重跑 |
| 3 | 系统辨识 | **无真机** | 执行器只能用公开参数（23.7/30.1、45.43/15.70）+ 保守 DR 覆盖误差，**不能**做 ActuatorNet。⚠️ 那两个模型矛盾（damping 0.1 vs 2、膝 45.43 vs 35.55）也只能靠 DR 覆盖（见 §3.B 备注）|
| 4 | 感知依赖 | **必须依赖雷达；雷达失效 → 按平地盲走退化** | 采用**单策略**方案：把"地图质量"当 DR 维度（Miki 60/30/10），**不做双策略切换** → 部署框架零逻辑改动。框架 fill 默认 0.0 恰好等价于"假设平地"（z=0.268 时 feature=0），与退化语义天然一致 |
| 5 | 指令范围 | **适当扩大** | 建议 vx ∈ [0, 1.0] m/s、vy ±0.4、wz ±1.5 rad/s（reward/gait_freq 按此重标）；⚠️ **所有 eval 的 cmd 必须落在训练区间内**（Go1 的 eval_stairs 用 0.6 而训练只到 0.5）|
| 6 | home 姿态 | **开源惯例** | 稳态实测 **z = 0.268 m** → 新 `scan_base` / `base_height_target` / 起身高度带全部按 **0.268** 标定（不是 Go1 的 0.28，也不是 get-up-isaaclab 的 0.30）。训练 home ≠ 框架 keyframe，`prepose_s` 必须摆到训练 home |

---

## 1. 现状盘点

### 1.1 训练侧（mjx-go1-getup）可复用的骨架

| 资产 | 位置 | Go2 怎么办 |
|---|---|---|
| 走路环境（地形/奖励/终止/obs） | `envs/go1_walk.py` | 复制成 `envs/go2_walk.py`，改模型/几何/home/常量 |
| 起身环境 v2 | `envs/go1_getup_v2.py` | 同上（配方保留，动作空间重写） |
| brax PPO 训练入口 | `train/train_go1.py` / `train_getup.py` | 直接复用（只改默认机型/元数据落盘） |
| 地形生成器 | `sim/gen_terrain.py` | 复用，另加斜坡/箱子/沟壑；Go2 站高 0.27 ≈ Go1 0.278，特征高度可直接沿用 |
| 评估/探针 | `sim/eval_*.py` / `probe_handover.py` | 复用，阈值改成 Go2 |
| 本轮新加的动作空间体检 | `sim/audit_action_space.py` | M0 就在 Go2 上跑一遍 |

### 1.2 部署侧（mjx-go2-mujoco2ros2）已经就绪的接口 —— 这就是训练要对接的契约

| 事实 | 位置 | 对训练的含义 |
|---|---|---|
| 50 Hz 策略 + 500 Hz 电机层，`sim_dt=0.002` | `config/go2_sim.json` | 训练保持 `ctrl_dt=0.02 / sim_dt=0.002`（已一致） |
| LowCmd 语义 `tau = kp(q_des−q) + kd(0−dq) + tau`，策略档 **kp=20 / kd=1 / tau=0** | `config/go2_sim.json:policy`，`docs/interfaces.md §1` | 训练用 `<position kp=20 kv=1>` 即等价；Go2 MJCF 自带 damping 0.1 → 有效阻尼 ≈1.1 |
| 观测 91 维 = 48 base + 35 scan + 8 phase，逐段对齐训练 `_get_obs` | `docs/interfaces.md §1.1`、`docs/obs-parity.md` | **建议保持这个布局**，只换 home/base/命令坐标系 |
| 点云 → 高程图 → 35 点扫描（仿真/真机同一份代码） | `src/go2_bridge/perception.py` | 训练侧的虚拟雷达**必须复刻它的语义**（5 cm 格、max-z、decay 1 s、未知格 hold-fill 4 格、缺格双线性重归一） |
| 雷达默认参数：2000 线 / 8 m / 安装高 0.3 m / vfov [-0.5, 0.2] rad | `config/go2_sim.json:perception` | 训练侧随机化这几个量（含安装误差） |
| 域差清单 17 条（带证据） | `config/domain_gap.json` | 见 §1.3 的映射表 |
| 信号劣化注入（延迟/丢帧/噪声/偏置/量化） | `src/go2_bridge/domain_gap.py` | 部署侧的"降级档"，训练侧的 DR 要覆盖同名参数 |
| Go1 策略在 Go2 上：1.24 m/12 s，且对 height scan 来源敏感 | `docs/policy-port.md §4/§5` | 必须重训 |

### 1.3 域差 → 训练侧动作项（DG 编号来自 `config/domain_gap.json`）

| DG | 内容 | 部署侧状态 | 训练侧要做的 |
|---|---|---|---|
| DG-01 | 质量 12.74 → 15.21 kg（trunk 5.20 → 6.92） | open | 用 Go2 模型训练即可关闭；另加质量随机化 ×U(0.9,1.1) |
| DG-03 | 关节 damping 1.0 → 0.1（Go2 模型） | mitigated（部署 kd 补） | 训练模型保持 0.1 + 部署同款 PD；damping/frictionloss 随机化 |
| DG-06 | 高度扫描：真值 → 点云（盲区/量化/噪声） | **open** | §3.C 虚拟雷达 |
| DG-10/14 | IMU/观测噪声、量化 | mitigated | 观测噪声 DR（高斯+偏置+量化，别只用均匀） |
| DG-11 | 观测延迟/丢帧 1–3 拍 | mitigated | §3.A 延迟队列 |
| DG-12 | 关节范围不同（Go2 前后大腿不对称） | mitigated | 逐关节动作参数（§2.2）；对称正则要按 Go2 重写 |
| DG-13 | 策略推理 3–8 ms | mitigated | 网络保持 128×128（足够小）；不需要 GRU 的话就别加 |
| DG-15 | 标称站高/height_scan base | mitigated | **实测 Go2 沉降站高**，训练与部署同值（唯一入口参数） |
| DG-08 | PD 控制律 | closed | 训练 kp=20/kv=1 对齐 |
| 新增 | 力矩-转速包络、一阶滞后、kp/kd 误差 | 记在 README「理想力矩源」 | §3.A |

---

## 2. Go2 训练契约（新的"唯一真相"）

### 2.1 模型：`models/go2/go2_position.xml`

从框架的 `models/go2/go2.xml`（unitree_mujoco 官方 Go2 MJCF，BSD-3）机械变换而来：

| 项 | Go1 训练模型 | **Go2（本方案）** |
|---|---|---|
| 髋 abduction | ±0.863 | **±1.0472** |
| 大腿 | 统一 [-0.686, 4.501] | **前 [-1.5708, 3.4907] / 后 [-0.5236, 4.5379]（不对称！）** |
| 膝 | [-2.818, -0.888] | **[-2.7227, -0.83776]** |
| 力矩上限 | 髋/大腿 ±23.7，膝 ±35.55 | **髋/大腿 ±23.7，膝 ±45.43** |
| damping / frictionloss / armature | 1.0 / 0.1 / 0.01 | **0.1 / 0.2 / 0.01** |
| 足端 | 球 r=0.023, z=-0.213, friction 0.6 | **球 r=0.022, pos(-0.002,0,-0.213), friction 0.4** |
| home（action=0 的姿态） | [0,0.8,-1.5]×2 + [0,1.0,-1.5]×2 | ✅**已定：开源惯例** = [hip ±0.1, 大腿 0.8(前)/1.0(后), 膝 −1.5]（unitree_rl_gym / IsaacLab）。**本机只读实测**（Go2 MJCF + 手写 PD kp=20/kd=1，CPU MuJoCo，3 s 沉降）：稳态 **z = 0.268 m**、倾角 **1.58°**、|τ|max **6.3 N·m**（上限 23.7）、**ncon = 4**（干净四足）。对照 MJCF keyframe [0,0.9,−1.8]：**z=0.20 m、倾角 6.9°、ncon=6**（多 2 个接触）→ **keyframe 那个 home 在 kp=20 下不是好站姿，框架的 stand_pose 建议一起换** |
| keyframe z | 0.35 | **0.27** |
| 质量 | trunk 5.204 / 总 12.743 | **trunk 6.921 / 总 15.206** |
| 关节 damping / 足端摩擦 | 1.0 / 0.6 | ⚠️ **两份公开 Go2 模型互相矛盾**：unitree_mujoco（= 我们框架用的）damping **0.1**、足摩擦 **0.4**；MuJoCo Menagerie go2.xml damping **2**、足摩擦 **0.8**。这个差异直接决定站立 PD 与接触行为，**M0 必须用手册/真机确认**（DG-03 就是按 0.1 写的） |
| 膝力矩/转速 | ±35.55 N·m | ⚠️ **两份公开 Go2 URDF 也矛盾**：unitree_ros `go2_description` = **45.43 N·m / 15.70 rad·s⁻¹**；unitree_rl_gym `resources/.../go2.urdf` = **35.55 N·m / 20.07 rad·s⁻¹**（35.55 正好是 **Go1** 的膝值，疑为沿用未改）。unitree_mujoco 的 ctrlrange 用的是 45.43 → 暂以 45.43/15.70 为准，并用真机小角度大扭矩测试确认 |
| 大/小腿长度、髋外偏 | 0.213 / 0.213 / 0.0955 | 一致 → **腿部 FK 可直接复用** |

**改动清单**（M0 的活）：
1. `<motor ctrlrange=...>` → `<position kp="20" kv="1" forcerange="...">`（forcerange 取原 ctrlrange）。
2. 补训练环境要用的传感器（**已核实 Go2 MJCF 的缺口**）：
   - Go2 只有足端 **geom**（`FR/FL/RR/RL`，位于 calf body 内，`pos="-0.002 0 -0.213"`），
     **没有足端 site**；而训练环境要 `site_xpos`（足端位置、`feet_to_hip`、`feet_impact`）→
     必须在每条 calf 里补 `<site name="FR" pos="-0.002 0 -0.213" type="sphere" size="0.022"/>`（FL/RR/RL 同理）。
   - sensor 段照 Go1 补齐：`framelinvel(objname=imu) name=global_linvel`、`framezaxis→upvector`、
     4×`framelinvel(objname=<足>) name=<F>_global_linvel`、
     4×`contact(geom1=<足>, geom2="floor", reduce="mindist", data="found") name=<F>_floor_found`。
   - 已有可复用：`imu_quat / imu_gyro / imu_acc / frame_vel`（改名或加别名即可）。
3. 足端几何命名已核对：Go2 的足端 geom 就叫 `FR/FL/RR/RL`，与 `FEET` 顺序兼容（补 site 时沿用同名）。
4. 地形场景 `scene_go2_terrain.xml`：floor 换 hfield（沿用 `sim/gen_terrain.py` 产物），
   约束同 Go1（geom 名必须仍是 `floor`、priority=0）。
5. 软接触覆写（`solimp 0.9/0.95, condim=3`）——训练沿用 Go1 的做法；**部署侧保持 XML 原样**（DG-07）。

### 2.2 动作空间（★ 这是 Go1 最大的技术债）

Go1 现状：`target = home + 0.5·clip(a, ±8)` → 目标可以 ±4 rad，而髋行程只有 1.73 rad。
★ 实测：89% 的 (帧,关节) 落在机械限位上、58% 的目标偏离 home 超过整段行程、原始输出中位 12.7。

**Go2 方案（推荐 A，保守 B）**：

    A) 归一化 + 有界：  u = tanh(a);  target_j = mid_j + half_j · u_j
       mid_j = (lo_j+hi_j)/2, half_j = (hi_j-lo_j)/2      # 逐关节，前/后大腿分开
    B) 锚定 + 小 clip： target_j = home_j + half_j · clip(a_j, ±1)

- A 的好处：目标**按构造**落在关节行程内（不再需要 ctrlrange 夹取）、12 个通道量纲一致、
  探索噪声在归一化空间里均匀、网络输出有界（不会漂到 ±30）。
- 训练侧**不要**再保留低通滤波（Go1 起身用的 `a_f=0.8a+0.2a_prev`）——
  那会让部署必须复刻同一个滤波器；真实的滞后用 §3.A 的延迟/一阶滞后建模更准。
- **开源先例**：rl_sar 的 Go2 配置已经在用**逐关节**动作尺度（hip 0.125，thigh/calf 0.25），
  且动作是 tanh 有界的 [−1,1] —— 方向上和我们 §2.2 一致（只是它的 thigh 尺度仍偏小）。
- 代价：不能 warm-start Go1 策略（本来也要重训）。

> ⚠️ **别直接抄 get-up-isaaclab 的 `desired_clip_actions=3.0`**：Go1 上实测 C=3 只有 11.3% 成功、
> C=8 才 37.1%，原因是 **Go1 前大腿行程 5.19 rad（要够到 4.501 的 kip）**。
> **Go2 前大腿行程正好也是 5.19 rad**（[-1.5708, 3.4907]），所以同一个坑会原样再踩一次。
> 用 §2.2 的归一化写法就从根上避免了（目标按构造覆盖整段行程）。

### 2.3 观测（建议保持 91 维）

    [0:3]   机体系线速度（部署来自 LegOdometry，世界系→按 IMU 姿态转机体系）
    [3:6]   机体系角速度              [6:9]  projected_gravity
    [9:12]  指令（机体系）            [12:24] 关节角 − **Go2 训练 home**
    [24:36] 关节速度                  [36:48] 上一步动作（原始，未 clip）
    [48:83] height_scan 35            [83:87] cos(phase)   [87:91] sin(phase)

- 保持 91 维 → **部署框架零改动**（`obs.py` 只把 `home_q`/`scan_base` 换成 Go2 的值）。
- ⚠️ **分辨率是个被忽略的变量**：legged_gym/unitree_rl_gym 的 Go2 用 **17×11 = 187 点**（1.6 m × 1.0 m，
  ±0.1 m 均匀噪声），我们现在只有 **7×5 = 35 点**。部署侧 `HeightMapper` 本来就是 5 cm 地图，
  采样点数是纯配置 —— 若 M2/M3 发现台阶/沟壑上信息不够，**加密扫描网格只改 obs 维度**，比加记忆更便宜。
- 可选（+35×K）：scan 的 K 帧历史，用来扛近场盲区（★ 平地 9/35 可见，台阶背面更差）。
  这是对部署框架的**小幅改动**（`obs.py` 多拼 K-1 帧，`WALK_OBS_DIM` 变大），
  但换来的是"看不见的格子靠记忆补"，是 Miki 2022 那类工作验证过的路线。
- 建议**不要**把高度图当唯一感知：保留"盲走 + 高度图辅助"的降级路径（部署 health 层已有 L1 通道）。

### 2.4 判据（训练=评估=部署，一份代码）

走路：`is_healthy` 用**相对躯干下方地形**的 z 区间 + 姿态阈值（Go1 的 17.5° 经验值可沿用）。
起身：`up_z > 0.95`（世界竖直）且 `z_rel ∈ [0.20, 0.34]`，**容错计数**（满足 +1 / 不满足 −1）≥ 25 步。
→ 抽成 `envs/criteria.py`，env / `sim/eval_*` / `sim/view_go1.py` / `probe_handover.py` 全 import 同一份。

### 2.5 控制率与训练规模

`ctrl_dt=0.02`（50 Hz）、`sim_dt=0.002`、`episode_length=750`（走路）/ 400（起身）；
走路 8192 env、起身 768–2048 env（★ Go1 实测 768 env 22–25k fps，8 GB 卡有余量，
但 PPO 的样本多样性可以再大）；`njmax` ≥ **1024**（Go1 起身实测倒地单 world 需要 ~1300 约束）。

---

## 3. Go2 训练侧要新建的东西（按优先级）

### A. 时序非理想（最容易漏、收益最直接）

| 项 | 建模方式 | 参考量级 |
|---|---|---|
| 动作延迟 | 环形缓冲：每 episode 采 `d ~ {0,1,2}` 拍（0–40 ms），小概率 3–4 拍 | 部署实测 p95 抖动 4.7 ms、历史 bug 20 ms → 50 Hz 档建议 0–2 拍（0–40 ms），小概率 3–4。**开源参考**：DreamWaQ_Lab Go2 `min_delay=0/max_delay=4`（物理步）、**get-up-isaaclab（Go2）仿真最大延时 2 物理步 = 10 ms @200 Hz**、walk-these-ways `lag_timesteps=6`、IsaacLab `DelayedPDActuatorCfg.min_delay/max_delay`（单位=物理步） |
| 观测延迟 | obs 取 `t−k` 的状态。**MMDR 实测**：纯本体推理 4 ms、本体+视觉 40 ms → 显式随机化 proprio latency **[0, 0.04] s**（我们策略前向 <1 ms + 传输，取 [0, 40] ms） | 部署 p95 抖动 4.7 ms |
| 控制周期抖动 | `dt × U(0.95,1.05)` 或跳过（部署侧是 Pacer 严格 50 Hz） | 部署 RTF=1.00 |
| 执行器一阶滞后 | `tau ← tau + (tau_cmd − tau)·dt/τ_lag`，`τ_lag ≈ 10–20 ms` | 需真机辨识（开放问题 Q3） |
| 力矩-转速包络 | `tau_max(v) = saturation_effort·(1 − v/velocity_limit)`，再夹 `effort_limit` | ⚠️ IsaacLab issue #7479 实测：**共享的单值配置忽略了膝的近 1.917 减速比** → calf 速度 ×2、力矩 ×0.52（这就是为什么我们的包络要逐关节给）。**已核实**（unitree_ros `go2_description.urdf`）：髋/大腿 23.7 N·m / 30.1 rad·s⁻¹；**小腿 45.43 N·m / 15.70 rad·s⁻¹** —— 小腿速度上限只有髋的一半，包络必须**逐关节**给。注意 IsaacLab 的 Go2 资产用单一值 `effort_limit=23.5 / saturation_effort=23.5 / velocity_limit=30.0`，**低估小腿力矩、高估小腿转速**，别照抄 |
| PD 增益误差 | `kp,kd × U(0.8,1.2)`（逐关节） | 部署 kp=20/kd=1。开源范围：walk-these-ways Kp ×[0.8,1.3] / Kd ×[0.5,1.5]；MMDR Kp [40,90] / Kd [0.4,0.8]；DreamWaQ_Lab Go2 (0.9,1.1) |
| 电池压降 | `sat_effort × U(0.9,1.0)` | 真机电压随电量变化 |

### B. 物理域随机化（照抄成熟配方 + 本框架的量）

- playground `go1/randomize.py`（本机已装）：地面摩擦 U(0.4,1.0)、`dof_frictionloss` ×U(0.9,1.1)、
  `armature` ×U(1.0,1.05)、躯干 CoM ±0.05 m、连杆质量 ×U(0.9,1.1)、躯干质量 ±1.0 kg、`qpos0` ±0.05 rad。
- legged_gym：`friction_range=[0.5,1.25]`、`added_mass_range=[-1,1] kg`、push 每 15 s ≤1.0 m/s；观测噪声尺度（×scale）：dof_pos 0.01 / dof_vel 1.5 / lin_vel 0.1 / ang_vel 0.2 / gravity 0.05 / height 0.1。
- **walk-these-ways（Go1 真机）**：附加质量 [−1, 3] kg、地面摩擦 [0.05, 4.5]、restitution [0,1]、CoM [−0.1,0.1] m、电机强度 ×[0.9,1.1]、Kp ×[0.8,1.3]、Kd ×[0.5,1.5]；`randomize_lag_timesteps=True`。
- **MMDR**：质量 ×[0.8,1.2]、惯量 ×[0.5,1.5]、电机摩擦 [0,0.05] N·m·s/rad、电机强度 ×[0.8,1.2]、地面摩擦 [0.5,1.25]、Kp [40,90]、Kd [0.4,0.8]。
- **DreamWaQ_Lab（Go2）**：电机强度 (0.9,1.1)、刚度/阻尼因子 (0.9,1.1)、push x±0.3 / y±0.1 m/s。
> 范围取舍：Go2 真机比 A1/Go1 重、电机更热，建议**先用 walk-these-ways/legged_gym 的保守档**（摩擦 [0.5,1.25]、质量 ±10%、Kp ×[0.8,1.2]），再按真机表现放宽。

**决策 #3（无真机辨识）的补救：DR 要覆盖"参数不确定"本身。** 本机只读实测给了一个有用的界：
关节 damping 取 0.1 / 1.0 / 2.0 时，开源 home 下的**静态站姿几乎不变**（z = 0.268 / 0.273 / 0.274 m，倾角 1.58° / 1.57° / 1.60°，力矩不变）。
→ 那份"两份 Go2 模型 damping 0.1 vs 2"的矛盾**对站姿影响很小**，但它会影响**接触瞬间与步态**（阻尼是接触振荡的耗散通道）。
所以：damping 用 U(0.1, 2.0) 这种宽区间随机化，膝力矩用 45.43 但把 `effort` 也随机化到 [0.8, 1.0]× 覆盖 35.55 的可能 —— 用不确定性换鲁棒性，而不是去猜哪个对。
- 本框架已有：`push_config`（每 250 步 ±0.5 m/s）、`obs_noise` —— Go2 上把 push 提到
  legged_gym 量级并**同时用**于评估（Go1 的鲁棒性评估一直是关的）。
- 足端摩擦（Go2 0.4）单独随机化 —— 和地面摩擦是两个独立参数，别只随机一个。

### C. 感知：训练侧虚拟雷达（★ 本方案的核心新增）

**目标**：训练时 actor 看到的 35 维**不能**来自 `terrain_height(xy)` 真值，而必须来自
"和部署同一条管线"的点云 → 高程图。部署侧的实现是 `src/go2_bridge/perception.py`，
训练侧要复刻它的语义：

**✅ 已确认的雷达：Unitree 4D LiDAR L1（Go2 标配，官方参数已核实）**

| L1 参数 | 值 | 对训练侧的含义 |
|---|---|---|
| FOV | **360° × 90° 半球**（非重复扫描） | ⚠️ **已纠正**：手册说"可对雷达上方 360° 的三维空间进行测量"指的是**传感器自身 +Z 半球**；Go2 上是**倒装**（URDF `radar_joint` rpy=(0, **2.8782 rad = 164.91°**, 0)）→ 传感器 +Z 在机体系 = (0.2604, 0, **−0.9655**)，即**指向地面、前偏 15.1°** → 地面可见（正下方完全可见）。框架的 `vfov=[-0.5,0.2]`（−28.6°~+11.5°）**方向是对的但太窄**（L1 实际覆盖 ≈[−90°, +15°]），我们的 35 格落在 −87°~−20° → 近场几格在仿真里根本打不到（这就是"平地只 9/35 可见"的主因） |
| 周向扫描 | **11 Hz** | 地图更新 ≈ 11 Hz，**与 50 Hz 策略解耦**（年龄 0–90 ms 随机化） |
| 竖直扫描 | 180 Hz | 单帧内竖向扫 180/11 ≈ 16 次 |
| 采样 | 43200 点/s，**有效 21600 点/s** | **≈1964 点/帧** —— 框架默认 `n_rays=2000` 正好等价于一帧，不用改 |
| 精度 | **±2.0 cm** | 测距噪声按 ±2 cm（或 σ≈1 cm）注入，和框架现在的 0.02 m 一致 |
| 近处盲区 | **0.05 m** | 倒装后雷达离地约 **0.221 m**（= 站高 0.268 − 0.0468）→ 正下方地面 0.221 m ≫ 0.05 m，**脚下近场可见** |
| 量程 | 30 m@90% / 15 m@10% | 局部高程图用 8 m 就够（与 elevation_mapping_cupy 同）|

| 部署语义 | 训练侧对应 |
|---|---|
| `SimLidar`（MuJoCo ray-cast 点云，世界系） | `VirtualLidar`：对 hfield 做射线求交（JAX） |
| `HeightMapper`：5 cm 世界系格网、每格取 max z、`decay_s=1.0` 超时、未知格在 4 格内找最近新鲜格 hold-fill、否则 0.0 | 同参数量、同填充策略（用**局部固定大小网格**实现，语义等价、JAX 友好） |
| 缺失格的双线性按可用权重重归一 | 同上 |
| `scan()`：yaw-only 旋转、`clip(z_base − h − base, ±0.35)×5` | 与训练 `height_scan_features` 逐点相同 |

> ✅ **本机已核对**（mujoco 3.12.0 / mjx）：`mujoco.mjx._src.ray._RAY_FUNC` 的分发表只有
> box/capsule/ellipsoid/mesh/plane/quad/sphere/triangle —— **没有 HFIELD**，所以训练侧**不能**直接用
> `mjx.ray` 打地形，必须自己写 hfield 求交（下面两种）。CPU 版 `mj_rayHfield` 是有的，
> 但那是单环境 CPU 路径，不能进 MJX 批量训练循环。上游 issue #2155 已于 2026-06-15 关闭，
> 但**升级 MuJoCo 之后要重新核对**这张表。

**✅ 雷达外参（官方 `unitree_ros/go2_description.urdf`，训练侧虚拟雷达直接用）**：

    radar_joint: parent=base, xyz=(0.28945, 0, -0.046825), rpy=(0, 2.8782, 0)   # 164.91° pitch
    ⇒ 传感器 +Z（FOV 半球轴）在机体系 = (0.2604, 0, -0.9655) → 指向地面、前偏 15.1°
    ⇒ 覆盖机体系仰角 ≈ [-90°, +15°]：正下方可见；前方到地平线以上 15°；
       后方浅仰角(> -15°)看不到 → 雷达后方 0.82 m 之外的地面略过
    ⇒ 35 格（-87°…-20°）**全部几何可见**，唯一剩下的未知是**躯干/大腿的自遮挡**

**✅ 自遮挡实测（`sim/probe_go2_lidar_occlusion.py`，纯 CPU MuJoCo 射线投射，本机已跑）**：

| 姿态 | 35 格可见 | 遮挡来源 |
|---|---|---|
| 名义四足站姿 | **30/35 = 85.7%** | FR_calf×2, RR_calf, RL_calf, FL_thigh |
| 三足（FL 摆动相） | **34/35 = 97.1%** | FL_thigh×1 |
| 躯干前倾 +5° | 29/35 = 82.9% | FR_calf×2, RR_calf, RL_calf, FL_calf×2 |
| 躯干后仰 −5° | 31/35 = 88.6% | FR_thigh, RR_calf, RL_calf, FL_thigh |
| **四姿态始终可见** | **28/35** | |
| **至少一种姿态可见** | **35/35** | |

半球采样一帧（4000 条射线，真机 ≈1964 点/帧 @11 Hz）：**地面 3534 条(88.4%)**、无回波 287、打到腿 123；
**35 格每帧都至少 1 点**，平均 75 点/格/帧（最密 457、最疏 7）。

→ **结论：遮挡不是瓶颈。** 瞬时可见 85.7%、跨姿态 100%、密度远高于 5 cm 地图所需（每格 ≥1 点即可）。
唯一系统性盲区是 **x = −0.3 那一排（机器人正后方 4/5 格）**，由后腿小腿遮挡，且随步态相位变化 →
框架 `HeightMapper` 的 hold-fill（1 s 衰减 / 4 格半径）正好覆盖；想彻底干净可把网格前移一格（x∈[−0.2,1.0]）。
训练侧虚拟雷达**可以先不模拟腿部遮挡**（JAX 里的 hfield 求交看不到机器人自己的腿），
因为它比噪声/时延/地图漂移次要一个数量级。
顺带修正：框架"平地 9/35 可见"的主因是 `vfov` 太窄 + 单帧稀疏，**不是**遮挡。

**训练侧必须复刻的两个"非理想"设计（来自 Miki 2022，这是本节的骨架）**：
1. **按 episode 采三档地图质量**：clean 60% / noisy 30% / **完全无地形信息 10%**（模拟遮挡/建图失败）。
   噪声作用域分三层：per-scan-point（逐点平移+高度扰动）/ per-foot / per-episode；
   地图更新率与策略解耦（**地图 20 Hz、策略 50 Hz**）。
2. **网络侧要有"退回本体感知"的通路**：Miki 用 attention gate + 地形重建 loss；
   PIE/DreamWaQ 用 GRU/VAE 记忆隐式补全。对应我们的方案：scan 历史帧 + 保留纯本体基线，
   并在健康层保留"高程图失效 → L1/降级"的路径。

**射线求交的两种实现**（按算力取舍）：
1. **极坐标累积最大（推荐，物理正确且便宜）**：在传感器极坐标网格 (R×A，例如 32 距离 × 64 方位)
   上采样地形高度 → 计算仰角 `atan2(h−h_s, r)` → 对每个方位沿距离做 `maximum.accumulate` →
   仰角 ≥ 前向累积最大值的格子才"可见"。一次 O(R·A) ≈ 2048 次地形采样/env/步，
   天然给出遮挡、FoV、量程截断。相对现有 35 点扫描约 30×，占单步开销可接受（先实测）。
2. **逐查询点可见性（更便宜的近似）**：只对 35 个采样点，从传感器沿 2D 直线步进 16 步判断是否被挡，
   35×16 = 560 次采样（≈8×）。先上这个也能把"台阶背面看不见"这个主要效应学到。

**必须注入的退化**（对齐 `SignalDegrader` 的口径；带 ★ 的是开源实测值，可直接照抄）：
> **RENet（Go1，raycaster 随机化）★**：深度 cap **2 m**；**+1% 距离噪声 + 高斯方差 0.01**；
> 传感器起点 ±**1 cm** 位置、±**5°** pitch/yaw；控制 dt 0.02 s；报告"噪声到 **30%** 就开始抖"。
> **真实高程图管线 ★**（elevation_mapping_cupy，ANYmal/Jetson Xavier）：resolution **0.04 m**、
> map_length **8 m**、点噪声 **σ=0.05·z²**、mahalanobis 2.0、outlier_variance 0.01、max_drift 0.1、
> update_variance/pose/acquire = **5/10/5 fps**，整条管线 **6.857 ms**（filtered 49.4 Hz / raw 16.1 Hz / Bpearl 20 Hz）。
> **TAMOLS ★**：点云→地图 **20 Hz**、WBC 400 Hz；空单元用**遮挡边界的 min 值 in-painting** + 重复中值 + 高斯平滑。
> **nvblox ★**：voxel 0.05 m。→ 我们的 `HeightMapper` 用 5 cm 格 + hold-fill，量级与这些一致，方向对。

- 量程/角度量化（真实角分辨率，例如 0.2°–1°）；
- 测距噪声 σ 1–3 cm、随距离增长；
- 丢点：`p ~ U(0, 5%)`，近场/掠射角更大（部署实测：平地上 35 格只有 9 格可见）；
- 自遮挡 / 无回波（掠射、镜面、太近）→ 该格进入"未知"；
- **时延**：点云比状态晚 1 拍（真实管线是"扫 → 建图 → 查询"）；
  ⚠️ **雷达更新率远低于控制率**：go2_ros2_sdk 实测 `point_cloud2` 只有 **~7 Hz**（高层 SDK 里 joint_states 更是只有 ~1 Hz，
  所以它不能用来跑 RL）。直接走 DDS 拿原始雷达话题也要按 10 Hz 量级设计：
  **高程图的更新时间要和控制频率解耦**，训练侧按 10 Hz 更新地图 + 地图年龄 0–140 ms 随机化，
  而不是每步都重建地图；
- 安装误差：高度 ±5 cm、俯仰 ±5°、yaw 标定误差 ±3°（真机外参标定误差是常见的失稳源）。

**评估侧**：新增一个"感知档位"矩阵（nominal / leg_fk / lidar_clean / lidar_noisy / 全盲），
用部署框架的 `demo_ros2_policy.sh <时长> go2 <profile> <scan_source>` 直接跑 —— 这套接口已经有了。

**成本/风险**：训练步时会涨（先按实现 1 实测，涨超过 50% 就退回实现 2）；
高程图会显著改变 obs 分布 → 必须**从头训**，不能从 Go1/无感知版本热启动。

### D. 奖励与终止（在 Go1 配方上做三处手术）

Go1 v23 走路奖励分项实测（★ 健康帧每步）：linear_vel 1.82 / feet_phase 1.64 / angular 0.93 /
feet_impact −0.26 / default_joint −0.15 / feet_slip −0.15 / ascent **+0.035**。

1. **ascent 改势能差**：`r = γ·W·h(s') − W·h(s)`（γ=0.99）。站着不动净收益恒为 0，
   所以 W 可以放大 10–30 倍；这是"上台阶 0%"最直接的修法（现在它只占奖励的 1%）。
2. **base_height 目标用 Go2 实测站高**（DG-15：不许靠调 base 去"对齐"真机，两边同值）。
3. **上台阶**：地形课程（0.18 m → 0.30 m，两次 restore）＋**出生分布**（50% 撒在楼梯口 ±0.5 m），
   比继续调权重有效；前视范围 x_max 0.9 → 1.2–1.5（进深 0.55 m 的台阶需要预判）。
   注意改 obs 维度必须重训。
4. 起身：动作空间归一化（§2.2）、判据对齐部署（§2.4）、成功奖金改成"站住 **且安静**
   （关节速度/动作幅度低于阈值）"、摔倒姿态池来自 **Go2 走路策略**的真实摔倒
   （Go1 训推分布差 11 个点就是这一步缺的）。
5. **站高常数要重标**：get-up-isaaclab 的 Go2 用 **0.30 m**，我们 Go1 用 0.28；Go2 官方 home 是
   `0,0.9,-1.8`（= 我们框架 keyframe）而 get-up-isaaclab 用 MuJoCo home 关键帧 + 0.30。
   必须实测 Go2 在**选定 home** 下的沉降站高，然后 base / 高度奖励 / 成功判据三处同值。
6. **起身和走路的 PD 增益可能不同**：get-up-isaaclab 部署侧走路 Kp=20/Kd=2.0、起身 Kp=25/Kd=2.0
   （切换时同时改增益）。我们框架现在只有一组 `policy.kp/kd` → 见 §5 第 7 条。

### E. 课程与初始分布

- 地形：平地 → 起伏+台地 → 加楼梯（`gen_terrain.py --stair_steps 3` → 5）；
  多阶段用 `--restore` 接（Go1 已验证），或按 legged_gym 的"走够远就升级"。
- 出生：xy 范围留边界余量（★ 出界虚空占 10.4%）；加"楼梯口出生"的混合分布。
- 起身：`joint_alpha` 现在是"每次运行一个固定值"，建议改成**由全局步数驱动的单次课程**
  （reset 时按计数器插值），省掉两次手动串接。

---

## 4. 里程碑与验收（每步都有可执行的命令）

| M | 内容 | 验收（必须全绿才进下一步） |
|---|---|---|
| **M0** | Go2 模型 + 环境移植 | ① `go2_position.xml` 能被 MuJoCo 加载，与框架 `models/go2/go2.xml` 做力矩/位置等价测试（Go1 的 `tests/test_go1_model_equivalence.py` 照搬）；② 在框架里 PD 站立：tilt < 1°，z ≈ 0.27；③ `sim/audit_action_space.py --task walk/getup` 在 Go2 上跑通 |
| **M1** | 走路基线（平地、本体感知） | `sim/eval_walk.py` 在平地固定出生集：存活 > 95%、速度跟踪误差 < 0.05 m/s；策略导出 npz 后框架里走 ≥3 m/12 s |
| **M2** | 地形 + 域随机化 + 延迟 | 地形存活率 ≥ Go1 v23 同口径；开 push/obs_noise/延迟的评估不掉 > 5 个点；上台阶 ≥ 30%（Go1 是 0%） |
| **M3** | 感知（虚拟雷达） | 训练侧 `scan` 与部署 `perception.py` 在同一 hfield 上逐点一致（±1e-3，含未知格填充）；`lidar_noisy` 档不掉 > 10 个点；全盲档能站不摔 |
| **M4** | 起身 | 部署判据下成功率 ≥ 85%（训练分布）/ ≥ 75%（走路真摔姿态池）；终态关节速度 < 1 rad/s（★ Go1 是 4.03）；交还后 500 帧摔倒率 ≤ 10% |
| **M5** | sim-to-sim 回归 | `demo_ros2_policy.sh 12 go2 <profile> lidar`：与训练环境同初始条件的轨迹一致（前 2 s 关节角 max diff < 0.05 rad）；健康层全程 L0 |
| **M6** | 真机清单 | 先吊起 → `--controller zero` 收状态 → 补 `rt/lowcmd` CRC32 → 关运动模式 → 低速小步 |

---

## 5. 部署框架需要的小改动（都在"为训练小幅改变"的范围内）

1. `config/go2_sim.json`：`policy.npz` 换 Go2；`policy.train_robot: "go2"`；
   `perception` 段换成真机雷达的实际参数；`domain_gap.profile` 增加与训练 DR 对应的档位。
2. `src/go2_bridge/obs.py`：`home_q` 默认 = Go2 训练 home；`scan_base` = 实测 Go2 站高；
   若采用 scan 历史，则 `WALK_OBS_DIM` 与拼接顺序同步（改动局限在这一个文件 + 契约文档）。
3. `config/domain_gap.json`：DG-01/03/12 重标为"由 Go2 重训关闭"；新增
   DG-18（雷达 FoV/丢点/遮挡）、DG-19（执行器包络/滞后）、DG-20（策略延迟，训练侧 DR 已覆盖）。
4. `docs/interfaces.md §1`：把"训练侧事实"更新为 Go2 契约（这是框架的"唯一真相"，必须同步）。
5. `models/policies/`：导出脚本泛化成 `export_policy.py`，并把
   **obs 维度 / home / scan_base / kp / kd / 动作 clip / 训练命令** 一并写进 json ——
   Go1 那次"eval 用默认 action_clip 评出假 0%"的坑就是元数据没跟着策略走。
6. `src/go2_bridge/controllers.py`：策略档的 `action_clip_to_train_joint_range` 在 Go2 下
   要按 **Go2** 的范围夹（不再是 Go1 的）。
7. **PD 增益做成"每个策略一份"**：get-up-isaaclab 走路用 Kp=20/Kd=2.0、起身用 Kp=25/Kd=2.0。
   我们现在只有一组 `policy.kp/kd` —— 如果 Go2 上验证出"起身需要更大 Kp"，
   就要把增益从 `config.go2_sim.json:policy` 挪到每个策略的元数据里（导出 npz 时一起落盘）。
8. `src/go2_bridge/perception.py` 不需要改逻辑，但要把 `SimLidar` 的参数与**训练侧虚拟雷达**对齐
   （同一份 FoV/量程/角分辨率/丢点率），否则训练与部署的感知分布又对不上。
9. **`config/go2_sim.json:perception` 必须开工**：现在 `enable: false`、`vfov=[-0.5,0.2]`、height 0.30 —— 按决策 #1/#4，
   要改成 L1 的 360°×90° 视场、`enable: true`，并把 `height_scan_source` 从 `leg_fk`/`nominal` 切到雷达；
   同时补上 policy-port.md 里记的"真机 PointCloud2 → HeightMapper 订阅还没接"。
10. 按决策 #5 扩大指令范围后，`config/go2_sim.json` 与 `docs/interfaces.md §1` 里的 `command=(0.5,0,0)` 也要同步。

---

## 6. 开源资料清单（已核实链接）

### A. Go2 资产 / 训练 / 部署
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) — 官方 Go2 训练+部署。
  已核实的 Go2 配置：`stiffness=20 / damping=0.5 / action_scale=0.25 / decimation=4`，
  出生 z=0.42，默认关节角 = [hip ±0.1, 大腿 0.8(前)/1.0(后), 膝 −1.5]（**不是** unitree_mujoco keyframe 的 [0,0.9,−1.8]，见 §2.1 的候选 ②），`base_height_target=0.25`，
  DR：摩擦 [0.5,1.25]、加质量 [−1,1] kg、每 15 s 推 ≤1.0 m/s。
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) — 官方 MuJoCo + LowCmd/LowState 桥；
  **本框架的 Go2 MJCF 就来自这里**，训练侧应直接沿用同一份资产。
- [rl_sar](https://github.com/fan-ziqi/rl_sar) — MuJoCo + ROS 2 + libtorch/onnx，Go2 已支持 sim 与 real；
  与本框架定位最接近，值得对照它的观测组装与部署循环。
- [robot_lab](https://github.com/fan-ziqi/robot_lab) — IsaacLab 版 Go2 配置（rl_sar 的上游训练）。
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — 第二份 Go2 MJCF（交叉校验用）。
- [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) — Go1 joystick/getup 与
  `domain_randomize` 参考实现。**已核实：main 分支没有 Go2 环境**（只有 go1/g1/h1/barkour/spot/…），
  所以 Go2 的 env 得自己写。

### B. 时序 / 执行器 / 域随机化
- [IsaacLab actuators](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html) —
  `DelayedPDActuatorCfg`（`min_delay`/`max_delay`，单位=物理步）、`DCMotor`（四象限力矩-转速曲线，
  `saturation_effort`/`velocity_limit`/`effort_limit`）、`ActuatorNetMLP`（学出来的执行器模型）。
- Hwangbo et al., *Learning agile and dynamic motor skills for legged robots* (Science Robotics 2019) —
  ActuatorNet + 显式建模延迟的原始工作。
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (arXiv:2107.04034) — 本体感知自适应模块 + DR 清单。
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — 摩擦/推力/噪声的行业默认量级（含 obs 噪声尺度表）。
- [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) — **Go1 真机**、显式动作滞后随机化（`lag_timesteps=6`）+ 全套 DR + obs history/adaptation module，是与我们最同构的开源实现。
- [DreamWaQ_Lab](https://github.com/Qinzx03/DreamWaQ_Lab) — **Go2** + IsaacLab：`min_delay=0/max_delay=4`、DR、`deploy.yaml` 导出（可直接对照 §5 的元数据落盘做法）。
- MMDR / Imai et al. (arXiv:2109.14549) — 唯一显式给出**观测延时随机化区间**（[0,0.04] s）与推理耗时实测（4 ms / 40 ms）的工作。
- [Hwangbo et al. 全文（ar5iv）](https://ar5iv.labs.arxiv.org/html/1901.08652) — ActuatorNet：MLP 3×32 softsign、400 Hz / >100 万样本、平均误差 0.740 N·m（理想模型 5.74 N·m）。
- 本机参考实现：`mujoco_playground/_src/locomotion/go1/randomize.py`（摩擦/质量/CoM/qpos0 的 JAX 写法）。

### C. 点云 / 高程图 / 遮挡

| 项目 | 关键可抄的东西 |
|---|---|
| [Miki 2022, Science Robotics](https://ar5iv.labs.arxiv.org/html/2201.08117) / [复现页](https://perceptive-loco.github.io/) | 噪声高程图 + teacher-student 的奠基工作：地图 20 Hz / 策略 50 Hz；每只脚周围 5 个半径采样；噪声三层作用域（per-point / per-foot / per-episode）；**地图质量按 clean/noisy/盲 60%/30%/10% 采样**；地形重建 loss + attention gate 决定外感放行量 |
| [elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) | 真机高程图主流管线：0.04 m / 8 m / σ=0.05·z² / mahalanobis 2.0 / max_drift 0.1 / 5·10·5 fps / Jetson 上整条 6.857 ms |
| [TAMOLS](https://ar5iv.labs.arxiv.org/html/2206.14049) | 缺失/遮挡高程图的成熟滤波链：用**遮挡边界的 min 值 in-painting** + 重复中值 + 高斯平滑（可直接抄成 `HeightMapper` 的 fill 策略） |
| [grid_map](https://github.com/ANYbotics/grid_map) / [nvblox](https://github.com/nvidia-isaac/nvblox) | 多层 2.5D 容器（圆缓冲重定位）/ GPU TSDF-ESDF（voxel 0.05 m）——真机建图后端备选 |
| [Isaac Lab ray_caster](https://isaac-sim.github.io/IsaacLab/main/source/core-concepts/sensors/ray_caster.html) | 训练侧射线模拟的参数模板（LidarPatternCfg、update_period、`ray_cast_drift_range`）。⚠️ **内置只有 drift，没有逐 ray 高斯噪声/丢点，要自己加** |
| [RENet](https://ar5iv.labs.arxiv.org/html/2509.09283)（Go1） | 训练/部署不一致的显式修法：cap 2 m、+1% 距离噪声、起点 ±1 cm、±5° 姿态扰动；"噪声 30% 就开始抖" |
| [PIE](https://ar5iv.labs.arxiv.org/html/2408.13740) / [DreamWaQ](https://arxiv.org/abs/2301.10602) / [Extreme Parkour](https://ar5iv.labs.arxiv.org/html/2309.14341) | 深度 10 Hz / 动作 50 Hz 的双频设计；GRU/VAE 记忆隐式补全；"直接喂含噪高程图不如端到端蒸馏"的对照结论 |
| [MuJoCo issue #2155](https://github.com/google-deepmind/mujoco/issues/2155) | MJX 的 `mjx.ray` 缺 hfield。**本机已核对：3.12.0 的分发表里没有 HFIELD** → 必须自己写求交；CPU 版 `mj_rayHfield` 不能进批量训练 |
| **本仓库 `src/go2_bridge/perception.py`** | 训练侧要对齐的参考实现（5 cm 格、decay 1 s、hold-fill 4 格）——比论文更直接 |

### D. 起身 / 摔倒恢复（已核实的开源配方）

| 项目 | 关键可抄的东西 |
|---|---|
| [get-up-isaaclab](https://github.com/iit-DLSLab/get-up-isaaclab)（Go2，含真机 ROS2） | RL 50 Hz（sim dt=1/200 × decimation 4）；部署走路 Kp=20/Kd=2.0、起身 Kp=25/Kd=2.0，仿真执行器 Kp=20/Kd=1.5；力矩 hip/thigh 23.7(30.1 rad/s) / calf 45.43(15.7 rad/s)；**仿真最大延时 2 物理步 = 10 ms @200 Hz**；动作 scale 0.5 / clip ±3.0 / 一阶 α=0.8；站高目标 **0.30 m**；DR：摩擦 0.2–1.25、恢复系数 0–0.1、基座质量 ±5 kg、连杆 ×0.9–1.1、CoM ±0.02 m、外力 ±5 N/Nm、关节摩擦/armature ×0.8–1.2、Kp/Kd ×0.8–1.2、10–15 s 推 ±0.5、动作噪声 σ0.05+bias0.015、观测噪声 σ0.02+bias0.001；4096 env / 6 s episode / lr 1e-3 / γ0.99 / λ0.95 / MLP[128,128,128] |
| [HoST](https://arxiv.org/html/2502.08378) | 多 critic + **力课程**（200 N 起，每成功 −20 N，下限 0）+ **action bound 课程**（1.0 起，每步 −0.02，下限 0.25）；成功 = 站高过阈值并保持到 episode 末 |
| [HumanUP](https://arxiv.org/html/2502.12152v1) | 两阶段：Stage I 用 1 s 动作"先发现"，再把动作插值减速到 8 s 交 Stage II 做模仿 |
| [ANYmal Recovery](https://arxiv.org/html/1901.07517) | 四足分层起身鼻祖（self-righting / stand / selector 多网络）；从 0.5 m 落下 + 随机关节角训练；>100 次试验成功率 >97%，任意姿态 <5 s |
| [AFR](https://ar5iv.labs.arxiv.org/html/2412.16924) | 跨本体跨地形起身：height-map encoder + privileged critic；障碍 0.05–0.3 m、台阶高 0.05–0.25 m |
| [fallrec-rl](https://anubhav1772.github.io/fallrec-rl/) | Go1 阶段化起身 + **handoff 测试**（明确要求终态是 locomotion-ready stand） |
| [From Recovery to Locomotion](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-87.html) | 用 DAgger 把起身/行走蒸馏成一个控制器，免硬切 |
| 本机 `mujoco_playground/_src/locomotion/go1/getup.py` | Kp=35 / Kd=0.5 / action_scale=0.5 / drop_prob=0.6 |

**关于交还**：get-up-isaaclab 是**手动直接切换**（console `goUp/activate` + 同时切 Kp/Kd，**没有 blend/cooldown**），
这和我们 Go1 的实测一致（blend10 → 摔倒 38.3%，直接切 7.9%）。但公开资料里**没有**
"起身终态是发力状态、不能当行走初始目标"的明确论述 —— 那条结论目前仍是我们自己的实测，
属于**可以先沉淀为经验**的部分。

---

## 7. 已确认的输入 与 仍然开放的问题

### 7.1 已确认（2026-10-17，见 §0.1）
L1 雷达 / 机载算力盒(Jetson Orin) / 无系统辨识 / 必须依赖雷达且失效时退化为平地盲走 / 指令范围适度扩大 / home 用开源惯例。

### 7.2 仍然开放（不影响 M0，但会影响 M2–M5）
1. **L1 的竖直 90° 视场怎么分布**？官方只写"360°×90°"，没给上下边界。若是对称 ±45°，
   在 0.3 m 安装高下，**机身正下方 |xy| < 0.3 m 是一块结构性盲区**（我们 35 点网格里 x∈[−0.3,0.9]、y∈[−0.4,0.4]
   的前几格就落在里面）→ 策略必须学会"脚下没数据也能走"。这条要用真机点云实测确认（M5 的标定项）。
2. **L1 在 Go2 上的实际安装高与俯仰**（现在是框架默认 0.30 m / 水平）。它直接决定近场盲区半径 r = h/tan(θ_max)。
3. **地图更新与记忆深度**：L1 是 11 Hz 非重复扫描，单帧 ≈1964 点；我们是否加 scan 历史帧（obs +35×K）？
   还是先把地图做厚（多帧累加 + decay 1 s，现在 HeightMapper 已经是这个语义）？建议先 M3 实测再决定。
4. **机型版本**：Go2 标准版还是 Go2-W（轮足）？本节所有数字按标准 Go2。
5. **真机 rt/lowcmd 的 CRC32**（框架已知缺口）+ `PointCloud2 → HeightMapper` 的真机订阅还没写；
   按决策 #4，这两条从"可选"升级为**上线前必须补**。
