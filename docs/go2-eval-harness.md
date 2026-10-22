# Go2 评测工具: sim/eval_go2.py

> 任务: 判断 "Go2 策略能否在一定复杂地形上稳定行走"。本文件 = 工具说明书 + 自检报告 +
> 已知坑清单。所有数字都是本机 CPU MuJoCo 实测 (不占 GPU), 可复现。
> 写作时间 2026-10-17; 模型用 asset-smith 当日交付的 `models/go2/*`。

## 0. 快速开始

~~~bash
cd /home/tc63/mujoco/mjx-go2-training
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

# (1) 纯链路自检 (不需要策略): home 姿态顶住站立
python sim/eval_go2.py --policy none --dummy hold_home --terrain flat \
    --seeds 4 --duration 10 --tag selfcheck_hold

# (2) 摔倒检测自检
python sim/eval_go2.py --policy none --dummy zero --seeds 4 --duration 5

# (3) 真策略 (pkl 或 orbax checkpoint 目录) + 地形
python sim/eval_go2.py --policy policies/go2_walk_policy.pkl \
    --terrain terrain --cmd 0.5 0 0 --seeds 20 --duration 15

# (4) 交叉验证 (Go1 策略 + Go1 模型, 应当能走) —— 用 Go1 的终止阈值
python sim/eval_go2.py --policy policies/go1_walk_policy.pkl \
    --xml models/go1/scene_mjx_terrain.xml --vel-frame world --spawn random \
    --spawn-range 5 --healthy-z-min 0.22 --healthy-qxy-deg 17.5 \
    --cmd 0.5 0 0 --seeds 12 --duration 15
~~~

输出 (默认 `logs/go2_eval/`): `<tag>.json` (全部标量指标) + `<tag>.png` (4 联图)。
加 `--dump-trace` 存 npz 轨迹。

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `sim/eval_go2.py` | 主评测脚本 (CPU MuJoCo 50Hz 闭环 + 6 类指标 + PNG + JSON) |
| `logs/go2_eval/probe_settle_go2.py` | 诊断: 求解器档位 × 静置表现 (解释 `--solver` 默认值) |
| `logs/go2_eval/parity_check_sim.py` / `_parity_min.py` | 对拍探针: 我的 numpy obs vs 训练 env 本体 (见 §8 待办 2) |
| `logs/go2_eval/*.json` `*.png` | 自检结果、验证结果与曲线 (PNG 会自动注册 Noto Sans CJK, 中文标签可读) |

## 2. 观测 / 动作 / 终止: 与训练侧逐项对齐

真值来源是 `envs/go2_walk.py` (= `envs/go1_walk.py` 的 Go2 移植) 的**代码**, 不是文件头的
文档字符串 (那里有一处与代码不符, 见 §7 坑 1)。

### 2.1 actor obs (维度由策略权重第一层自动判定: 48 + 35·scan + 8·phase)

| 段 | 内容 | 训练侧 (`envs/go2_walk.py`) | `eval_go2.py` |
|---|---|---|---|
| 0:3 | 机体系线速度 ×2.0 | `_get_obs:1077-1080`, `obs_scale.linear_velocity=2.0` (:316) | 同式; `--vel-frame body/world` |
| 3:6 | 角速度 ×0.25 | `_get_obs:1081`, `:317` | 同 |
| 6:9 | 机体系重力 R^T·[0,0,-1] | `get_gravity:610-612` (`site_xmat[imu]`) | 同 (无 imu site 时退化为 base 机体系) |
| 9:12 | 指令 ×2.0 | `:1097` | 同 |
| 12:24 | `qpos[7:] - key_ctrl[0]` | `:1083-1084` | 同; home ctrl 由 keyframe `home` 读出 (`--home-pose` 可覆盖) |
| 24:36 | `qvel[6:]` **不缩放** | `:1085` | 同 (**坑 1**) |
| 36:48 | `last_action` = 上一步策略**原始**输出 | `:1100`, `:1032` | 同 |
| 48:83 | 高度扫描 35 维 (可选) | `height_scan_features:714-745` | `height_scan()` 逐点复刻 |
| 83:91 | 相位 [cos φ, sin φ] × 4 足 (可选) | `:1113-1124`, `_trot_phase0=[0,π,π,0]` (:573) | 同; 初相 = trot + U(0,2π), 每步 +2π·f·dt, f=1.8 Hz (:247) |
| — | 整体 clip(±100) | `:1135-1136`, `obs_clip=100` (:321) | 同 |

高度扫描细节 (逐字复刻): 网格 x∈[-0.3,0.9] 7 列 / y∈[-0.4,0.4] 5 行,
`meshgrid(indexing="ij")` 后 x 外层 y 内层; 只按 yaw 旋转网格 (不加 roll/pitch);
特征 = clip(z_base − h_ground − 0.268, ±0.35) × 5.0 (:354-371);
hfield 采样 col=(x/sx+1)/2·(ncol−1)、row=(y/sy+1)/2·(nrow−1) + 双线性 + 边界 clamp
(`terrain_height:670-712`)。

**独立佐证 (布局正确性的硬证据)**: `policies/go1_walk_policy.pkl` 的策略网第一层是
(91,128) → 91 = 48+35+8; 价值网 (162,128) → 162 = 91 + 71 (特权段)。两个维度与
`_get_obs` 的拼装顺序严格吻合, 反过来锁定了本表的布局假设。

### 2.2 动作

* 12 维**绝对关节位置目标**: `ctrl = clip(action, 关节位置范围)` (`step:901-907`)。
* 模型是 `<position kp=20 kv=1>` (asset-smith 交付的 `models/go2/go2_mjx_position.xml`):
  直接写 `data.ctrl`。
* 模型是 `<motor>` (例如部署仓库 `mjx-go2-mujoco2ros2/models/go2/go2.xml`, 其 ctrlrange 是
  **力矩** ±23.7/±45.43): 自动退化为显式 PD
  `tau = clip(kp*(q_des - q) - kd*qd, ctrlrange)`, `q_des = clip(a, 关节范围)`,
  默认 `--kp 20 --kd 1` —— 与部署侧 `go2_bridge/controllers.py:283-294` 的 LowCmd 语义一致。
* `--action-mode`: clip (默认, env 原语义) / soft (5% 内缩后 clip) / tanh_soft (±1 → 软范围线性映射)。

### 2.3 终止 (摔倒) 判据

训练侧 `_is_healthy:753-785` = z_rel∈(0.19,0.65) **且 |q[1]|,|q[2]| < 10°** ——
注意卡的是**四元数分量**, 不是真欧拉角 (小角下 ≈ ±20° 真倾角)。本脚本默认 `--fall-mode env`
复刻它, 同时另报真 roll/pitch (`roll_max_deg`/`pitch_max_deg`) 与 `--fall-mode tilt` 口径。
阈值可调: `--healthy-z-min/--healthy-z-max/--healthy-qxy-deg` (Go1 是 0.22;
`sim/eval_walk.py` 的测试环境用 17.5°)。

### 2.4 reset

qpos = home + U(−0.1,0.1) (全 19 维, 含四元数), ctrl = home_ctrl + 0.1·N(0,1),
`--spawn home|origin|random`。random 复刻训练: xy~U(−5,5)、随机 yaw、
z = h_ground + 0.32、基座初速 ~U(−0.3,0.3)。`--reset-noise 0` 可关掉全部初值噪声。

## 3. 物理档位 (--solver): 为什么默认 cpu

静置 home 姿态 + PD(kp=20,kd=1), 8 s, 无初值噪声 (`--dummy hold_home`):

| 模型 | cpu 100/50 | env 20/10 | soft 2/5 |
|---|---|---|---|
| `models/go2/scene_mjx_position.xml` (软足 solimp=0.9/0.95) | z=0.267 m, pitch 1.5-3.0°, **存活 100%** | z=0.226 m, pitch 漂到 8.5-19.8°, **存活 0%** | z=0.219 m, 勉强 |
| 部署 `scene_flat.xml` (硬足 condim=6) | z=0.273 m, pitch 1.5°, **存活 100%** | z=0.273 m, pitch 1.5°, 100% | z=0.230 m, pitch 7.8°, 0% |

* **默认 `--solver cpu` (MuJoCo 完整 Newton 100/50)**: 复现了任务给的实测值
  (home + PD 沉降 z≈0.268 m / 1.58°), 也是 "CPU Newton 语义" 的参考仓库对齐档。
* `--solver env` (env 的 hard_contact_iters=20/10) 对**软接触足端**的 Go2 训练模型
  接触约束不收敛: 躯干一路下沉、俯仰漂到 20° → **假摔**。这不是策略问题。
  评测 Go2 策略请保持 `--solver cpu`。
* 复现: `python logs/go2_eval/probe_settle_go2.py`。

## 4. 链路自检 & 交叉验证结果 (CPU MuJoCo, 2026-10-17)

| 运行 | 策略/动作 | 地形 | 规模 | 存活(env) | vx (m/s) | z_rel (m) | 其他 |
|---|---|---|---|---|---|---|---|
| selfcheck_hold_home | 无策略 / 顶住 home | 平地 | 4×10s | **100%** | −0.008 | 0.265 | 四足占空比 0.99, 全程 2.3s |
| selfcheck_zero | 无策略 / 零动作 | 平地 | 4×10s | 0% | −0.599 | 0.330 | 0.59s 判摔; 越软限位 17.8% |
| selfcheck_random | 无策略 / 随机 | 平地 | 4×10s | 0% | −0.465 | 0.358 | 0.53s 判摔 |
| selfcheck_hold_terrain | 无策略 / 顶住 home | **hfield** | 4×8s | **100%** | −0.008 | 0.268 | tilt max 5.2°, 分层 h<5cm 通过 100% |
| selfcheck_go1pkl_on_go2_terrain | Go1 策略 (91 维) | **hfield** | 4×8s | 0% (预期) | +0.040 | 0.263 | trot 72%, 0.9s 摔 —— Go1 策略在 Go2 上摔**是预期**, 只是证明加载→闭环→指标→出图全通 |
| **validate_go1_flat** ★ | Go1 策略 / Go1 模型 | 平地 | 8×15s | **87.5%** | **+0.511** | 0.280 | 跟踪误差 0.082 m/s, 稳态占比 88%, trot 91%, 四足占空比 0.43-0.53, 滑移 0.137 m/s, 前进 **6.80 m** |
| xcheck_go1_terrain_normoff | Go1 策略 / Go1 模型 | Go1 地形 | 12×15s | 16.7% (tilt 口径 75%) | +0.368 | 0.278 | 复刻 `eval_walk.py` 测试环境 (随机出生 + 17.5° 口径); trot 88%, 前进 −0.45±2.16 m |

★ **`validate_go1_flat` 是本 harness 的关键验证**: 用 Go1 策略在 Go1 模型上跑出
"跟踪误差 0.082 m/s、trot 91%、15s 走 6.8 m" 的干净步态, 说明 obs 拼装 (含 35 维扫描喂 0
与 8 维相位)、动作映射、物理与指标计算**整条链路是对的**。若 obs 有哪怕一段错位,
这个策略会在 1 秒内倒下 (对照: 归一化开关搞错时就是 0.2-0.4 s 摔)。

判读:
1. **物理与指标链路可信**: hold_home 在平地/地形上都能站满全程, 躯干高度 0.265-0.268 m
   与任务实测值 (0.268 m) 一致; 零/随机动作都在 0.6 s 内判摔 —— 摔倒检测、占空比、
   滑移、限位饱和、地形分层都有非平凡输出。
2. **地形档确实生效**: hfield 128×128, size=[6,6,0.3,0.096], h∈[0,0.30] m; 原点在压平区
   (h_ground≈0), 走远后分层统计随之变化。
3. Go1 策略在 Go2 上 0.9 s 摔倒 —— 符合任务预期 (只证明脚本可用, 不是 Go2 结论)。

## 5. 指标定义 (报告里每一项怎么算)

| 指标 | 定义 |
|---|---|
| 速度跟踪 | 机体系速度 R^T·qvel[:3] 的均值/标准差; 跟踪误差 mean‖v_cmd[:2] − v_body[:2]‖; 稳态段 = `--warmup` (默认 2s) 之后 |
| 稳态 | 未摔倒 且 稳态段 vx 标准差 < `--steady-std`(0.15) 且跑满时长 |
| 存活 | `--fall-mode env` 判据下未终止的 episode 比例 (另报 tilt 口径、平均/中位跌落时间) |
| 姿态 | z_rel = z − h_ground 的均值/最低; 真 roll/pitch 均值与最大值; 角速度 RMS |
| 步态 | 每足接触占空比 (接触 = contact 列表里 "足端 geom × 世界体 geom" 的配对, 等价训练侧 {F}_floor_found); trot 占比 = 两对角对接触状态不同的时间占比; is_trot = trot 占比>0.5 且四足占空比都在 0.35-0.65 |
| 足端滑移 | 接触帧的足端水平位移 / dt 的均值 (m/s) |
| 限位饱和 | 超出 "软限位" (范围 5% 内缩) 的 步×关节 比例 + 最大越界量; 另报目标顶到范围边界的比例 |
| 地形通过 | 从原点沿 +x 走 N 秒: 前进距离/路程; 按躯干下地形高度分箱 (默认 0/0.05/0.1/0.15/0.2/0.3/0.5 m), 每箱给 "到达轮数 / 通过轮数 / 通过率"; 摔倒位置 (xy + 摔倒处地面高) |

## 6. 命令行速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--policy` | none | pkl / orbax checkpoint 目录 / none |
| `--dummy` | hold_home | `--policy none` 时的动作: zero/hold_home/random/sine |
| `--terrain` | flat | flat / terrain (选 XML); `--xml` 可直接指定 |
| `--cmd vx vy wz` | 0.5 0 0 | 机体系速度指令 |
| `--seeds` `--duration` `--seed0` | 20 / 15 / 0 | episode 数、每轮秒数、起始种子 |
| `--out` `--tag` | logs/go2_eval / 自动 | 输出目录与文件名前缀 |
| `--solver` | cpu | cpu(100/50, 推荐) / env(20/10) / soft(2/5) / xml |
| `--action-mode` | auto→clip | clip / soft / tanh_soft |
| `--vel-frame` | body | obs 线速度坐标系 (Go2 训练是 body; Go1 是 world) |
| `--dof-order` | auto | model(qpos[7:] 原序, 同 env) / actuator(部署电机序) |
| `--home-pose` `--spawn` `--reset-noise` `--spawn-z` | keyframe / auto / 0.1 / 0.32 | 初值 |
| `--fall-mode` `--healthy-*` `--fall-tilt-deg` | env / 见 §2.3 / 30 | 存活判据 |
| `--height-scan` `--phase` | auto | 由策略 obs 维度自动判定 (48/56/83/91) |
| `--normalize` `--dist` | auto | brax obs 归一化 / 动作分布 (见坑 4) |
| `--foot-contact` | xml | xml=保持模型自带; soft=强制 MJX 软接触覆写 |
| `--dump-trace` `--no-plot` `--quiet` | off | 附加输出 |

## 7. 读源码发现的坑 (与直觉/文档字符串不符, 已按源码实现)

1. **`obs_scale.dofs_velocity=0.05` 没有被使用**。`_get_obs` 里是
   `dofs_velocity = data.qvel[6:]` 原样进 obs (go2_walk.py:1085); 文件头文档字符串写的
   "qvel_j*0.05" 与代码不符。部署侧 `src/go2_bridge/obs.py:343` 也标注了 "不缩放!"。
   本脚本照源码。
2. **动作没有 "soft range 内缩 5%" 的线性映射**。5% 内缩只用在 joint_limit 奖励
   (`_soft_lowers/_soft_uppers`); `step()` 是 `clip(action, ctrlrange)` ——
   与部署侧 "先按关节范围夹紧, 再下发 kp/kd" 一致 (`controllers.py:283-294`)。
3. **终止判据卡四元数分量** |q[1]|,|q[2]| < 10° (≈ 真欧拉 ±20°); Go2 的
   healthy_z_range=(0.19,0.65) (Go1 是 0.22)。跨模型评测一定要改 `--healthy-*`,
   否则会把 "站得住但倾了 12°" 的轮次判成摔倒。
4. **brax 的 obs 归一化: normalizer.count>0 不是判定依据**。`ppo.train` 返回的
   params = (normalizer, policy, value), 而 `running_statistics.update` 是
   **无条件**执行的 (`brax/training/agents/ppo/train.py:611-619`) ——
   即使 normalize_observations=False, running stats 也会被更新。所以必须看训练开关:
   * orbax checkpoint: 读 `ppo_network_config.json` 的 normalize_observations (可靠);
   * pkl: 没有记录 → 本脚本 auto 默认 **off** (与 `sim/common.py` / `sim/eval_walk.py`
     一致), 要显式打开用 `--normalize on`;
   * 若打开, obs 必须包成 {"state": obs} 传给推理函数, brax 才会用 `normalizer_select`
     取对那一份 (state) 统计量 (`networks.py:582-589, 747-755`);
   * **实测 A/B** (Go1 策略 + Go1 平地 + 8×15s, 其余参数相同):
     `--normalize off` → 存活 **100%**、vx **0.462**、跟踪误差 0.082;
     `--normalize on`  → 存活 66.7%、vx 0.101、跟踪误差 0.440。
     即该 pkl 训练时 normalize_observations=False, 用 on 会把量纲搞错。
5. **模型关节顺序必须与执行器顺序一致**: obs 的 12:24/24:36 段是 qpos[7:]/qvel[6:]
   (模型序), 而 ctrl 是执行器序。Go1 的 `go1_mjx_position.xml` 两者同序 (FR,FL,RR,RL);
   **部署仓库的 `models/go2/go2.xml` 关节声明序是 FL,FR,RL,RR 而执行器是 FR,FL,RR,RL** ——
   本脚本自动检测并打印 warn, 按 `--dof-order actuator` 走部署电机序。
   `models/go2/go2_mjx_position.xml` 已同序 (自检无 warn)。
6. 部署模型的 keyframe home 是**部署站姿** (每腿 0,0.9,−1.8, z=0.27), 不是训练姿态;
   训练模型的是 Go2 惯例姿态 (hip ±0.1 / 前腿 thigh 0.8 / 后腿 1.0 / calf −1.5, z=0.32)。
   `--home-pose unitree_rl_gym` 可显式指定后者 (兼容 X_joint 与 X 两种关节命名)。

## 8. 已知问题 / 待办

1. **地形档交叉验证与实验日志仍有差距**: xcheck_go1_terrain_normoff (terrain + 随机出生 +
   17.5° 口径, 12×15s) 得存活 16.7% (tilt 30° 口径 75%), vx 0.368, 跟踪误差 0.19;
   而日志参照值 (`docs/experiment-log.md` §19.14/§25) 是 v20@20.05M 存活 56.6% / vx 0.187,
   v22@40M 存活 80.9% / vx_body 0.215。按可能性排序的原因:
   (a) `policies/go1_walk_policy.pkl` 与日志里那几个 checkpoint **未必是同一个**
       (v20 的交付物是 `models/go1_walk_policy_v20_20M.pkl`, 且 §19.14.5 说明最后一次
       `pickle.dump` 是手动补的);
   (b) 评测口径: 本脚本固定 cmd=0.5 (最难档) 且初速 0, 日志的评估脚本 `sample=True`
       (指令含 vx=0 档) 且带 ±0.3 出生速度扰动 → 日志口径更容易;
   (c) 后端物理: v20/v22 训练是 MJX-warp + 软接触足端 + iterations=2, 本脚本是 CPU Newton +
       XML 原样; `eval_walk.py` 也是硬接触口径, 但没有逐 episode 对拍过。
   注意存活那 16.7% 的缺口主要来自 z 窗口 (Go1 z_rel 下界 0.22): tilt 口径下 75% 存活,
   说明摔的轮次是 "蹲下去了" 而不是 "翻过去了"。
   **建议**: 等 GPU 空出来用 `sim/eval_walk.py` 对同一个 pkl 跑一次并排对表; 或把训练侧 env
   用 `impl="jax"` 起在 CPU 上做逐位 obs 对拍 (见待办 2)。
2. **obs 逐位对拍: 已拿到 step-0 的逐位一致证据, 后续步仍待补**
   (`logs/go2_eval/parity_check_sim.py`, `_parity_min.py`): 直接起训练侧 `Go1Walk` (MJX-JAX
   后端, 跑 CPU), 把它的 `MjData` 灌进我的 numpy 实现, 比 `env.obs["state"]`:
   * **reset 后第 0 步: 48 维全部 max|Δ| = 5.96e-08 (float32 精度级, 即逐位一致)** ——
     这一条同时锁定了布局、各段缩放、`site_xmat` 重力、`qpos[7:]−key_ctrl[0]`、
     `qvel[6:]` 不缩放、`last_action` 与 ±100 clip。
   * 第 1 步: 修正 `--vel-frame` (Go1 训练侧是 `frame="global"`, 对应本脚本的
     `--vel-frame world`) 后 **48 维里 45 维仍逐位一致** (0:6 速度段、9:48 指令/关节/动作段
     的 max|Δ| ≤ 5e-7), 只剩 **6:9 projected_gravity 段** 有 5.4e-03 的差 (≈0.3°),
     且第 2 步同量级。该残差来自 MJX 与 MuJoCo 两个引擎对 site 姿态矩阵/四元数归一化的
     实现细节, 不是 obs 拼接问题 (若拼错段, 差会是 O(1) 的整段错位)。
     **结论: obs 拼接与缩放已逐位对齐; 唯一未解释项是跨引擎的 3 维重力数值差 (~5e-3)。**
   * Go2 模型无法用 MJX-JAX 起 env (不支持 cylinder-box 碰撞对, `put_model` 抛
     NotImplementedError); Go2 侧的正确性由 §4 的 `validate_go1_flat` 行为验证 +
     维度指纹 (91/162) 保证。
3. **orbax checkpoint 分支未实测**: 代码路径与 `sim/common.py` 一致
   (`ppo_network_config.json` + `ppo_ckpt.load`), 但本机还没有 Go2 checkpoint。
   注意 orbax 要求**绝对路径** (日志 §18.3 的坑); 本脚本内部会用 `os.path.abspath`。
4. **无 hfield 模型上的高度扫描喂 0**: 策略需要 35 维扫描但用平地 XML 时, 本脚本按部署侧
   `obs.py` 的约定喂 0 并打印 warn (平地上该特征本就 ≈ −0.01)。要评地形能力必须用
   `models/go2/scene_mjx_terrain.xml`。
5. **地形难度分层口径**: 现在按 "躯干下方地形高度" 分箱。若想按台阶/坡度分箱, 可复用
   asset-smith 生成的 `models/go2/assets/terrain.json` 元数据。

## 9. Go2 策略训出来后怎么评 (建议流程)

~~~bash
# 1) 冒烟 (4 seed × 8s, 平地)
python sim/eval_go2.py --policy CKPT_OR_PKL --terrain flat --seeds 4 --duration 8

# 2) 正式平地评估 (20 seed × 15s): 存活率 / 速度跟踪 / 步态 / 姿态
python sim/eval_go2.py --policy PKL --terrain flat --cmd 0.5 0 0 \
    --seeds 20 --duration 15 --tag go2_flat_v1 --dump-trace

# 3) 地形通过率 (20 seed × 15s, 从原点沿 +x)
python sim/eval_go2.py --policy PKL --terrain terrain --cmd 0.5 0 0 \
    --seeds 20 --duration 15 --tag go2_terrain_v1

# 4) 多指令档 (侧移/转向; 指令是机体系)
for c in "0.3 0 0" "0.6 0 0" "0.0 0.3 0" "0.3 0 0.5"; do
  python sim/eval_go2.py --policy PKL --terrain terrain --cmd $c \
      --seeds 10 --duration 12 --tag "go2_cmd_$c"
done
~~~

判据建议 (目标 = "能在一定复杂地形上稳定行走"):
* 平地存活率 ≥ 95%, 地形存活率 ≥ 70%;
* 稳态 vx 跟踪误差 ≤ 0.15 m/s, 轮内 vx 标准差 ≤ 0.15;
* 四足占空比都在 0.35-0.65 且 trot 占比 > 50%;
* 地形分层: h≥10cm 档通过率 ≥ 50%, 摔倒位置不集中在第一档;
* 关节越软限位比例 < 1%, 力矩均值 < 15 N·m (峰值 < 45 N·m)。
