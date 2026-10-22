# Go2 v3: 观测里的"真实速度估计误差"

> 唯一改动: **actor 的 obs 0:3 (基座线速度)** 从"真值 + ±0.05 m/s 白噪声"
> 换成"腿式里程计的真实误差形态": **常值偏置 + OU 慢漂 + 观测延迟** (+ 更大白噪声)。
> critic 与奖励仍用真值。地形/奖励/接触参数与 v2 完全一致 (单变量)。
>
> 代码: `envs/go2_walk.py` (`obs_noise.lin_vel_bias/drift_sigma/drift_tau/delay_steps`) +
> `train/train_go2.py` (`--obs_vel_bias/--obs_vel_drift/--obs_vel_tau/--obs_vel_delay/--obs_noise_linvel`)；
> 验证脚本: `sim/probe_obs_vel_error.py`；训练入口: `sim/start_go2_v3_train.sh`。

---

## 1. 为什么改这个

v2 训练里 obs 0:3 就是 `data.qvel[0:3]` (真值) + `obs_noise.lin_vel=0.05` 的均匀白噪声。
真机上没有真值, 只有腿式里程计; 而里程计的误差**不是白噪声**, 是:

1. **常值偏置** (装配/足端半径/接触阈值不对称) —— 白噪声平均掉, 偏置不会;
2. **慢漂** (打滑、滤波状态、姿态误差累积) —— 时间相关, 策略若盲目信任会被带偏;
3. **延迟** (接触检测 + 低通滤波, 20–40ms = 1–2 个控制步) —— 高速时导致相位滞后。

只加白噪声等于假设"误差无偏且无记忆", 与真实误差的结构相反。

### 1.1 实测校准 (sim 内现有 LegOdometry, 12s 闭环, 世界系对真值)

| 统计量 | vx | vy |
|---|---|---|
| 偏置 (mean) | −0.073 m/s | −0.043 m/s |
| 即时误差 std | 0.231 m/s | 0.160 m/s |
| 1s 平滑后 std (=慢漂) | 0.061 m/s | 0.034 m/s |
| 速度量级 | est 0.546 vs truth 0.568 | |

真机会更差 (编码器量化、IMU 零偏、足端打滑)。所以训练注入取比 sim 实测略保守:

| 参数 | 训练注入 | 依据 |
|---|---|---|
| `lin_vel_bias` | **0.12 m/s** (逐轴 U(−0.12,0.12)) | sim 偏置 0.07, 真机更大 |
| `lin_vel_drift_sigma` | **0.08 m/s** (OU, τ=2s) | sim 慢漂 std 0.06 |
| `lin_vel_delay_steps` | **2 步** (0–2 步 = 0–40ms) | 接触检测+滤波的典型滞后 |
| `lin_vel` 白噪声 | **0.15 m/s** (原 0.05) | sim 即时 std 0.23; legged_gym 用 0.5 |

注: `--obs_noise` 本身的 `lin_vel` 是**白噪声** (传感器抖动), 与上面的
bias/drift/delay (估计器误差) 是两个独立来源, 同时打开。

---

## 2. 实现

### 2.1 env 侧 (`envs/go2_walk.py`)

`default_config().obs_noise` 新增 4 个字段 (默认全 0 = 与 v2 逐位一致):

```python
lin_vel_bias=0.0,        # 每回合常数偏置半宽 (m/s, 逐轴 U(-b,b))
lin_vel_drift_sigma=0.0, # OU 稳态 std (m/s)
lin_vel_drift_tau=2.0,   # OU 相关时间 (s)
lin_vel_delay_steps=0,   # 最大延迟 (控制步, 50Hz -> 1 步 = 20ms)
```

* `reset`: 采样每回合 `vel_bias` (逐轴均匀)、`vel_delay` (0..D 均匀)、
  初始化延迟环形缓冲 `vel_hist` (D+1 步) 和 OU 状态 `vel_drift=0`;
* `step`: 先按延迟取历史速度, 再叠加 bias + OU 漂移, 得到
  `vel_est_world` (世界系, 与部署侧 `obs.py` 的输入同语义 —— 由它负责转机体系);
* `_get_obs`: **actor** 的 0:3 用 `vel_est` 转机体系 ×2;
  **critic** (`privileged_state`) 和**奖励**继续用真值 —— 特权信息不污染;
* 新增训练指标 `obs/vel_err` = ‖est − truth‖ (m/s), 在训练日志里能直接看到
  注入误差的实际量级。

### 2.2 训练侧 (`train/train_go2.py`)

```bash
--obs_vel_bias 0.12 --obs_vel_drift 0.08 --obs_vel_tau 2.0 --obs_vel_delay 2 \
--obs_noise_linvel 0.15
```

不传这些参数 = v2 行为逐位不变 (已在 `sim/probe_obs_vel_error.py` 的 A 档验证:
全 0 时 `max|est-truth| = 0.000e+00`)。

---

## 3. 验证 (`sim/probe_obs_vel_error.py`, 8 env × 120 步)

| 档位 | 每回合偏置 \|mean\| | 即时误差 std | 总 RMS | 自相关 lag1/lag10/lag50 | 延迟分布 |
|---|---|---|---|---|---|
| A) 全 0 (v2) | **0** (`max|est−truth|=0`) | 0 | 0 | — | — |
| B) v3 推荐档 | 0.066 m/s | 0.089 | 0.121 | +0.86 / +0.45 / **+0.32** | [3,2,3] (0/1/2 步均匀) |
| C) 对齐 sim 实测 | 0.041 | 0.043 | 0.068 | +0.90 / +0.70 / **+0.51** | — |

* A 档证明**对旧配方零影响**;
* B 档的 lag50 = +0.32 (而白噪声应 ≈0) 说明慢漂成分确实注入了;
* D 档检查 actor obs: `obs[0:3]/2` 与 `est_body` 的差 = 白噪声幅度 (0.14 ≤ 0.15), 说明
  0:3 确实换成了估计值且白噪声按预期叠加。

---

## 4. 训练 (`sim/start_go2_v3_train.sh`)

```bash
cd mjx-go1-getup
bash sim/start_go2_v3_train.sh               # 40M 步, ~2.5h, logdir models/go2_v3_logs
bash sim/start_go2_v3_train.sh 10000000      # 或先跑 10M 看趋势
```

与 v2 的唯一差异就是上面那 5 个 obs 参数 (地形、奖励、接触、envs=3072/显存 0.45 均按 v2 经验)。

---

## 5. 训完必须做的验收 (否则不知道这改动有没有用)

1. **带估计器闭环**: 可视化脚本里用 `--linvel estimator` (真机路径) 跑 15s,
   与 v2 对比: 偏航漂移、速度跟踪误差、横滚偏置、存活。命令:
   `bash scripts/view_go2_policy.sh --headless --duration 15 --record logs/v3_est --linvel estimator`
2. **带真值闭环** (`--linvel oracle`): 确认为了鲁棒性没有把最优点牺牲太多
   (预期: 比 v2 略差 1-3% 属正常代价)。
3. **地形评测** (`sim/eval_go2.py --terrain terrain` 与 `probe_go2_warp_eval.py`):
   若明显变差, 说明噪声过大, 回到 §1.1 表格把 bias/drift 各降 1/3 重训。
4. 训练日志看 `obs/vel_err` 是否落在预期量级 (0.1–0.2 m/s)。

---

## 6. 诚实的边界

* 这个改动让策略**不迷信**速度观测, 但**不能修好里程计**。真机上仍需:
  ①修 LegOdometry (当前 sim 里偏置 0.07、瞬时误差 0.23 已经不小);
  ②导航层用外部定位/heading 纠偏 (策略的偏航漂移问题本轮按用户要求不处理)。
* 注入的是"结构正确、参数近似"的误差, 不是真机误差的精确分布 —— 参数只能靠
  真机数据回流继续标定。
* critic 用真值是有意的 (不对称 actor-critic 的特权学习), 部署时 critic 不存在, 不影响。

---

## 7. 事故与修正 (2026-10-18)

第一次 v3 训练 (models/go2_v3_logs) 在 33.03M 步 **CUDA OOM 崩掉**
(Failed to allocate 55MB, 与 v2 同一类错误: 8GB 卡与主机桌面共享, 桌面占 ~2.8GB),
且事后核查发现启动脚本**漏了 `--hard_contact`** —— 那次跑的是软接触物理
(= v1 的 5.2x sim2sim gap 那套), 与部署模型不一致, **不能作为 v2+obs误差 的单变量对照**,
其 33M 检查点作废 (目录保留作证据)。

**修正后的 v3** (`sim/start_go2_v3_finetune.sh`):
* 从 **v2 硬接触最终检查点** (models/go2_v2c_logs/checkpoints/000007372800,
  37.26M 步, 已收敛) **续训 10M 步**, lr 2e-4;
* 唯一语义变化 = obs 0:3 的估计误差 (bias±0.12 / OU 0.08/2s / 延迟 0..2 步 / 白噪声 0.15);
* `--hard_contact` 已确认打开 (日志: "v5 hard_contact: 足端硬接触 ... Newton 20 / ls 10");
* 显存: envs 3072→2560, mem fraction 0.45→0.42, num_evals 16→6 (减少 eval 期峰值);
* 代价 (诚实标注): 比 v2 多 10M 步, A/B 对比时不能把"额外训练"与"观测误差"完全分开;
  但 v2 在 30M 后 eval 已平台 (1295→1330→1295), 10M 额外步数贡献预计很小。

**v4** (混合地形) 同时修正: 原脚本也漏了 `--hard_contact`, 已补; envs/mem 同上。
接力顺序: v3 微调完成 → 三组 CPU 评测存证 → v4 自动启动。

---

## 8. A/B 结果 (2026-10-18): 强误差注入是**负结果**

v3 (从 v2 硬接触 37.26M 续训 10M 步, obs 0:3 = 真值 + bias 0.12 + OU 0.08/2s +
延迟 0..2 步 + 白噪声 0.15) 完成后, 用 `scripts/eval_v3_vs_v2.sh` 做了
v2/v3 x oracle/estimator 四组 15s 平地闭环 (同一份部署模型/控制器/指令, 只换策略与速度来源):

| 组 | 前向位移 | 机体速度 | yaw 漂移 | 横滚/倾角 | 备注 |
|---|---|---|---|---|---|
| v2 + oracle | (6.63, -1.84) | 0.541 m/s | **-3.2 deg/s** | tilt 2.5° | 基线 |
| v2 + estimator | (7.44, +0.72) | 0.572 | -1.5 | z 0.180 (蹲) | 里程计误差大 (std 0.25) |
| v3 + oracle | (5.86, -4.93) | **0.663 m/s (+33%)** | **-7.0 deg/s** | tilt 2.9° | 严重超速 + 偏航加倍 |
| v3 + estimator | (0.62, -4.89) | 0.440 | **-8.0 deg/s** (yaw -13° -> -116°) | tilt 2.8° | 基本在原地转圈 |

同时 v3 的训练 eval_reward 反而更高 (best 1508 vs v2 1330, 都是带噪观测下评测),
但闭环行为更差: **+33% 超速、偏航漂移 2 倍、对角占空比不平衡从 1.47x 恶化到 1.84~2.07x**。

解释: 训练时 obs 0:3 带"延迟 + 偏置", 策略学到的是"补偿滞后的观测量"; 部署时输入
(oracle 或实际里程计) 的滞后/偏置与训练分布不同 -> 过度反应 (超速) 且左右蹬地不对称被放大。
**教训**: 估计误差注入必须与真实估计器统计量**匹配**, 并且 eval_reward (在同样带噪观测下)
不能作为部署好坏的判据 —— 必须用 oracle/estimator 双闭环 A/B。

**决定**: v3 不作交付 (保留 policies/go2_walk_v3.pkl 作为负结果证据); v4 回退到 v2 的观测
设置 (白噪声 0.05, 无 bias/漂移/延迟), 只把地形换成混合地形 —— 单变量 = 地形。
估计误差的想法需要重新标定 (更小的 bias/无延迟/按真实里程计谱匹配) 再试。
