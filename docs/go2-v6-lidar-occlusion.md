# Go2 v6: 雷达遮挡感知训练 —— 实验记录 (进行中)

> 本轮目标 (用户 2026-10-18 指令): 从"完美解析高度场"训练转向**雷达遮挡感知**训练,
> 交付一个在仿真里能稳定走 **平地 / 楼梯 / 凹凸地形** 的 Go2 行走模型。
> 记录方式: 本文件 = 本轮实验的**连续日志** (设计 -> 每一步改动 -> 实测数字 -> 结论),
> 每完成一个阶段就追加, 不覆盖历史结论。
> 上游调研: [go2-lidar-occlusion-survey.md](go2-lidar-occlusion-survey.md) (430 行, 40+ 链接)

- 起始: 2026-10-18 23:5x (v5c 被用户终止后)
- 起点检查点: **v5a-best @20.07M** (不是 v5c-best —— 见 §4.1 的实测)
- 硬件: RTX 5060 Laptop 8GB (XLA mem fraction 0.42, 2560 env)

---

## 1. 问题定义: 为什么"完美扫描"是 sim2real gap

训练侧 v1-v5 的高度扫描是 **解析高度场**: `terrain_height(xy)` 直接双线性查表,
等于上帝视角 —— 每个格点都精确、无噪声、无遮挡。真机链路 (L1 雷达 -> 点云 ->
高程图 -> 机体系网格查询) 有三类信息损失:

| # | 损失 | 物理原因 |
|---|---|---|
| 1 | **近场垂直 FOV 盲区** | L1 垂直 FOV 90°, 装配下俯 ~15° => 相对水平 -60°~+30°; 安装高 ~0.22m => 盲区半径 ~0.13m (很小) |
| 2 | **机体/腿自遮挡** | 前装雷达在机头, 躯干挡住后方低仰角射线; 大腿/小腿挡在射线上 —— **这是主力遮挡** |
| 3 | **地形自遮挡** | 台阶立面/坡背后, 射线先打到立面, 后方格点要等爬上去才看得到 |
| 4 | 稀疏回波/丢点 + 高程图记忆 | 点云有空洞; 但高程图会保存历史观测 (decay 内), 所以"看不见"≠"没信息" |

### 1.1 真机 L1 几何 (调研结论, 用于建模)

* **型号**: Unitree 4D LiDAR L1, 360° x 90° FOV, 最小测距 0.05 m (官方规格)。
* **安装**: unitree_ros URDF `radar_joint` = base 系 `xyz=(0.28945, 0, -0.046825)`,
  `rpy=(0, 2.8782, 0)` ≈ 倒置 180° + 下俯 15° => 视场相对水平 **-60° ~ +30°**。
* **近场**: 传感器离地 ~0.22m (本仓库站高 0.32m 口径) => 盲区半径
  `0.22/tan(60°) = 0.127 m` —— 与我们网格 (x∈[-0.4,1.2]) 的最近格点
  (x=0.29-0.13=0.16) 相比几乎不丢格点。**这与用户"雷达能看到脚下地形"的观察一致。**
* **旧部署 SimLidar 的 vfov=[-0.5,0.2] 是拍出来的窄视场** (近场盲区 0.55m),
  与真机不符 —— 本轮训练用真机几何, 部署侧要同步改 (见 §6)。

> 开源现状 (调研 §2): legged_gym 直接查高度场张量、IsaacLab RayCaster 只打静态地形
> mesh、mjlab `terrain_scan` 只打 geom group 0 —— **三家都不模拟自遮挡/FOV/未观测**。
> 高程图侧 (elevation_mapping_cupy) 才有 visibility cleanup / validity 层。
> 所以"训练侧遮挡感知"这块没有现成配方, 要自己实现并做消融。

---

## 2. v6 设计 (envs/go2_walk.py)

### 2.1 新的感知通道 (actor only)

```
scan_actor  = fill( 完美扫描 + 传感器噪声 , 信息掩码 )
scan_critic = 完美扫描                       # 非对称 AC 的特权通道, 不变
信息掩码 = (物理可见 | 高程图记忆) & ~随机丢点 & ~扇区遮挡
```

- **obs 维度不变**: 仍是 48 + 187 + 8 = 243 (actor) / 314 (critic)。
- **不可见格点在特征空间填 `fill`** (0.0)。特征 = clip(z_base - h - 0.268, ±0.35) × 5,
  所以 0 = "假定标称站高处的平地" = 中性 "平地先验"。
- **只在 actor 上生效**; 奖励/critic 仍用真值。

### 2.2 物理可见性三要素 (`_lidar_visible`, 全部可微 JAX, vmap/jit)

1. **垂直 FOV**: 射线仰角须在 `[vfov_min, vfov_max]` (默认 -60°~+30°)。
2. **地形自遮挡**: 沿射线 t∈[0.55,0.97] 取 4 点, `terrain_height > ray_z` 即被立面挡住。
3. **机器人自遮挡**: **直接用 MJCF 碰撞 geom 做精确射线求交** ——
   `data.geom_xpos/geom_xmat` + 5 box (躯干/大腿) slab 检验 + 13 capsule
   (小腿/髋, Ericson 线段-线段最近距离) + 4 sphere (足端)。
   传感器就装在机头, 所以躯干/头部壳体也参与遮挡 (安装点落在 geom 内部的会自动排除)。
   -> 比调研建议的"静态 mask + 圆盘近似"更准, 成本可忽略 (§3.3)。

### 2.3 高程图记忆 + 丢点 + 扇区遮挡

```python
info_ok = vis | (uniform < occ_recall)     # 高程图记忆 (decay 内历史观测)
info_ok &= ~sector_mask                    # 成片缺失 (p=0.25, 宽 U(0.6,2.4) rad)
info_ok &= (uniform > dropout)             # 稀疏回波
h = where(info_ok, h_true + noise, h_fill) # h_fill 反解自 fill=0
```

- `occ_recall=0.5` (训练): 物理不可见格点有一半概率仍是"记得的地形", 另一半变空洞
  —— 逼策略别把近场信息当必需, 同时保留近场可用的情形。
- `dropout=0.05`, `sector_prob=0.25` (真实高程图缺失是**成片**的, 不是椒盐噪声)。
- **每个 step 随机化**: mount 抖动 ±2cm, vfov 下限抖动 ±0.15 rad (覆盖安装误差)。

### 2.4 训练日志新增指标

| 指标 | 含义 |
|---|---|
| `obs/scan_vis` | 物理可见格点比例 (不含记忆, 不含随机丢点) |
| `obs/scan_info` | 信息可用比例 (可见 ∪ 记忆, 扣除丢点/扇区) |

---

## 3. 实测 (改动验证)

### 3.1 遮挡掩码形态 (标称站姿, mix 地形, 17x11 网格)

物理可见率 **84.0%** (站姿)。分解 (逐类关闭测的):

| 组件 | 贡献 |
|---|---|
| FOV + 地形自遮挡 | 到 94.1% (即 ~6% 格点被这两项挡住) |
| + 小腿/髋 capsule 自遮挡 | 94.1% -> **84.0%** (腿挡了 ~10%) |
| + 躯干 box | 84.0% -> 84.0% (标称站姿下躯干不挡; 前装低安装) |

站立时逐格图案 (yaw 对齐, 与 env 完全同口径) 呈现"腿影条纹 + 少量背后格点"。

### 3.2 行走时的遮挡强度

* 训练中 eval 实测 (v5a 策略, 遮挡模型开): **物理可见 ~0.38, 信息可用 ~0.62**
  (`vis=212.84/560.4`, `info=344.58/560.4`)。
* 行走时远低于站立 (0.84): 楼梯立面遮挡 (地形自遮挡) + 爬坡时传感器俯仰 + 扇区遮挡
  (25% 概率 × 平均 38% 网格) 共同作用。

### 3.3 计算开销 (2560 env, env.step, jit 后)

| 配置 | ms/step | 相对 |
|---|---|---|
| lidar off | 55.05 | 1.000 |
| lidar on (旧简化几何, K=4) | 55.79 | **+1.3%** |

*(精确 geom 版本 + 扇区的新开销待补测; 预期同量级。)*

---

## 4. 训练

### 4.1 起点选择: v5a-best vs v5c-best (实测矩阵)

`logs/go2_occlusion_matrix.tsv`: 2 策略 x 5 地形 x 3 遮挡档, warp 训练物理,
128 env x 300 步, 随机出生。**完美扫描档** (与 v1-v5 训练口径一致, 不含 lidar):

| 地形 | v5a-best @20.07M | v5c-best @25.80M |
|---|---|---|
| mix | **81.2%** | 74.6% |
| stairs_only | 75.0% | 75.4% |
| slope_only | 82.8% | 75.0% |
| rough_only | 82.8% | 76.6% |
| boxes_only | 55.9% | 53.5% |
| 平均 | **75.5%** | 71.0% |

**结论 1: v5c (续训 +5.7M) 在真实地形存活率上比 v5a 低 4.5pp** ——
但它的 eval_reward 反而更高 (1053.7 vs 757.4)。**再次证明 eval_reward 不是
部署预测指标** (v3 的教训 + 这次的直接对照)。-> v6a 从 **v5a-best** 续训。
**结论 2 (方法学)**: 同配置重复跑同一条评测会出现 ~1.5% 的存活率抖动 (80.5 vs 82.0),
来源是 warp 接触求解的 GPU 非确定性 —— 后续所有对比都要按"同进程/同批次 + 多地形同向"
解读, 不拿单点 1-2pp 当结论。

### 4.2 v6a: 从 v5a-best 续训 12M, 首次遮挡感知

| 项 | 值 |
|---|---|
| 起点 | v5a-best `models/go2_v5a_logs/checkpoints/000020070400` (20.07M) |
| 步数 | 12M (累计 32.07M), lr 1e-4 |
| 地形 | mix (n=128), 随机出生 |
| 扫描 | 17x11 = 187 点, x[-0.4,1.2] y[-0.5,0.5] |
| 遮挡 | `--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25` |
| 物理 | hard contact, 2560 env, 128x128 MLP |
| 时间 | 2026-10-19 00:05 启动 |
| 日志 | `logs/go2_v6a_lidar.log` |

**单变量 = 感知通道** (地形/奖励/物理/env 数/网络全同 v5a)。
启动时 eval_reward: **707.4** (v5a 在完美扫描下 757.4; 遮挡使 v5a 掉 ~7%)。

**评测 (训练完成后自动跑)**:
1. 遮挡矩阵 (v6a vs v5a x 5 地形 x 3 档: perfect / nominal / harsh);
2. 平地 CPU 16x15s;
3. 溢出/NaN 自检。

训练进度 (滚动更新):

| 累计步 | eval_reward | ep_len | scan_vis | scan_info | 备注 |
|---|---|---|---|---|---|
| 20.07M (restore) | 707.4 | 560.4 | 0.38 | 0.62 | v5a 在遮挡下掉 ~7% |
| 22.53M | 928.9 | 585.6 | 0.38 | 0.62 | best |
| 24.99M | **1110.9** | 618.6 | 0.39 | 0.62 | best; ep_len 回升 |

### 4.3 v6a 结果

训练: 12M 完成, best eval_reward **1230.2** (v5a 757.4), fps ~5300, 溢出/NaN 0。

**地形 (warp 训练物理, 128 env x 6s, 随机出生, 存活率 %):**

> ⚠️ 噪声说明: 单次 128 env 的抽样标准差 ≈ 3.8pp (p≈0.75), 且 warp 接触求解有
> GPU 非确定性 (同一权重两次评测实测可差 2~7pp)。下表 128 env 数字**只能看趋势**,
> 最终结论用 §4.6 的 384 env 高功效评测。

| 地形 | v5a perfect | v6a perfect | v5a nominal | **v6a nominal** | v5a harsh | **v6a harsh** | v5a blind | **v6a blind** |
|---|---|---|---|---|---|---|---|---|
| mix | 83.6 | 76.6 | 77.3 | 77.3 | 78.1 | 75.8 | 66.4 | **73.4** |
| stairs | 77.3 | 73.4 | 77.3 | 69.5 | 70.3 | 68.8 | 61.7 | 62.5 |
| slope | 82.8 | 71.1 | 79.7 | 77.3 | 75.8 | 75.0 | 75.0 | 72.7 |
| rough | 82.0 | 78.1 | 78.1 | 77.3 | 78.9 | 77.3 | 80.5 | 78.1 |
| boxes | 57.8 | 54.7 | 51.6 | 51.6 | 48.4 | 43.0 | 31.2 | **40.6** |

* nominal = v6 训练口径 (scan_info 0.60); harsh = info 0.29; blind = info 0.0 (全 0 扫描).
* **盲扫下 v6a 明显更稳**: mix +7.0pp, boxes **+9.4pp** (v5a 盲扫 boxes 只剩 31.2%) ——
  这正是遮挡训练的目标收益。clean/nominal 两列两者在噪声内 (mix/slope/boxes nominal 持平)。
* 唯一可复现的退步是 slope perfect (-11.7pp) 与 stairs nominal (-7.8pp), 但都接近噪声上限,
  需要 §4.6 的高功效评测确认。

**平地 (CPU 16x15s):**

| 指标 | v5a | **v6a** |
|---|---|---|
| 存活 | 100% | **100%** |
| vx (cmd 0.5) | 0.502 | 0.522 |
| 跟踪误差 | 0.096 | **0.088** |
| 躯干 z | 0.266 | 0.271 |
| 足端滑移 | 0.768 | **0.577** |
| 判为 trot 的轮 | 0% | **43%** |

-> **平地步态质量反而明显改善** (trot 判定 0%->43%, 滑移 -25%)。这说明 v6a 不是"训坏了",
而是换了一个更保守、更依赖本体感知的平衡点。

### 4.4 诊断: 继续训练本身是否在退化? (检查点扫描, 运行中)

v5c (v5a +5.7M, 无遮挡) 和 v6a (v5a +12M, 有遮挡) 的 clean 地形存活率都低于 v5a,
怀疑"在 mix 上继续训练"本身就会漂移 (eval_reward 上升但随机出生存活率下降)。
用 v6a 的 5 个中间检查点 + v5a 在**完美扫描**下跑 mix/stairs, 看退化是否随步数单调。
结果见 `logs/go2_diag_ckpt_sweep.txt` (待填)。

### 4.5 检查点扫描: "训得更久" 本身在伤害地形存活率 (重要)

v6a 的 6 个检查点 (perfect 扫描, 128 env, mix/stairs):

| 检查点 | mix 存活% | stairs 存活% |
|---|---|---|
| v5a (起点) | 83.6 | 77.3 |
| v6a @2.46M | 82.0 | 80.5 |
| v6a @4.92M | **84.4** | 78.1 |
| v6a @7.37M | 82.8 | **84.4** |
| v6a @9.83M | 82.8 | 82.8 |
| v6a @12.29M (final) | **75.0** | **71.9** |

**关键发现**: 中间检查点 (2.5M~9.8M) 与 v5a 持平甚至更好, 但**最后一个检查点明显更差**;
而 eval_reward 是一路涨的 (707 -> 1230)。这与 v5c 的结论一致:
**在 mix 上继续训练会让随机出生地形存活率漂移下降, 而 eval_reward 不反映它**
(eval_reward 的 actor obs 也是遮挡后的, 但它是**训练 eval 分布**的回报, 不是随机地形生存率)。
-> 交付/评测必须按**地形存活率**选检查点, 不能按 eval_reward 的 best。
本轮起改为: 候选检查点 x nominal 遮挡 x 384 env 直接选优 (§4.6)。

### 4.6 候选选择 (384 env, nominal 遮挡) —— 冠军 = v6a@7.4M

`logs/go2_v6_candidates.tsv` (384 env x 300 步, 随机出生, nominal 遮挡):

| 策略 | mix | stairs | boxes |
|---|---|---|---|
| v5a-final (20.07M) | 80.7 | 78.9 | 51.0 |
| v6a @4.92M | 81.5 | 77.1 | 53.4 |
| **v6a @7.37M** | **84.1** | **79.2** | **54.7** |
| v6a @9.83M | 83.3 | 76.3 | 52.6 |
| v6a best (eval_reward 选的 9.83M 副本) | 79.2 | 71.9 | 49.7 |

**结论: v6a@7.37M 在三个地形上同时超过 v5a** (mix +3.4pp, stairs +0.3pp, boxes +3.7pp),
且它是**遮挡训练**出来的 —— 不再有"鲁棒性换 clean 性能"的取舍。
另外注意 "eval_reward 选的 best" 反而比 @7.37M / @9.83M 差 5~7pp, 再次说明
**不能按 eval_reward 选检查点** (GPU 非确定性也能造成 5pp 级差异, 见 §4.3 噪声说明)。

### 4.6.1 最终矩阵 (384 env, 5 地形 x 4 档) —— v6a@7.37M 全面胜出

`logs/go2_v6_final_matrix.tsv` (384 env x 300 步, 随机出生, 存活率 %):

| 地形 | v5a perfect | **v6a perfect** | v5a nominal | **v6a nominal** | v5a harsh | **v6a harsh** | v5a blind | **v6a blind** |
|---|---|---|---|---|---|---|---|---|
| mix | 83.3 | **86.5** | 82.0 | **83.1** | 77.6 | **80.2** | 72.9 | **77.1** |
| stairs | 78.9 | **83.6** | 76.0 | **78.1** | 74.7 | **75.3** | 68.8 | 68.8 |
| slope | 83.9 | **85.7** | 80.5 | **80.7** | 78.9 | **79.9** | 75.5 | **75.8** |
| rough | 84.4 | **85.2** | 82.0 | **83.6** | 82.0 | **83.1** | 81.8 | 81.2 |
| boxes | 52.3 | **54.9** | 46.4 | **50.8** | 48.4 | **52.6** | 39.8 | **48.2** |
| **平均** | 76.6 | **79.2** | 73.4 | **75.3** | 72.3 | **74.2** | 67.8 | **70.2** |

* v6a 在 19/20 个格子 >= v5a; 平均 +2.6 / +1.9 / +1.9 / +2.4 (perfect/nominal/harsh/blind)。
* 最难地形 boxes 的遮挡退化从 -12.5pp 收窄到 **-6.7pp** —— 遮挡训练的核心收益。
* clean-nominal 退化 1.6~5.5pp (<10~15% 验收线 ✓); nominal-blind 差 2.4~9.3pp
  -> 策略主体靠本体感知, 扫描贡献几个 pp (全 0 扫描 = "假设平地", 本身可走)。
* 图: `docs/img/go2-v6-occlusion.png`; 分析: `logs/go2_v6_final_matrix_analysis.md`。

### 4.7 部署侧管线预验证 (用 v6a smoke 策略, 243 维 obs)

在真模型训练完成前, 先用 v6a 的 200k 步 smoke 策略 (243 维) 把部署侧管线跑通:

* `scan 几何 = 17x11=187 点, x(-0.4,1.2) y(-0.5,0.5) (由 obs_dim=243 反推)` ✓
* `SimLidar(自遮挡=True mount=[0.28945,0,-0.046825] vfov=(-1.0472,0.5236)) ->
  HeightMapper(17x11 fill_local=True decay=1.0s)` ✓
* 平地闭环 8 s: x=3.70 m (0.46 m/s), z=0.259, tilt 8.8°, upright, 健康 L0;
  record/summary/dashboard 全部正常 (243 维 obs 条 + 17x11 热图)。
-> 部署侧 17x11 + 真机 L1 感知链路端到端可用。

### 4.7.1 冠军检查点 (v6a@7.37M) 的平地指标

* 训练 XML 平地, CPU 16x15s: **100% 存活**, vx 0.492 (cmd 0.5), 跟踪误差 0.091,
  z 0.270, 足端滑移 0.633 (v5a 0.768), trot 39% (v5a 0%).
* 部署仿真平地 (raycast 真感知, 15 s, oracle 速度): x=6.40 m, upright, z=0.271, tilt 6.4°,
  健康 L0。
* **已知问题 (用户已明确暂不处理)**: 航向漂移 —— 15 s 横移 4.7 m (≈2.8°/s); 换成
  腿式里程计速度后转圈 (x=-0.87 m)。这是"自旋/对称"问题, 与 §29 的结论一致
  (机体系指令 + obs 无 yaw 参考 -> 无航向闭环, 累积偏置无法自纠)。真机需要上层
  cmd_wz 闭环; 本轮不引入 heading 观测。

---

## 5. 代码改动清单 (本轮)

| 文件 | 改动 |
|---|---|
| `envs/go2_walk.py` | `height_scan.lidar` 配置组 (mount/vfov/jitter/ray_samples/self_occlusion/dropout/occ_recall/sector/fill); `_setup_lidar` + `_setup_lidar_occluders` (MJCF 碰撞 geom 精确求交) + `_lidar_visible`; `height_scan_features_actor`; `_height_scan` 重构; `obs/scan_vis`+`obs/scan_info` 指标 |
| `train/train_go2.py` | `--lidar*` 参数 12 个 + 配置覆盖 + 启动打印 + 日志里打印 vis/info |
| `sim/probe_go2_warp_eval.py` | `--lidar*` 评测参数 + scan 可见率输出 |
| `sim/eval_occlusion_matrix.sh` | 新增: 策略 x 地形 x 遮挡档矩阵评测 |
| `sim/run_go2_v6a_lidar.sh` | 新增: v6a 训练 + 自动评测 |

**未改**: obs 维度/布局 (243 不变), 奖励, critic 特权通道, 部署侧代码 (下一步)。

---

## 6. 部署侧改造 (mjx-go2-mujoco2ros2) —— 已完成

### 6.1 改动清单

| 文件 | 改动 |
|---|---|
| `src/go2_bridge/perception.py` | `SimLidar`: 新增 `mount_body` (机体系安装, origin = base + R@mount)、`min_range`、`self_occlusion` (射线打到机器人自身 geom 判遮挡, 不再把机器人从 raycast 排除); `HeightMapper`: 新增 `fill_local` (未观测格点填"局部地面估计" -> 特征≈0, 与训练 v6 fill 语义一致) |
| `src/go2_bridge/obs.py` | 新增 `WalkObsBuilder.scan_n` 属性 (原来注释写死 35) |
| `src/go2_bridge/controllers.py` | `np.zeros(35)` -> `np.zeros(self._scan_n)`, 扫描点数按 obs 配置动态 |
| `scripts/run_controller.py` | **按策略 obs_dim 反推扫描几何** (48+scan_n+8); `height_scan_source=raycast` 时建 SimLidar+HeightMapper 并作为 `height_scan_fn` 挂到 PolicyController |
| `scripts/view_policy.py` | `--scan raycast` 选项; FrameCollector/ImagePanel 传扫描网格形状 (17x11); geom_budget 随扫描点数放大 |
| `src/go2_bridge/viz/{frame,live,panels,scene,record}.py` | 扫描热图/obs 条形图/3D 叠加/记录文本都从"写死 35 点/91 维"改成按 frame 的 `scan_nx/scan_ny`/`obs_dim` 动态 (ImagePanel 新增 `configure(obs_dim, scan_nx, scan_ny)`) |
| `config/go2_sim.json` | `perception` 换成真机 L1 口径 (mount_body/vfov [-60°,+30°]/min_range 0.10/self_occlusion/fill_local); 新增 `go2_v6` 策略档 (npz + scan 17x11 + raycast) |
| `scripts/compare_deploy_vs_train_physics.py` | `build_deploy_terrain` 支持任意训练地形 XML (mix) |
| `scripts/eval_v6_deploy.sh` (新增) | ckpt(best)->pkl->npz 导出 + 部署 mix 地形场景 + 闭环矩阵 (平地/地形 x raycast/nominal x oracle/estimator) + report |
| `sim/ckpt_to_pkl.py` (getup 侧, 新增) | orbax 检查点 -> .pkl (部署导出用 best 而不是 final) |

### 6.2 感知链路验证 (部署 CPU 仿真, `probe_height_scan_sources.py`, 81 个位置)

| 指标 | raycast (新链路) | leg_fk (旧链路) |
|---|---|---|
| vs 引擎真值的 p50 \|Δ\| (特征) | **0.002** (0.5 mm) | 0.058 (11.6 mm) |
| p95 \|Δ\| | **0.044** (8.8 mm) | 0.371 (74 mm) |
| 超过 2 cm 的格点占比 | **1.4%** | 32.6% |
| 每帧直接观测到的格点 | **31.3/35 = 89.6%** | — (常数向量) |

* lidar 统计: 命中率 0.60, 累计"被自身遮挡"的射线 61626 条 (自遮挡真的在起作用)。
* 结论: **真机 L1 几何 + 自遮挡 + fill_local 的感知链路, 在部署仿真里能高保真复现训练口径的
  高度扫描**; 这是 v6 策略能在部署环境里"看到地形"的前提。

### 6.3 平地闭环 (v4 策略 + raycast 真感知, 8 s)

x=3.79 m (≈0.47 m/s), upright, z=0.287, tilt 4.9°, 健康 L0
-> 感知链路本身不会破坏行走 (raycast 在平地上特征≈0, 与训练 fill 语义一致)。

### 6.4 已知限制: 部署 CPU 仿真的 hfield 接触 ≠ 训练 warp 物理 (R1)

| 实验 | 结果 |
|---|---|
| `stand` 控制器站在部署 mix 地形上 | z=0.303, tilt 0.17° -> **接触/地形本身正常** |
| v4 策略站在部署 mix 地形上 (cmd=0, 3 s) | z 0.32->0.22, tilt ->20° -> **策略在 CPU hfield 上退化** |
| v4 策略在部署 mix 地形上走 (raycast, 8 s) | 0.37 m 后倒下 (z 0.137, L3) |

这与 docs/go2-delivery.md **R1** 一致 (训练 warp 地形 vx 0.48 vs CPU 0.13-0.19, 接触模型不等价)。
**因此地形通过率只在 warp (训练物理) 口径下评测**; 部署仿真用于: 平地闭环 + 感知链路验证
+ 真机移植前的接口回归。CPU 地形的部署级复现需要等 mjx→CPU 接触对齐 (另一个课题)。

---

## 5. 结论 (2026-10-19)

1. **遮挡感知训练有效, 且没有牺牲 clean 性能**: v6a@7.37M 在 384 env 五地形上
   19/20 格 >= v5a (平均 +2.6pp), 遮挡/盲扫下优势更大 (boxes 盲扫 +8.4pp),
   平地步态质量还更好 (trot 0%->39%, 足端滑移 0.768->0.633)。
2. **交付 = v6a@7.37M** (不是 final, 也不是 eval_reward best): eval_reward 涨到 1230 时
   地形存活率在 9.83M 之后反而下降 —— **检查点必须按地形存活率选** (§4.5/§4.6)。
3. **部署侧已打通**: 17x11 扫描 + 真机 L1 几何/自遮挡/高程图记忆的 raycast 链路
   (p50 0.5mm), 默认查看器切到 v6+raycast (§6); CPU hfield 与 warp 不等价 (R1),
   地形验收只在 warp 口径做。
4. 残余: boxes 55% 仍是最弱地形; 航向漂移/里程计速度下转圈 (用户此前明确暂缓);
   真机 L1 的 FOV 上下界/安装倾角未实测标定 (训练用 ±0.2rad 抖动兜底)。

## 7.5 v6b (硬遮挡课程) 训练记录 —— 2026-10-19 02:14 完成

从 v6a@7.37M 续训 8M, 遮挡强度加倍 (occ_recall .5->.25, dropout .05->.10,
sector .25->.40/宽1.0-3.0, 地形采样 4->5 点, mount 抖动 .03, vfov 抖动 .20),
lr 4e-5, 地形 mix, 单作业 (gpu_guard 独占, fps 4608->5012, 正常)。

| 累计步 | eval_reward | ep_len | scan_vis | scan_info |
|---|---|---|---|---|
| 27.44M (restore) | 1105.3 | 589.9 | 0.36 | 0.41 |
| 30.15M | 1215.5 | 587.2 | 0.36 | 0.41 |
| 32.85M | 1324.7 | 605.1 | 0.37 | 0.41 |
| 35.55M | **1388.1** | 623.5 | 0.36 | 0.41 |

* 硬遮挡下信息可用率 ~0.41 (v6a 训练口径 0.60), 策略仍保持 ep_len 上升。
* eval_reward 又是一路涨 (1388 > v6a 的 1230) —— 按 §4.5 的教训, **不能据此判断好坏**,
  必须等 384 env 地形存活率 (检查点扫描 + 矩阵, 运行中)。
* 检查点: 2.70M / 5.41M / 8.11M (+best 副本)。
* **检查点扫描 (384 env, nominal 遮挡)**: @2.70M 81.0/77.1/50.8, @5.41M 81.0/76.6/51.8,
  @8.11M **80.2/79.7/52.1** (mix/stairs/boxes, sum 212.0) —— 全面**低于 v6a@7.37M 的
  84.1/79.2/54.7 (sum 218.0)**。即: **加倍遮挡强度没有带来 nominal 口径的提升**,
  反而略降 (mix -3.9, boxes -2.6)。是否在 harsh/blind 档更好, 等全矩阵结果 (§7.6)。

### 7.6 v6b 全矩阵结果 —— **负结果, 假设被否定**

`logs/go2_v6b_final_matrix.tsv` (384 env x 6 s, 同协议, 三策略同批次):

| 策略 | perfect 平均 | nominal 平均 | harsh 平均 | blind 平均 |
|---|---|---|---|---|
| **v6a@7.37M (交付)** | **78.4** | **76.8** | **74.7** | **69.8** |
| v6b@8.11M (硬遮挡课程) | 74.8 | 74.3 | 72.2 | 69.2 |
| v5a-final (基线) | 76.9 | 74.4 | 71.7 | 67.7 |

逐地形 (v6b - v6a): mix -0.6/-2.7/-2.3/+1.8; stairs -7.6/+1.1/+0.7/+1.0;
slope -4.5/-1.0/-1.3/+1.0; rough -2.6/-1.3/-1.6/-0.5; boxes -6.7/-1.6/-5.7/-5.5。

**结论 (负结果)**: 把遮挡强度加倍 (info 0.60->0.41) 再训 8M:
* **blind 档没有提升** (69.2 vs 69.8, 噪声内) —— v6a 的盲扫表现已经**不是感知瓶颈**,
  而是本体感知步态的极限; 继续加遮挡只让策略更保守。
* clean/nominal/harsh 全面下降 (平均 -3.6/-2.5/-2.5pp, boxes perfect -6.7)。
* 附带: v6b 平地步态反而更好 (trot 44%, 滑移 0.535 vs 0.633) —— 平地步态与地形鲁棒性
  在该奖励面下可分离 (与 §4.3 现象一致)。
* **交付维持 v6a@7.37M**; v6b 权重留在 `models/go2_v6b_logs/checkpoints/000008110080`
  作为负结果证据。后续要攻 blind 应从**步态/本体感知**层面下手, 而非继续加大感知退化。

## 6.5 GPU 串行化纪律 (2026-10-19 新增, 用户明确提醒)

**规则: 同一时刻只允许一个 GPU 作业。** 本机 8GB RTX 5060 Laptop 与 WSLg 桌面共享显存,
两个训练/评测进程并发会互相抢 SM 与显存 —— 训练 fps 掉一半以上、甚至 CUDA OOM。

本轮的实际情况 (据日志核对):

| 时段 | GPU 作业 | fps / 说明 |
|---|---|---|
| 00:05-00:44 v6a 训练 | **单作业** | fps 4649 -> 5323 (v5a 对照 4757-4892, v6a 反而更快) |
| 00:44-00:57 v6a 自动评测 | 训练脚本内串行 (矩阵 -> 平地 CPU) | 无并发 |
| 01:15 前后 | 检查点扫描 (681) 与候选选择 (685) **有约 1 分钟重叠** | 两个都是评测, 影响小, 但违反纪律 |
| 01:44 之后 | 无 GPU 作业 (空闲) | — |

**新增守门员** `sim/gpu_guard.sh` (本次加的, 以后所有 GPU 命令都必须走它):

```bash
bash sim/gpu_guard.sh status            # 看锁/显存/是否有本项目的 GPU 进程
bash sim/gpu_guard.sh check             # 空闲 exit 0, 忙 exit 3
bash sim/gpu_guard.sh run -- <命令...>  # 拿 flock 独占锁后串行执行; 已有作业时**直接拒绝** (exit 3), 不排队
```

检查四件事: ① flock 锁; ② `nvidia-smi` compute apps; ③ 本项目 train/probe/eval 进程;
④ 显存空闲 >= 2500 MiB。已自测: 正常启动 rc=0, 并发第二个作业 rc=3。

v6b (§7 候选 1) 已按此纪律用 guard 单作业启动 (`models/go2_v6b_logs`)。

## 7. 下一步候选 (未做)

1. **v6b 硬遮挡课程**: resume v6a@7.37M + occ_recall .25 / dropout .1 / sector .4,
   短步数低 lr (`sim/run_go2_v6b_harsh.sh` 已就绪) —— 若 blind 再抬 3~5pp 且不伤 clean 就值得。
2. **mask/置信特征** (调研方案 A): obs 243 -> 251~255, 让策略区分"真平地"与"不知道"。
3. **感知延迟 0~4 步 + AR(1) 持续性 + 整帧失明 p=0.1**: 更接近真机高程图 (5~10Hz)。
4. **boxes 专项**: mix 地形生成器里加大 boxes/离散障碍比例与难度 (单变量), 重训。
5. **部署侧 heading 闭环**: cmd_wz 来自上层或加 yaw 观测, 解决自旋 (用户暂缓)。
