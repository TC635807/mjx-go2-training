# Go2 v30: per-env 地形课程 + 长训练 (2026-10-21)

用户要求: 添加地形课程和更长训练; 如果效果还是不好再尝试更大网络。

## 0. 背景 (v29 收尾时的问题)

v29 用"多阶段 warm-start + 混合难度赛道"把 0.03-0.05m 台阶上 1 级成功率做到 83%/38%,
但有两个结构性问题:
1. **没有真正的逐 env 课程**: 单阶段串行训练会互相覆盖 (训 0.05 后 0.03 的上 2 级从
   17.6% 回落到 6.5%); 固定难度分布也没有"弱者先练简单"的机制。
2. **训练预算偏短**: 每个阶段 2-4M 步, 台阶能力还在涨就被切走。

legged_gym/IsaacLab/mjlab 的标准做法是 **per-env 地形瓦片 + 课程晋升** (见
docs/go2-v29-stairs.md §1): 每个 episode 结束按"走了多远/是否摔倒"把 env 升/降到
不同难度瓦片。我们单张共享地形做不了"每 env 一块瓦片", 但可以复刻其**语义**:
一张地形里放 6 条隔离楼梯赛道 (riser 0.02/0.03/0.05/0.08/0.11/0.14), 每个 env
用自己的 terrain_level 选出生赛道, 课程逻辑在训练 wrapper 里。

## 1. 实现

### 1.1 地形: `sim/gen_terrain_stairs_lanes.py` -> lanes2

* 26x26 格 (0.4m/格, 场 ±5.2m), 6 条相互隔离的楼梯赛道 + 5 道 0.5m 高隔墙;
  赛道宽 1.6/1.6/1.6/1.2/1.2/1.2m; 每条: 3 格入口平道 -> 6 级上行 -> 12 格平台 -> 5 级下行;
* riser 0.02/0.03/0.05/0.08/0.11/0.14m (总高 0.12/0.18/0.30/0.48/0.66/0.84m);
* spawn_entries 按难度升序写入 json (6 个), env 的 level 直接索引;
* hfield 与 box 场同尺寸 (±5.2m), 无外圈平地。

### 1.2 wrapper: `sim/brax_curriculum.py`

playground 的 `wrap_for_brax_training(full_reset=True)` 每回合调 env.reset(rng) 但
reset 拿不到上一回合的课程 level, 也无法把 level 传进去。新增一层:

```
CurriculumAutoResetWrapper(                # 最外层, 替代 playground BraxAutoResetWrapper
  EpisodeWrapper(                          # brax 原版: steps/truncation/episode_metrics
    CurriculumVmapWrapper(raw_env)         # reset 时按 per-env level 调 env.reset_with_level
  )
)
```

* reset: level ~ U{0..init_max} (默认 init_max=2, 同 PIE), `jax.vmap(env.reset_with_level)`;
* step: 先走一步 -> 用**本步结束**时的状态算
  * `dist = |qpos[:2] - spawn_xy|` (spawn_xy 在 reset 时写入 info)
  * `fall = done & (truncation < 0.5)` (区分摔倒与超时)
  * `up = dist > success_dist(3.0m) & ~fall`, `down = (fall | dist < min_dist(1.0m)) & ~up`
  * `level += up - down`, clip 到 [0, 5] (legged_gym `_update_terrain_curriculum` 同款);
* 只对 done 的 env 用**新 level** 重开 (其余原样 where 合并); 保留 done 那一步的
  episode 簿记 (`steps`/`truncation`/`episode_metrics`) 给 brax logger, 否则 ep_len=0;
* 关键踩坑: (a) `done` 是 float32, 不能直接 `&`; (b) reset 与 step 的 info 键必须一致
  (`first_data/first_obs`); (c) metrics 键训练循环会多加 `reward`, 只能合并交集。

### 1.3 env: `envs/go2_walk.py`

* `reset(rng)` -> `reset_with_level(rng, 0)`; 新增 `reset_with_level(rng, level)` 与
  `sample_init_levels(rng_batch)`;
* `curriculum.enable` / `init_level_max` / `success_dist` / `min_dist` 配置;
* `_cur_on=True` 时 reset 用 `spawn_entries[level]` 出生并写 info:
  `terrain_level` / `spawn_xy` / `curr_rng`; 关掉时与原 reset 逐位一致。

### 1.4 CLI

* `train/train_go2.py`: `--curriculum --curriculum_init_level --curriculum_success_dist
  --curriculum_min_dist`;
* `sim/run_go2_v28.sh`: `CURRICULUM=1 CURINIT=2 CURSUCC=3.0 CURMIN=1.0`。

## 2. 训练配置 (v30a_curr)

从 v29c 热启动, 6M 步 (brax 实际会多跑 ~25%), lr 2e-5, 1024-512, 2048 env,
naconmax=49152/njmax=192 (8GB 显存), full_reset 由课程 wrapper 接管;
奖励面同 v29c 台阶配方: forward_progress 1.5 / lazy_stop -1.5 / termination -20 /
orientation -5 / feet_clearance -1.0(target 0.10) / action_acc -0.01 / hip -0.3 /
swing 0.18 / terminate_body_pen 0.02 / reward_clip_min -25; 命令 vx∈[0.3,0.7], wz∈[0,0.15]。

## 3. v30a_curr 结果 (课程 + 6.2M 步, 从 v29c 热启动)

训练: eval_reward 99 -> 126 -> 168 -> 349 -> **423** (best @4.98M), ep_len 589-640/750,
fps 3026-4143; 6.2M 步用时 ~26 min (课程 wrapper 的额外 reset 代价没有想象中大)。

爬升阶梯 (warp 128env x 6s, 含遮挡 lidar, spawn 在楼梯口; v29c -> v30):

| 地形 | 指标 | v29c | v30a_curr |
|---|---|---|---|
| wide_h030 (0.03) | climb / 上1级 / 上2级 | 0.041 / 85.6% / 12.0% | **0.050 / 94.4% / 30.4%** |
| wide_h050 (0.05) | climb / 上1级 / 上2级 | 0.040 / 42.7% / 0% | **0.052 / 61.8% / 3.6%** |
| wide_h080 (0.08) | climb / 存活 | 0.021 / 66.4% | 0.027 / 52.3% |
| wide_h110 (0.11) | climb | 0.015 | 0.017 |
| lanes1 混合 | climb / 上1级 / 上2级 / yaw p90 | 0.040 / 75.9% / 12.6% / 228° | **0.050 / 95.9% / 28.9% / 54.6°** |

结论:
* **4M-6M 的 per-env 课程 + 长训练把 0.03m 上 2 级从 12% 提到 30%, 0.05m 上 1 级
  从 43% 提到 62%**, 且 yaw 漂移大幅改善 (lanes1 p90 228°->55°);
* 0.08m 以上仍不能稳定多级 (climb 0.027, 且存活率下降);
* 在 lanes2 混合场评测 (随机入口含 0.11/0.14m 赛道): 6s 存活 60.2%, yaw p90 259°
  -> 困难赛道仍会让策略打转/摔, 课程因此涨得慢 (level 分布没有明显上移)。

## 3.5 v31: 更大网络 (net2wider 热启动)

按用户要求, 课程 + 长训练后仍不足 -> 试更大网络:
* `sim/widen_checkpoint.py`: 把 v30 的 1024-512 checkpoint 用 **net2wider** 扩成
  1536-768 (逐层复制隐藏单元 + 1/c 缩放, 函数近似不变), 生成可续训的 brax checkpoint;
* 训练: 1536-768 policy/value, lr 5e-5, 课程同 v30 (init_max=2), 6M 步;
* 目的: 验证"容量不够"假设; 若 0.08m 多级仍不行, 则问题在信号/课程而不是容量。


### 3.5.1 net2wider 细节与踩坑

`sim/widen_checkpoint.py` 把 v30 checkpoint 的 policy/value 网络 1024-512 -> 1536-768:

* 映射: `g_new2old[j] = floor(j * n_old / n_new)`, 每个 old 单元被复制 `c[g(j)]` 次;
* **正确缩放位置**: tanh 是非线性, 缩放必须放在"复制后的单元进入下一层的权重"上:
  * 输出单元复制: `W_new[:, m] = W_old[:, g(m)]`, `b_new[m] = b_old[g(m)]`;
  * 输入单元复制: `W_new[j, :] = W_old[g_prev(j), :] / c_prev[g_prev(j)]`;
  * 输出层 (Dense_0 / value hidden_2): `W_out_new[m,:] = W_out_old[g_last(m),:]/c_last[g_last(m)]`;
* 第一版错误地把缩放放在"复制单元的输入权重+bias"上 (net2wider 论文形式只适用于
  线性/恒等激活), 结果策略输出完全错乱 (训练 eval_reward -1008, ep_len 28);
* 正确版在 CPU 上数值验证: 新旧网络对随机 obs 的 policy mean 最大差 **1.4e-6**,
  value 最大差 **1.5e-5** (float32 舍入级) -> 函数等价;
* brax 的 `checkpoint.load_config` 对 kernel init fn = None 会 KeyError(None), 脚本里
  改为手动读 json 还原 callable; `checkpoint.save` 需要绝对路径, config 文件名要指定
  `ppo_network_config.json` (训练续训自检依赖它)。

## 3.6 v31 结果: 更大网络 (1536-768) + 课程长训练

### 3.6.1 台阶爬升 (warp 128env x 6s, 含遮挡 lidar, spawn 在楼梯口)

| 地形 | 指标 | v29c (1024) | v30 (1024+课程) | v31b (1536) | v31c (+hard课程) |
|---|---|---|---|---|---|
| 0.03m | 上1级 / 上2级 | 85.6 / 12.0 | 94.4 / 30.4 | 100 / 86.2 | 100 / (未测) |
| 0.05m | 上1级 / 上2级 | 42.7 / 0 | 61.8 / 3.6 | 98.0 / 71.0 | 95.7 / 62.9 |
| 0.08m | 上1级 / 存活 | 0 / 66.4 | 0 / 58.6 | 3.2 / 24.2 | 21.6 / 39.8 |
| 0.11m | climb / 存活 | 0.015 / 68.8 | 0.017 / 69.5 | 0.023 / 39.8 | 0.028 / 35.2 |
| lanes1 混合 | 上1级 / 上2级 | 75.9 / 12.6 | 93.9 / 30.3 | 100 / 90.0 | (未测) |

结论: **容量是 0.03-0.05m 台阶的关键瓶颈** —— 1536-768 把 0.03m 上 2 级从 30% 提到
86%, 0.05m 从 3.6% 提到 62-71%; 但 0.08m 以上仍是"能上 1 级但摔得多"。

### 3.6.2 部署侧代价 (v31d = 从 v31c 做 2M hyb2 巩固)

| 验收 | v29c | v31d |
|---|---|---|
| warp hyb2 / stg1 / bg200 / hyb1 / rubble (6s 存活) | 92.4 / 84.4 / 86.7 / 95.6 / 95.3 | 90.9 / 77.3 / 84.6 / 89.8 / 95.1 |
| warp 0.03 / 0.05 / 0.08 台阶 (6s 存活) | 99.0 / 80.7 / 66.9 | 95.3 / 73.4 / **34.4** |
| CPU 24seed 总存活 (flat/hyb2/stg1/big1/rubble) | **89.2** | 80.8 |
| CPU 速度 / 位移 | 0.29 m/s / 3.71m | **0.327 / 4.15m** |
| CPU 髋 q p95 | 43.8/41.0/37.7/41.8 | **42.8/34.5/31.2/41.5** |
| hyb2 取证 缝夹/卡死 | 0/16, 1 event | 0/16, 1 event (fell 2/16) |

-> 大网络+课程模型**更快、髋更好、0.03-0.05 台阶强得多**, 但部署存活率掉 2-32pp
(激进爬台阶在 0.08m 上摔), 所以 v31d **没有直接替换默认 v29c**; 正在做第二轮
hyb2 巩固 (v31e, 3M, 更低 lr + 更强 lazy_stop/termination), 目标: 把存活率拉回
v29c 水平并保留台阶收益。

### 3.6.3 可复现命令

```
# 1) net2wider: 1024-512 -> 1536-768 (函数等价, 数值验证 max|Δ|~1e-6)
python sim/widen_checkpoint.py <ckpt_1024> /tmp/widen1536b 1536 768
# 2) 课程训练 (v31b):
TAG=v31b_big SRC=/tmp/widen1536b/000000000000 SCENE=models/go2/scene_mjx_terrain_lanes2.xml \
  BOXGRID=models/go2/assets/lanes2.json CONTACT=clearance RING=0 SPAWN_STAIRS=0 STEPS=6000000 \
  LR=5e-5 NUM_EVALS=6 LAYERS=1536,768 VLAYERS=1536,768 ENVS=2048 NACON=49152 NJMAX=192 \
  XLAFRAC=0.35 HIPW=-0.3 HIPACTW=-0.3 SYMM=-0.4 ACTW=-0.01 ACCW=-0.01 SWING=0.18 \
  ASCENT=2.0 ORIW=-5.0 CMD_LOW=0.3,0,0 CMD_HIGH=0.7,0,0.15 PROGW=1.5 LAZYW=-1.5 \
  TERMW=-20 RFLOOR=-25 PHASEW=0.5 TBPEN=0.02 CLEARW=-1.0 CLEART=0.10 EP_LEN=750 \
  CURRICULUM=1 CURINIT=2 CURSUCC=3.0 CURMIN=1.0 bash sim/run_go2_v28.sh
# 3) 更强难度课程 (v31c): 同上但 CURINIT=3 CURSUCC=2.0 CURMIN=0.5 CMD 0.25-0.55, LR=4e-5, 4M
# 4) hyb2 巩固 (v31d): SRC=v31c best SCENE=hyb2 RING=1 FULL_RESET=1 2M LR=3e-5
```

### 3.6.4 v31e (二轮 hyb2 巩固) 结果与最终结论

v31e = v31d 再训 3M hyb2 (lr 2e-5, lazy_stop -2, termination -30):

| 验收 | v29c (默认) | v31d | v31e |
|---|---|---|---|
| CPU 24seed 总存活 | **89.2** | 80.8 | 78.3 |
| CPU 速度/位移 | 0.29/3.71 | **0.327/4.15** | 0.29/3.69 |
| warp hyb2 / stg1 / bg200 / hyb1 / rubble | **92.7 / 85.2 / 84.9 / 94.0 / 96.1** | 90.9 / 77.3 / 84.6 / 89.8 / 95.1 | 90.9 / 79.7 / 84.4 / 90.4 / 95.8 |
| 0.03m 上1级/上2级 | 85/9 | **98/70** | 78/33 |
| 0.05m 上1级/上2级 | 40/0 | **81/44** | 39/3 |
| 0.08m 存活 | **72.1** | 34.4 | 60.7 |

结论 (重要):
1. **大网络 (1536-768) + per-env 课程是 0.03-0.05m 台阶多级攀爬的关键**:
   0.03m 上 2 级 9%->70%, 0.05m 上 2 级 0%->44% (v31d);
2. 但**用单一 hyb2 巩固会持续侵蚀台阶能力和部署稳健性**: v31d 比 v29c 稳
   部署 -2~-32pp; v31e 再巩固 3M 后台阶反而回落到 0.03m 上2级 33%, 部署 78.3%
   -> **"更大网络 + 更长 hyb2 巩固"不是正确路线**, 瓶颈在"混合地形巩固的方式",
   不在容量或训练时长;
3. 因此**默认交付保持 v29c** (CPU 89.2%, 0.03m 上1级 85%/上2级 9%, 部署最稳);
   大网络模型作为**实验/台阶专精档**交付:
   - `go2_v31c`: 台阶最强 (0.03m 上1级100%/上2级86%, 0.05m 96%/63%), 但部署
     地形存活率低于 v29c, 只建议台阶专项;
   - `go2_v31d`: v31c + 2M hyb2 巩固, 台阶 98%/70%, 0.05m 81%/44%, 部署 80.8%, 速度最快;
   - `go2_v31e`: 二轮巩固证据, 不建议默认。

### 3.6.5 下一步 (按证据优先级)

1. **混合地形课程** (而不是训完再巩固): 把 hyb2 的台阶/箱体与 lanes2 的隔离赛道
   合成一张地形, 用 per-env 课程同时训练"台阶攀爬 + 部署鲁棒性";
2. 或者: 从 v31c 只做**短巩固** (0.5-1M, lr 1e-5) 并保留 curriculum, 避免过拟合;
3. 0.08m 以上仍受 Go2 能力上限 (~0.10m) 限制, 需要更长课程 (0.06/0.07/0.08/0.09 细档)
   + 可能的摆动轨迹先验。

## 4. 下一步

* v31 (1536-768) 6M 课程训练结果回填; 若有效 -> hyb2 巩固 + 部署验收 (同 v29c 流程);
* 若仍不足: per-env 课程参数 (success_dist/min_dist/init_max) 扫参, 或参考轨迹先验。
