# Go2 交付说明 —— 能在地形上稳定行走的模型

> **最新交付 (2026-10-20 10:25): v25b@2.05M (1024-512 + 速度观测随机化 + 地形课程) —— 详见 [go2-v7-terrain.md](go2-v7-terrain.md)**
>
> * 243 维 obs = 48 本体 + **187 点高度扫描 (17x11)** + 8 步态相位; MLP **1024-512** (tanh);
>   训练链: v5a 20M (hfield) -> v6a 12M (雷达遮挡) -> v11 12M (**真竖面**) -> v13 20M (混合几何 hyb2)
>   -> v17 20M (512-256 热启动) -> v20 20M (1024-512 热启动) -> v24 (big1 大混合地形课程)
>   -> **v25b (速度观测随机化 + hyb2 巩固 2M)**。
> * 关键发现: ①**hfield 没有竖面** (0.094m 格把 0.1m 台阶抹成 ~50° 坡; 竖面版 33.3% vs
>   hfield 80.5%, 旧数字高估真实几何能力); ②**网络宽度直接换能力**: 128->512-256 竖面 +5.0pp,
>   512-256->1024-512 再 +3.6pp (均用"精确等价扩展"热启动); ③**速度观测随机化** (白噪声 +
>   每回合偏置 + OU 漂移 + 20ms 延迟) 是部署鲁棒性的关键 —— 未加随机化的 v24/v25a 仿真更好
>   但 CPU 部署闭环里会绕圈, 加了之后 v25b 两种速度源都稳。
> * v25b 512 env 6s 存活: **真竖面均值 80.0%** (方块 73.6 / 楼梯 70.9 / 混合 88.7 / 86.7),
>   **hfield 均值 78.9%** (mix 81.8 / stairs 74.6 / slope 76.0 / rough 83.6 / boxes 78.3),
>   **碎石 80.7%** (v20 73.8, +6.9); 平地 CPU **100%** (vx 0.488, 跟踪 0.092, 滑移 0.279, trot 71%)。
>   **官方 384 env 复核 (2026-10-20)**: 竖面均值 79.3 (方块 74.2 / 楼梯 70.8 / 混合 85.9/86.5),
>   hfield 均值 80.6 (mix 82.8 / stairs 75.8 / slope 77.9 / rough 85.4 / boxes 81.0), 碎石 80.5。
> * 部署侧: `policy_profiles.go2_v25b` + `view_go2_policy.sh` 默认已切到 v25b; 闭环 oracle
>   12.6m/20s、estimator 7.8m/20s, 均 upright ✓。
> * 交付文件: `policies/go2_walk_v25b.pkl` / 部署仓 `models/policies/go2_walk_v25b.npz`
>   (252KB, obs 243, layers [1024,512])。
> * 回滚: `PROFILE=go2_v20` / `go2_v24` / `go2_v25a` (仿真最强 82.4/80.1/83.8 但部署绕圈, 勿用) /
>   `go2_v17` / `go2_v13` / `go2_v6`。
> * 可复现: `sim/gen_terrain_big.py` (5x5 大混合地形)、`sim/eval_vertical_vs_hfield.sh`、
>   `sim/eval_rubble.sh`、`sim/warmstart_bignet.py`、`sim/probe_sweep_params.py`。