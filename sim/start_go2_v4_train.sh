#!/usr/bin/env bash
# Go2 v4 训练 —— 与 **v2** 的差异: 地形从"单张旧图"换成"混合地形 (128^2, 9.4cm 格距,
# 起伏+台地+1 条楼梯)" 换成 "混合地形 (128^2, 9.4cm 格距, 3x3 patch,
# 台阶/斜坡/随机起伏/波/方块/Perlin)" = models/go2/scene_mjx_terrain_mix.xml。
#
# 依据: docs/go2-terrain-survey.md (mjlab / IsaacLab ROUGH_TERRAINS 配方)
# **不做**观速度估计误差 (v3 的 A/B 显示强误差注入会带来 +33% 超速与 2x 偏航漂移,
# 见 docs/go2-v3-obs-vel-error.md §7): v4 = v2 配方 + 混合地形, 单变量 = 地形。
# 分辨率: n=128 (9.4cm) —— warp 的 hfield 接触在更细网格上会超 50/对上上限
# (v4 第一次跑 n=256 时 5973 次溢出 -> 接触被丢弃, 训练已中止).
#
# 生成地形: /home/tc63/mujoco/.venv/bin/python sim/gen_terrain_go2_mix.py --n 256 --preset mix --seed 1
# 实测: 40M 步 ~2.6h; 3072 env + mem fraction 0.45 (同 v3)。
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-40000000}"
LOGDIR="${2:-models/go2_v4_logs}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42

exec /home/tc63/mujoco/.venv/bin/python -u -m train.train_go2 \
  --num_timesteps "$STEPS" --num_evals 16 \
  --sb3_full --sb3_envs 2560 --hard_contact \
  --sample_cmd --fwd_turn --body_frame \
  --cmd_low "0,-0.4,-1.5" --cmd_high "1.0,0.4,1.5" \
  --tilt_deg 17.5 --natural_gait --dangle_penalty --layers 128,128 \
  --terrain --terrain_xml models/go2/scene_mjx_terrain_mix.xml \
  --height_scan --push --obs_noise \
  --climb_reward --soft_landing --gait_phase --body_contact \
  --ascent --ascent_w 4.0 --ascent_gate 0.2 \
  --symmetry --symmetry_w -0.4 \
  --logdir "$LOGDIR" \
  --save_name go2_walk_v4.pkl
