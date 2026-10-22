#!/usr/bin/env bash
# Go2 续训 (从 brax/orbax 检查点恢复)。
#
# 为什么能续、以及要注意什么 (已在 brax 0.14.2 源码里核实):
#   brax/training/agents/ppo/train.py:725-732 的 restore 分支只做两件事:
#       normalizer_params <- params[0]
#       params(policy/value) <- params[1], params[2]
#   **优化器状态 (Adam 动量/二阶矩) 与 env_steps 都不在检查点里** -> 从零重建、
#   步数从 0 记。所以:
#     1) 必须降 lr (默认 3e-4 会带一个较大的重启瞬态)。本脚本默认 2e-4:
#        本次是"同一任务中途续训", 不是"收敛后微调", 所以不像 Go1 的 v22/v23
#        那样压到 1e-4 —— 那个档位是为了不把收敛好的策略推走。
#     2) 必须换 --logdir: brax 的 checkpoint.save 是 force 覆盖, 相对步数名
#        (2.5M/5M/...) 会重名, 不换 logdir 会**静默覆盖**上一段训练的证据。
#        train_go2.py 的 _check_ckpt_overwrite 会在启动时拦住你。
#     3) --step_offset 只影响日志里打印的"累计步数", 不改训练语义。
#
# 用法:
#   bash sim/resume_go2_train.sh <检查点目录> <本段步数> <新 logdir> [lr]
# 例 (从 2.75M 续到 40M, 走硬接触 v2 配方):
#   bash sim/resume_go2_train.sh \
#       models/go2_v2_hard_logs/checkpoints/000002752512 37250000 \
#       models/go2_v2b_logs 2e-4
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?需要检查点目录}"
STEPS="${2:?需要本段步数}"
LOGDIR="${3:?需要新的 logdir (不能与检查点所在的那次训练相同)}"
LR="${4:-2e-4}"

# orbax 要求路径**绝对** (train.py 里 checkpoint.load 会抛
# "Checkpoint path should be absolute"), logdir 同理。统一在这里转绝对路径。
CKPT="$(readlink -f "$CKPT")"
mkdir -p "$LOGDIR"
LOGDIR="$(readlink -f "$LOGDIR")"

if [ ! -f "$CKPT/ppo_network_config.json" ]; then
  echo "不是 brax 检查点目录 (缺 ppo_network_config.json): $CKPT" >&2
  exit 2
fi

# 从检查点目录名解析出已训步数 -> step_offset (仅日志用)
OFF="$(basename "$CKPT")"
OFF="${OFF#0}"; OFF="${OFF:-0}"
echo "续训: ckpt=$CKPT 已训=$OFF 本段=$STEPS 新 logdir=$LOGDIR lr=$LR"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 这块 8GB 卡是和主机桌面共享的: 0.60 档实测在 32.6M 步处被主机挤爆
# (warp 分配 55MB 失败 -> FFI callback error)。可用环境变量调低。
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.50}"
SB3_ENVS="${SB3_ENVS:-4096}"

exec /home/tc63/mujoco/.venv/bin/python -u -m train.train_go2 \
  --num_timesteps "$STEPS" --num_evals 16 \
  --sb3_full --sb3_envs "$SB3_ENVS" --hard_contact \
  --sample_cmd --fwd_turn --body_frame \
  --cmd_low "0,-0.4,-1.5" --cmd_high "1.0,0.4,1.5" \
  --tilt_deg 17.5 --natural_gait --dangle_penalty --layers 128,128 \
  --terrain --height_scan --push --obs_noise \
  --climb_reward --soft_landing --gait_phase --body_contact \
  --ascent --ascent_w 4.0 --ascent_gate 0.2 \
  --symmetry --symmetry_w -0.4 \
  --restore "$CKPT" --step_offset "$OFF" --lr "$LR" \
  --logdir "$LOGDIR" \
  --save_name go2_walk_policy_hard.pkl
