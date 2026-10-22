#!/usr/bin/env bash
# GPU 串行化守门员 —— 同一时刻只允许 **一个** GPU 作业 (训练/评测/探针)。
#
# 背景: 本机是 8GB RTX 5060 Laptop 且与 WSLg 桌面共享显存。同时跑两个训练/评测进程时
# 两边都会被拖慢 (显存竞争 + SM 抢占), 实测训练 fps 会掉一半以上, 甚至 CUDA OOM。
# 本项目所有占 GPU 的命令都必须经过这里: 拿不到锁就**直接拒绝启动**, 而不是排队等
# 一个已经卡住的作业。
#
# 用法:
#   bash sim/gpu_guard.sh status            # 看当前是否有 GPU 作业 / 锁 / 显存
#   bash sim/gpu_guard.sh check             # 空闲 -> exit 0; 忙 -> exit 3
#   bash sim/gpu_guard.sh run -- <命令...>  # 加锁后串行执行 (锁随命令生命周期)
#
# 说明: 锁文件 /tmp/go2_gpu.lock 用 flock 持有, 进程异常退出时内核自动释放。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK=/tmp/go2_gpu.lock
MIN_FREE_MIB=${MIN_FREE_MIB:-2500}

gpu_busy_reasons() {
  local reasons=()
  # 1) flock 被占用 -> 有别的 gpu_guard 作业。
  #    注意: "run" 模式里本进程**已经持有**该锁, 不能再用 flock -n 探测自己
  #    (子 shell 新开 fd 会锁失败 -> 误判为"忙")。用 GPU_GUARD_HELD 标记跳过自检。
  if [ "${GPU_GUARD_HELD:-0}" != "1" ]; then
    if ! flock -n "$LOCK" true 2>/dev/null; then
      reasons+=("lock:$LOCK")
    fi
  fi
  # 2) nvidia-smi 上的 compute 进程 (桌面图形进程不算)
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null || true)
  if [ -n "$apps" ]; then
    reasons+=("compute_apps:$(echo "$apps" | tr '\n' ' ')")
  fi
  # 3) 本项目的训练/评测进程 (防止绕过 guard 直接起的作业)
  local pids
  pids=$(pgrep -af 'train[.]train_go2|probe_go2_warp_eval[.]py|eval_go2[.]py|final_matrix_v6[.]sh|diag_candidates[.]sh|diag_ckpt_sweep[.]sh' 2>/dev/null | grep -v gpu_guard | awk '{print $1}' | tr '\n' ' ')
  if [ -n "$pids" ]; then
    reasons+=("procs:$pids")
  fi
  printf '%s\n' "${reasons[@]:-}"
}

cmd=${1:-status}
case "$cmd" in
  status)
    echo "== GPU 状态 =="
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "显存空闲: ${free} MiB (要求 >= $MIN_FREE_MIB)"
    busy=$(gpu_busy_reasons)
    if [ -n "$busy" ]; then echo "忙: $busy"; else echo "空闲: 可以启动一个 GPU 作业"; fi
    ;;
  check)
    busy=$(gpu_busy_reasons)
    if [ -n "$busy" ]; then echo "[gpu_guard] 忙, 拒绝启动: $busy" >&2; exit 3; fi
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -lt "$MIN_FREE_MIB" ]; then
      echo "[gpu_guard] 显存空闲仅 ${free} MiB (< $MIN_FREE_MIB), 拒绝启动" >&2; exit 3
    fi
    echo "[gpu_guard] 空闲, 允许启动"
    ;;
  run)
    shift || true
    [ "${1:-}" = "--" ] && shift
    if [ "$#" -eq 0 ]; then echo "用法: gpu_guard.sh run -- <命令...>" >&2; exit 2; fi
    exec 9>"$LOCK"
    if ! flock -n 9; then
      echo "[gpu_guard] 已有 GPU 作业持有锁 $LOCK, 拒绝并发启动: $*" >&2
      exit 3
    fi
    export GPU_GUARD_HELD=1   # 已持锁: 后续忙检查跳过锁自检 (必须在忙检查之前)
    busy=$(gpu_busy_reasons)
    if [ -n "$busy" ]; then
      echo "[gpu_guard] 检测到 GPU 忙 ($busy), 拒绝并发启动: $*" >&2
      exit 3
    fi
    echo "[gpu_guard] $(date +%H:%M:%S) 获得 GPU 独占锁, 开始: $*"
    trap 'flock -u 9 2>/dev/null || true' EXIT
    "$@"
    rc=$?
    echo "[gpu_guard] $(date +%H:%M:%S) 作业结束 rc=$rc"
    exit "$rc"
    ;;
  *)
    echo "未知命令: $cmd (status|check|run)" >&2; exit 2 ;;
esac
