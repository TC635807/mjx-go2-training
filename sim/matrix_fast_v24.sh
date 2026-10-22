#!/usr/bin/env bash
# v24 快速遮挡矩阵: 多策略 x 6 地形 x 4 遮挡档, 384 env x 6s。
# 用 probe_sweep_params.py (参数作为 jit 实参, 同地形只编译一次) -> 比逐个 probe 进程快 4-6 倍。
# 输出 TSV 与 sim/eval_occlusion_matrix.sh 兼容 (policy/terrain/cfg/surv/vx/z/scan_vis/scan_info),
# 因此可直接喂 sim/analyze_occlusion_matrix.py 与 sim/plot_occlusion_matrix.py。
#
# 用法: bash sim/matrix_fast_v24.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=${OUT:-logs/go2_v24_matrix.tsv}
ENVS=${ENVS:-384}
STEPS=${STEPS:-300}
POLS=${POLS:-"v6a_7.4M=models/go2_v6a_logs/checkpoints/000007372800,v7_6.1M=models/go2_v7_logs/checkpoints/000006144000,v7_8.2M=models/go2_v7_logs/checkpoints/000008192000"}
TR="mix=models/go2/scene_mjx_terrain_mix.xml,stairs=models/go2/scene_mjx_terrain_stairs_only.xml,slope=models/go2/scene_mjx_terrain_slope_only.xml,rough=models/go2/scene_mjx_terrain_rough_only.xml,boxes=models/go2/scene_mjx_terrain_boxes_only.xml,v7=models/go2/scene_mjx_terrain_v7.xml"
COMMON="--ckpts $POLS --terrains $TR --envs $ENVS --steps $STEPS --spawn random --cmd 0.5 0 0 --foot-contact hard --healthy_z_ref feet"

run_cfg() {
  local cfg="$1"; shift
  echo "===== cfg=$cfg ====="
  $PY sim/probe_sweep_params.py $COMMON "$@" --out "/tmp/go2_m24_$cfg.tsv" 2>&1 \
    | grep -E '^\\[sweep\\]|surv=|DONE|Traceback|Error'
}

run_cfg perfect
run_cfg nominal --lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 \
  --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25
run_cfg harsh --lidar --lidar_occ_recall 0.0 --lidar_dropout 0.20 --lidar_sector_prob 0.5
run_cfg blind --lidar --lidar_occ_recall 0.0 --lidar_dropout 1.0

echo "===== merge -> $OUT ====="
"$PY" - "$OUT" <<'PYEOF'
import os, sys
out = sys.argv[1]
cfgs = ("perfect", "nominal", "harsh", "blind")
rows = []
for cfg in cfgs:
    p = "/tmp/go2_m24_%s.tsv" % cfg
    if not os.path.isfile(p):
        print("!! missing", p); continue
    with open(p, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            pol, ter, surv, vx, z = parts[0], parts[1], parts[2], parts[3], parts[4]
            rows.append((pol, ter, cfg, surv, vx, z))
with open(out, "w", encoding="utf-8") as fh:
    fh.write("policy\tterrain\tcfg\tsurv\tvx\tz\tscan_vis\tscan_info\n")
    for pol, ter, cfg, surv, vx, z in rows:
        fh.write("%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\n" % (pol, ter, cfg, surv, vx, z))
print("wrote", out, len(rows), "rows")
PYEOF

echo "===== analyze ====="
"$PY" sim/analyze_occlusion_matrix.py "$OUT"
echo "===== plot ====="
"$PY" sim/plot_occlusion_matrix.py "$OUT" docs/img/go2-v24-occlusion.png || true
echo "DONE -> $OUT"
