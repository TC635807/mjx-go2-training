#!/usr/bin/env bash
# 有效出生 (valid spawn) 评测: 排除"出生就骑在箱棱/台阶棱上"的初始条件。
# 用法: bash sim/eval_spawn_valid.sh [out.tsv]
# 与 matrix_fast_v24.sh 同口径, 只是多传 --spawn_max_slope 0.25 (局部坡度 rise/run 上限)。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tc63/mujoco/.venv/bin/python
OUT=${1:-logs/go2_v24_validspawn.tsv}
POLS=${POLS:-"v6a_7.4M=models/go2_v6a_logs/checkpoints/000007372800,v8_best=models/go2_v8_logs/checkpoints/best"}
TR="mix=models/go2/scene_mjx_terrain_mix.xml,stairs=models/go2/scene_mjx_terrain_stairs_only.xml,slope=models/go2/scene_mjx_terrain_slope_only.xml,rough=models/go2/scene_mjx_terrain_rough_only.xml,boxes=models/go2/scene_mjx_terrain_boxes_only.xml"
for cfg in perfect nominal; do
  echo "===== cfg=$cfg (spawn_max_slope=0.25) ====="
  if [ "$cfg" = perfect ]; then
    EXTRA=""
  else
    EXTRA="--lidar --lidar_occ_recall 0.5 --lidar_dropout 0.05 --lidar_mount_jitter 0.02 --lidar_vfov_jitter 0.15 --lidar_sector_prob 0.25"
  fi
  $PY sim/probe_sweep_params.py \
    --ckpts "$POLS" --terrains "$TR" --envs 384 --steps 300 \
    --spawn random --spawn_max_slope 0.25 --cmd 0.5 0 0 --foot-contact hard \
    --healthy_z_ref feet $EXTRA --out "/tmp/go2_vs_$cfg.tsv" 2>&1 \
    | grep -E '^\\[sweep\\]|surv=|DONE|Traceback|Error'
done
$PY - "$OUT" <<'PYEOF'
import sys
out = sys.argv[1]
rows = []
for cfg in ("perfect", "nominal"):
    try:
        fh = open("/tmp/go2_vs_%s.tsv" % cfg, encoding="utf-8")
    except OSError:
        continue
    fh.readline()
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6:
            rows.append((p[0], p[1], cfg, p[2], p[3], p[4]))
with open(out, "w", encoding="utf-8") as fh:
    fh.write("policy\tterrain\tcfg\tsurv\tvx\tz\tscan_vis\tscan_info\n")
    for r in rows:
        fh.write("%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\n" % r)
print("wrote", out, len(rows))
PYEOF
echo "DONE -> $OUT"
