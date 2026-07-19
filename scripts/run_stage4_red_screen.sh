#!/usr/bin/env bash
set -euo pipefail

root=/nvme-data/ligm
cd "$root/repo"
export HF_HUB_OFFLINE=1
uv_bin="$HOME/.local/bin/uv"

"$uv_bin" run --no-sync ligm-train configs/smoke-stage4-red.yaml
"$uv_bin" run --no-sync ligm-train configs/stage4-random-10m.yaml
"$uv_bin" run --no-sync ligm-train configs/stage4-red-full-10m.yaml
"$uv_bin" run --no-sync ligm-train configs/stage4-red-route-10m.yaml

for variant in full route; do
  "$uv_bin" run --no-sync python -m ligm.online_report \
    "$root/runs/stage4-red-${variant}-10m-seed11/online-evaluation" \
    "$root/runs/stage4-random-10m-seed11/online-evaluation" \
    --output "$root/runs/stage4-red-${variant}-curve.json"
done
