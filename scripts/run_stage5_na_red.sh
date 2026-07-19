#!/usr/bin/env bash
set -euo pipefail

root=/nvme-data/ligm
cd "$root/repo"
export HF_HUB_OFFLINE=1
uv_bin="$HOME/.local/bin/uv"

"$uv_bin" run --no-sync ligm-train configs/smoke-stage5-na-red.yaml
"$uv_bin" run --no-sync ligm-train configs/stage5-na-red-10m.yaml
"$uv_bin" run --no-sync python -m ligm.online_report \
  "$root/runs/stage5-na-red-10m-seed11/online-evaluation" \
  "$root/runs/stage4-random-10m-seed11/online-evaluation" \
  --output "$root/runs/stage5-na-red-curve.json"
