#!/usr/bin/env bash
set -euo pipefail

root=/nvme-data/ligm
repo="$root/repo"
random_run="$root/runs/stage1-random-seed11"
selection="$random_run/online-evaluation/selection.json"

while [[ ! -f "$selection" ]]; do
  if ! pgrep -f 'ligm-train .*random.*\.yaml' >/dev/null; then
    echo "Random reference run exited without a selection report" >&2
    exit 1
  fi
  sleep 60
done

python3 - "$selection" <<'PY'
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if selection["early_stopped"]:
    raise SystemExit("Random reference run unexpectedly reported early stop")
if selection["stopped_at_tokens"] < 1_000_000_000:
    raise SystemExit("Random reference run did not reach 1B tokens")
PY

cd "$repo"
export HF_HUB_OFFLINE=1
"$HOME/.local/bin/uv" run --no-sync ligm-train configs/stage2-online-ligm.yaml
"$HOME/.local/bin/uv" run --no-sync python -m ligm.online_report \
  "$root/runs/stage1-ligm-seed11/online-evaluation" \
  "$root/runs/stage1-random-seed11/online-evaluation" \
  --output "$root/runs/stage2-online-curve.json"
bash scripts/evaluate_stage2_milestones.sh
"$HOME/.local/bin/uv" run --no-sync python -m ligm.decision \
  "$root/runs/stage2-online-curve.json" \
  --output "$root/runs/stage2-decision.json"
