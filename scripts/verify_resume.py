import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path("/nvme-data/ligm")
REPO = ROOT / "repo"
FULL = ROOT / "runs/smoke-resume-full"
INTERRUPTED = ROOT / "runs/smoke-resume-interrupted"
CHECKPOINT = INTERRUPTED / "checkpoints/tokens-32768.pt"


def run(config: str) -> None:
    subprocess.run(
        [str(Path.home() / ".local/bin/uv"), "run", "--no-sync", "ligm-train", config],
        cwd=REPO,
        check=True,
    )


def metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    shutil.rmtree(FULL, ignore_errors=True)
    shutil.rmtree(INTERRUPTED, ignore_errors=True)
    run("configs/smoke-resume-full.yaml")

    process = subprocess.Popen(
        [
            str(Path.home() / ".local/bin/uv"),
            "run",
            "--no-sync",
            "ligm-train",
            "configs/smoke-resume-start.yaml",
        ],
        cwd=REPO,
        start_new_session=True,
    )
    while not CHECKPOINT.exists():
        time.sleep(0.05)
    os.killpg(process.pid, signal.SIGTERM)
    process.wait()
    run("configs/smoke-resume-continue.yaml")

    full_weights = load_file(FULL / "final/model.safetensors")
    resumed_weights = load_file(INTERRUPTED / "final/model.safetensors")
    for name in full_weights:
        torch.testing.assert_close(full_weights[name], resumed_weights[name], rtol=0, atol=0)

    full_metrics = metrics(FULL / "metrics.jsonl")
    resumed_metrics = metrics(INTERRUPTED / "metrics.jsonl")
    assert [item["loss"] for item in full_metrics] == [item["loss"] for item in resumed_metrics]
    print("resume_verification=exact")


if __name__ == "__main__":
    main()
