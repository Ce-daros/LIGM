import argparse
import json
from pathlib import Path

from ligm.online import build_online_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    report = build_online_curve(
        args.candidate_dir,
        args.reference_dir,
        args.samples,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
