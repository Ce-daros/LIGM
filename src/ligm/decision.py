import argparse
import json
from pathlib import Path


def exploratory_decision(curve: dict) -> dict:
    point_500m = next(
        (point for point in curve["points"] if point["tokens_seen"] >= 500_000_000),
        None,
    )
    reached_500m = point_500m is not None
    local_guard_passed = bool(
        point_500m and point_500m["local_guard"]["passed"]
    )
    long_advantage = bool(
        point_500m and point_500m["long"]["absolute_difference"] > 0.0
    )
    unlocked = reached_500m and local_guard_passed and long_advantage
    return {
        "decision_point": point_500m["tokens_seen"] if point_500m else None,
        "reached_500m": reached_500m,
        "local_guard_passed": local_guard_passed,
        "long_recovery_advantage": long_advantage,
        "unlock_mldr_probe": unlocked,
        "promote_seeds_22_33": unlocked,
        "unlock_entropy_ablation": unlocked,
        "unlock_gain_only_ablation": unlocked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("curve", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curve = json.loads(args.curve.read_text(encoding="utf-8"))
    decision = exploratory_decision(curve)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
