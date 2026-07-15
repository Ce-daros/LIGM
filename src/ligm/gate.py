import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_gain(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return float("inf") if candidate > 0 else 0.0
    return (candidate - baseline) / baseline


def evaluate_gate(root: Path, full: bool) -> dict:
    ligm_synthetic = _read(root / "ligm-synthetic.json")
    random_natural = _read(root / "random-natural.json")
    ligm_natural = _read(root / "ligm-natural.json")

    score_correlation = ligm_synthetic["distance_information_gain_spearman"]
    long_gain = _relative_gain(
        ligm_natural["buckets"]["long"]["accuracy"],
        random_natural["buckets"]["long"]["accuracy"],
    )
    local_change = (
        ligm_natural["buckets"]["local"]["accuracy"]
        - random_natural["buckets"]["local"]["accuracy"]
    )
    checks = {
        "distance_information_gain_spearman": {
            "value": score_correlation,
            "threshold": 0.2,
            "passed": score_correlation >= 0.2,
        },
        "long_recovery_relative_gain": {
            "value": long_gain,
            "threshold": 0.05,
            "passed": long_gain >= 0.05,
        },
        "local_recovery_absolute_change": {
            "value": local_change,
            "threshold": -0.005,
            "passed": local_change >= -0.005,
        },
    }
    if full:
        random_mldr = _read(root / "random-mldr.json")
        ligm_mldr = _read(root / "ligm-mldr.json")
        entropy_natural = _read(root / "entropy-natural.json")
        mldr_gain = ligm_mldr["ndcg_at_10"] - random_mldr["ndcg_at_10"]
        entropy_gap = (
            ligm_natural["buckets"]["long"]["accuracy"]
            - entropy_natural["buckets"]["long"]["accuracy"]
        )
        checks["mldr_ndcg_at_10_gain"] = {
            "value": mldr_gain,
            "threshold": 0.005,
            "passed": mldr_gain >= 0.005,
        }
        checks["long_recovery_vs_entropy"] = {
            "value": entropy_gap,
            "threshold": 0.0,
            "passed": entropy_gap >= 0.0,
        }
    return {
        "full_gate": full,
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    report = evaluate_gate(args.root, args.full)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
