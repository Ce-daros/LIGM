import argparse
import json
from pathlib import Path


def summarize(root: Path) -> dict:
    runs = []
    for metric_file in sorted(root.glob("**/metrics.jsonl")):
        metrics = [json.loads(line) for line in metric_file.read_text().splitlines() if line]
        if not metrics:
            continue
        runs.append(
            {
                "run": str(metric_file.parent),
                "steps": len(metrics),
                "tokens_seen": metrics[-1]["tokens_seen"],
                "final_loss": metrics[-1]["loss"],
                "mean_tokens_per_second": sum(item["tokens_per_second"] for item in metrics)
                / len(metrics),
                "peak_memory_gib": max(item["peak_memory_gib"] for item in metrics),
            }
        )
    return {"runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("report.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.root), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
