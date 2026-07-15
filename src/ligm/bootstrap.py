import argparse
import json
from pathlib import Path

import numpy as np


def _scores(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {item["query_id"]: item["ndcg_at_10"] for item in report["per_query"]}


def paired_stratified_bootstrap(
    base_paths: list[Path],
    random_paths: list[Path],
    ligm_paths: list[Path],
    samples: int = 10_000,
    seed: int = 11,
) -> dict:
    systems = {
        "base": [_scores(path) for path in base_paths],
        "random": [_scores(path) for path in random_paths],
        "ligm": [_scores(path) for path in ligm_paths],
    }
    query_ids = sorted(systems["base"][0])
    matrices = {
        name: np.array(
            [[run[query_id] for query_id in query_ids] for run in runs],
            dtype=np.float64,
        )
        for name, runs in systems.items()
    }
    generator = np.random.default_rng(seed)
    differences = {
        "ligm_minus_base": np.empty(samples),
        "ligm_minus_random": np.empty(samples),
    }
    for sample in range(samples):
        indices = generator.integers(
            0,
            len(query_ids),
            size=(len(base_paths), len(query_ids)),
        )
        means = {
            name: np.take_along_axis(matrix, indices, axis=1).mean()
            for name, matrix in matrices.items()
        }
        differences["ligm_minus_base"][sample] = means["ligm"] - means["base"]
        differences["ligm_minus_random"][sample] = means["ligm"] - means["random"]

    return {
        "samples": samples,
        "seed": seed,
        "queries": len(query_ids),
        "paired_seeds": len(base_paths),
        "comparisons": {
            name: {
                "mean": float(values.mean()),
                "ci95": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
            }
            for name, values in differences.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, action="append", required=True)
    parser.add_argument("--random", type=Path, action="append", required=True)
    parser.add_argument("--ligm", type=Path, action="append", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = paired_stratified_bootstrap(
        args.base,
        args.random,
        args.ligm,
        samples=args.samples,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
