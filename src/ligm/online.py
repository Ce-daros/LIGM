import json
from pathlib import Path


def compare_local_recovery(
    candidate: dict,
    reference: dict,
    max_local_drop: float,
) -> dict:
    candidate_accuracy = candidate["buckets"]["local"]["accuracy"]
    reference_accuracy = reference["buckets"]["local"]["accuracy"]
    delta = candidate_accuracy - reference_accuracy
    return {
        "candidate_local_accuracy": candidate_accuracy,
        "reference_local_accuracy": reference_accuracy,
        "local_delta": delta,
        "threshold": -max_local_drop,
        "passed": candidate_accuracy >= reference_accuracy - max_local_drop,
    }


def read_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
