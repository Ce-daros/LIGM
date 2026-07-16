import json
from pathlib import Path

import numpy as np


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


def _document_arrays(report: dict, bucket: str) -> tuple[np.ndarray, np.ndarray]:
    documents = report["document_results"]
    indices = [document["document_index"] for document in documents]
    if indices != list(range(len(documents))):
        raise ValueError("Online reports must contain ordered document indices")
    correct = np.asarray(
        [document["buckets"][bucket]["correct"] for document in documents],
        dtype=np.float64,
    )
    counts = np.asarray(
        [document["buckets"][bucket]["count"] for document in documents],
        dtype=np.float64,
    )
    return correct, counts


def paired_document_bootstrap(
    candidate: dict,
    reference: dict,
    bucket: str,
    samples: int = 10_000,
    seed: int = 20260716,
) -> dict:
    candidate_correct, candidate_counts = _document_arrays(candidate, bucket)
    reference_correct, reference_counts = _document_arrays(reference, bucket)
    if candidate_correct.shape != reference_correct.shape:
        raise ValueError("Candidate and reference reports use different document counts")
    if not np.array_equal(candidate_counts, reference_counts):
        raise ValueError("Candidate and reference reports do not use identical masked positions")

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        candidate_correct.size,
        size=(samples, candidate_correct.size),
    )
    resampled_counts = reference_counts[indices].sum(axis=1)
    candidate_accuracy = candidate_correct[indices].sum(axis=1) / resampled_counts
    reference_accuracy = reference_correct[indices].sum(axis=1) / resampled_counts
    differences = candidate_accuracy - reference_accuracy
    point_difference = (
        candidate_correct.sum() / candidate_counts.sum()
        - reference_correct.sum() / reference_counts.sum()
    )
    return {
        "absolute_difference": point_difference,
        "confidence_interval_95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def build_online_curve(
    candidate_dir: str | Path,
    reference_dir: str | Path,
    samples: int = 10_000,
    seed: int = 20260716,
) -> dict:
    candidate_dir = Path(candidate_dir)
    reference_dir = Path(reference_dir)
    candidate_reports = {
        int(path.name.removeprefix("tokens-").removesuffix("-natural.json")): path
        for path in candidate_dir.glob("tokens-*-natural.json")
    }
    reference_reports = {
        int(path.name.removeprefix("tokens-").removesuffix("-natural.json")): path
        for path in reference_dir.glob("tokens-*-natural.json")
    }
    if not candidate_reports:
        raise ValueError("Candidate online-evaluation directory has no reports")
    if set(candidate_reports) - set(reference_reports):
        raise ValueError("Reference curve is missing candidate token points")

    points = []
    for token_count in sorted(candidate_reports):
        candidate = read_report(candidate_reports[token_count])
        reference = read_report(reference_reports[token_count])
        point = {
            "tokens_seen": token_count,
            "local": paired_document_bootstrap(
                candidate,
                reference,
                "local",
                samples,
                seed,
            ),
            "long": paired_document_bootstrap(
                candidate,
                reference,
                "long",
                samples,
                seed,
            ),
            "local_guard": candidate["local_guard"],
        }
        points.append(point)
    return {"points": points}
