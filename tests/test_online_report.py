import pytest

from ligm.online import paired_document_bootstrap


def report(correct: list[int], counts: list[int]) -> dict:
    return {
        "document_results": [
            {
                "document_index": index,
                "buckets": {
                    "long": {"correct": item_correct, "count": item_count}
                },
            }
            for index, (item_correct, item_count) in enumerate(zip(correct, counts, strict=True))
        ]
    }


def test_paired_bootstrap_uses_document_level_counts() -> None:
    result = paired_document_bootstrap(
        report([8, 9], [10, 10]),
        report([7, 8], [10, 10]),
        "long",
        samples=100,
        seed=1,
    )

    assert result["absolute_difference"] == pytest.approx(0.1)
    assert result["confidence_interval_95"] == pytest.approx([0.1, 0.1])


def test_paired_bootstrap_rejects_different_masked_positions() -> None:
    with pytest.raises(ValueError, match="identical masked positions"):
        paired_document_bootstrap(
            report([8, 9], [10, 10]),
            report([7, 8], [9, 10]),
            "long",
        )
