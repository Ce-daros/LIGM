from ligm.decision import exploratory_decision


def point(tokens: int, long_difference: float, local_passed: bool = True) -> dict:
    return {
        "tokens_seen": tokens,
        "long": {"absolute_difference": long_difference},
        "local_guard": {"passed": local_passed},
    }


def test_exploratory_decision_unlocks_only_at_positive_500m_point() -> None:
    decision = exploratory_decision(
        {"points": [point(100_000_000, -0.01), point(500_010_000, 0.002)]}
    )

    assert decision["decision_point"] == 500_010_000
    assert decision["unlock_mldr_probe"]
    assert decision["promote_seeds_22_33"]


def test_exploratory_decision_stays_locked_after_local_guard_failure() -> None:
    decision = exploratory_decision(
        {"points": [point(500_010_000, 0.002, local_passed=False)]}
    )

    assert not decision["unlock_mldr_probe"]
    assert not decision["promote_seeds_22_33"]


def test_exploratory_decision_stays_locked_before_500m() -> None:
    decision = exploratory_decision({"points": [point(499_999_999, 0.01)]})

    assert decision["decision_point"] is None
    assert not decision["unlock_entropy_ablation"]
