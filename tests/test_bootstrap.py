import json

import pytest

from ligm.bootstrap import paired_stratified_bootstrap


def _write_scores(path, offset):
    path.write_text(
        json.dumps(
            {
                "per_query": [
                    {"query_id": f"q-{index}", "ndcg_at_10": index / 10 + offset}
                    for index in range(10)
                ]
            }
        )
    )


def test_paired_bootstrap_preserves_constant_system_difference(tmp_path):
    base = tmp_path / "base.json"
    random = tmp_path / "random.json"
    ligm = tmp_path / "ligm.json"
    _write_scores(base, 0.0)
    _write_scores(random, 0.1)
    _write_scores(ligm, 0.2)

    report = paired_stratified_bootstrap([base], [random], [ligm], samples=100)

    assert report["comparisons"]["ligm_minus_base"]["ci95"] == pytest.approx([0.2, 0.2])
    assert report["comparisons"]["ligm_minus_random"]["ci95"] == pytest.approx([0.1, 0.1])
