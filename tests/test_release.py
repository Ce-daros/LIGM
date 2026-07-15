import json

from ligm.release import _model_card


def test_model_card_reports_failed_gate_without_claiming_improvement(tmp_path):
    run = tmp_path / "run"
    results = tmp_path / "results"
    run.mkdir()
    results.mkdir()
    (run / "metrics.jsonl").write_text(
        json.dumps(
            {
                "tokens_seen": 100,
                "peak_memory_gib": 7.0,
                "tokens_per_second": 10.0,
            }
        )
    )
    (run / "resolved-config.json").write_text(
        json.dumps({"training": {"seed": 11}})
    )
    (results / "ligm-synthetic.json").write_text(
        json.dumps(
            {
                "distance_buckets": [
                    {"bucket": "512-2048", "accuracy": 0.2, "mean_information_gain": 0.1}
                ]
            }
        )
    )
    (results / "ligm-natural.json").write_text(
        json.dumps({"buckets": {"local": {"accuracy": 0.5}, "long": {"accuracy": 0.2}}})
    )
    (results / "mechanism-gate.json").write_text(json.dumps({"passed": False}))

    card = _model_card(run, results, "owner/model")

    assert "gate **did not pass**" in card
    assert "A failed gate is a negative experimental" in card
    assert "result, not evidence of improved long-document retrieval" in card
