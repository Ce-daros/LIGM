import json

from ligm.gate import evaluate_gate


def test_mechanism_gate_applies_preregistered_thresholds(tmp_path):
    (tmp_path / "ligm-synthetic.json").write_text(
        json.dumps({"distance_information_gain_spearman": 0.3})
    )
    (tmp_path / "random-natural.json").write_text(
        json.dumps({"buckets": {"local": {"accuracy": 0.8}, "long": {"accuracy": 0.4}}})
    )
    (tmp_path / "ligm-natural.json").write_text(
        json.dumps({"buckets": {"local": {"accuracy": 0.797}, "long": {"accuracy": 0.43}}})
    )

    report = evaluate_gate(tmp_path, full=False)

    assert report["passed"]
