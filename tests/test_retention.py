import pytest
import torch

from ligm.retention import (
    asymmetric_retention_loss,
    project_conflicting_gradients,
    true_class_margins,
)


def test_true_class_margins_use_largest_competing_logit():
    logits = torch.tensor([[1.0, 4.0, 2.0], [3.0, 2.0, 1.0]])
    labels = torch.tensor([1, 2])

    margins = true_class_margins(logits, labels)

    assert margins.tolist() == pytest.approx([2.0, -2.0])


def test_asymmetric_retention_penalizes_margin_regression_only():
    labels = torch.tensor([0])
    anchor = torch.tensor([2.0])
    improved = torch.tensor([[3.0, 0.0]])
    regressed = torch.tensor([[1.0, 0.0]])

    improved_loss, improved_risk, _ = asymmetric_retention_loss(
        improved, labels, anchor, 0.25, 0.125
    )
    regressed_loss, regressed_risk, _ = asymmetric_retention_loss(
        regressed, labels, anchor, 0.25, 0.125
    )

    assert improved_loss < regressed_loss
    assert improved_risk < regressed_risk


def test_projection_is_applied_per_block():
    remote = [torch.tensor([1.0]), torch.tensor([1.0])]
    retention = [torch.tensor([-1.0]), torch.tensor([2.0])]

    projected, negative_blocks, _ = project_conflicting_gradients(
        remote, retention, ["first", "second"]
    )

    assert projected[0].item() == pytest.approx(0.0)
    assert projected[1].item() == pytest.approx(1.0)
    assert negative_blocks == 1
