import torch
from torch import nn

from ligm.scoring import _true_log_probabilities, entropy_scores


def test_true_log_probabilities_support_dense_logits():
    logits = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [0.0, 4.0]]])
    labels = torch.tensor([[-100, 0, 1]])
    log_probabilities, probabilities = _true_log_probabilities(logits, labels)

    expected = logits[0, 1:].log_softmax(-1)[torch.arange(2), torch.tensor([0, 1])]
    torch.testing.assert_close(log_probabilities, expected)
    torch.testing.assert_close(probabilities, expected.exp())


def test_true_log_probabilities_support_sparse_logits():
    logits = torch.tensor([[3.0, 1.0], [0.0, 4.0]])
    labels = torch.tensor([[-100, 0, 1]])
    log_probabilities, _ = _true_log_probabilities(logits, labels)

    expected = logits.log_softmax(-1)[torch.arange(2), torch.tensor([0, 1])]
    torch.testing.assert_close(log_probabilities, expected)


class FixedLogits(nn.Module):
    def forward(self, input_ids, attention_mask, labels):
        return type("Output", (), {"logits": input_ids.float()})


def test_entropy_scores_rank_uniform_distribution_above_peaked():
    logits = torch.tensor([[[0.0, 0.0], [8.0, -8.0]]])
    labels = torch.tensor([[0, 1]])
    scores = entropy_scores(FixedLogits(), logits, torch.ones((1, 2)), labels)

    assert scores[0, 0] > scores[0, 1]
