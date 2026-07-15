import torch

from ligm.scoring import _true_log_probabilities


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
