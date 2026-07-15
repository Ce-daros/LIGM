import torch

from ligm.evaluate import _repetition_buckets


def test_repetition_buckets_separate_local_and_long_evidence():
    input_ids = torch.tensor([11, 12, 11, *range(13, 523), 12])
    attention_mask = torch.ones_like(input_ids)
    word_ids = torch.arange(input_ids.numel())
    selected = torch.zeros_like(input_ids, dtype=torch.bool)
    selected[2] = True
    selected[-1] = True

    assert _repetition_buckets(input_ids, attention_mask, word_ids, selected) == [
        (2, "local"),
        (input_ids.numel() - 1, "long"),
    ]
