import torch

from ligm.evaluate import _repetition_buckets, _single_token_markers


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


class Tokenized:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class MarkerTokenizer:
    ids = {
        " cobalt": [11],
        " amber": [12, 13],
        " violet": [14],
        " silver": [15],
        " crimson": [16, 17],
        " indigo": [18, 19],
        " coral": [20],
    }

    def __call__(self, text: str, *, add_special_tokens: bool) -> Tokenized:
        assert not add_special_tokens
        return Tokenized(self.ids[text])


def test_single_token_markers_use_sentence_initial_space_tokens() -> None:
    assert _single_token_markers(MarkerTokenizer(), 4) == [
        ("cobalt", 11),
        ("violet", 14),
        ("silver", 15),
        ("coral", 20),
    ]
