import torch

from ligm.masking import candidate_word_mask, random_word_mask, select_ligm_targets


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.arange(20).reshape(1, 20)
    word_ids = torch.tensor([[-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1]])
    return input_ids, word_ids


def test_random_mask_excludes_special_positions() -> None:
    input_ids, word_ids = _batch()
    generator = torch.Generator().manual_seed(11)
    batch = random_word_mask(input_ids, word_ids, 99, 100, generator)
    assert not batch.selected[0, 0]
    assert not batch.selected[0, -1]
    assert (
        torch.equal(batch.selected[0, 1:3], torch.tensor([True, True]))
        or not batch.selected[0, 1:3].any()
    )


def test_ligm_prefers_high_scoring_word_groups() -> None:
    input_ids, word_ids = _batch()
    generator = torch.Generator().manual_seed(11)
    candidates = candidate_word_mask(input_ids, word_ids, 99, generator)
    scores = torch.zeros_like(input_ids, dtype=torch.float32)
    scores[:, 10] = 100
    selected = select_ligm_targets(
        input_ids,
        word_ids,
        candidates.candidates,
        scores,
        99,
        100,
        generator,
    )
    if candidates.candidates[0, 10]:
        assert selected.selected[0, 10]
    assert not selected.selected[0, 0]
    assert not selected.selected[0, -1]
