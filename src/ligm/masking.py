from dataclasses import dataclass

import torch
from torch import Tensor

IGNORE_INDEX = -100


@dataclass(frozen=True)
class MaskedBatch:
    input_ids: Tensor
    labels: Tensor
    selected: Tensor
    candidates: Tensor | None = None
    scores: Tensor | None = None
    loss_weights: Tensor | None = None


def _groups(word_ids: Tensor) -> list[Tensor]:
    valid = word_ids[word_ids >= 0].unique(sorted=True)
    return [torch.where(word_ids == word_id)[0] for word_id in valid]


def _sample_groups(
    groups: list[Tensor], ratio: float, generator: torch.Generator
) -> tuple[list[Tensor], list[Tensor]]:
    target = round(sum(group.numel() for group in groups) * ratio)
    order = torch.randperm(len(groups), generator=generator).tolist()
    chosen: list[Tensor] = []
    remaining: list[Tensor] = []
    count = 0
    for index in order:
        group = groups[index]
        if count < target:
            chosen.append(group)
            count += group.numel()
        else:
            remaining.append(group)
    return chosen, remaining


def _mask_from_groups(shape: torch.Size, grouped_positions: list[list[Tensor]], device) -> Tensor:
    selected = torch.zeros(shape, dtype=torch.bool, device=device)
    for batch_index, groups in enumerate(grouped_positions):
        for positions in groups:
            selected[batch_index, positions.to(device)] = True
    return selected


def corrupt(
    input_ids: Tensor,
    selected: Tensor,
    mask_token_id: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    labels = input_ids.masked_fill(~selected, IGNORE_INDEX)
    corrupted = input_ids.clone()
    draws = torch.rand(input_ids.shape, generator=generator).to(input_ids.device)
    replace_mask = selected & (draws < 0.8)
    replace_random = selected & (draws >= 0.8) & (draws < 0.9)
    corrupted[replace_mask] = mask_token_id
    random_tokens = torch.randint(vocab_size, input_ids.shape, generator=generator).to(
        input_ids.device
    )
    corrupted[replace_random] = random_tokens[replace_random]
    return corrupted, labels


def random_word_mask(
    input_ids: Tensor,
    word_ids: Tensor,
    mask_token_id: int,
    vocab_size: int,
    generator: torch.Generator,
    ratio: float = 0.30,
) -> MaskedBatch:
    chosen = [_sample_groups(_groups(row), ratio, generator)[0] for row in word_ids.cpu()]
    selected = _mask_from_groups(input_ids.shape, chosen, input_ids.device)
    corrupted, labels = corrupt(input_ids, selected, mask_token_id, vocab_size, generator)
    return MaskedBatch(corrupted, labels, selected)


def candidate_word_mask(
    input_ids: Tensor,
    word_ids: Tensor,
    mask_token_id: int,
    generator: torch.Generator,
    ratio: float = 0.40,
) -> MaskedBatch:
    chosen = [_sample_groups(_groups(row), ratio, generator)[0] for row in word_ids.cpu()]
    candidates = _mask_from_groups(input_ids.shape, chosen, input_ids.device)
    labels = input_ids.masked_fill(~candidates, IGNORE_INDEX)
    corrupted = input_ids.masked_fill(candidates, mask_token_id)
    return MaskedBatch(corrupted, labels, candidates, candidates=candidates)


def select_ligm_targets(
    input_ids: Tensor,
    word_ids: Tensor,
    candidates: Tensor,
    scores: Tensor,
    mask_token_id: int,
    vocab_size: int,
    generator: torch.Generator,
    target_ratio: float = 0.20,
    replay_ratio: float = 0.10,
) -> MaskedBatch:
    selected_groups: list[list[Tensor]] = []
    for batch_index, row_word_ids in enumerate(word_ids.cpu()):
        groups = [
            group
            for group in _groups(row_word_ids)
            if candidates[batch_index, group.to(candidates.device)].all()
        ]
        scored = sorted(
            groups,
            key=lambda group: float(scores[batch_index, group.to(scores.device)].mean()),
            reverse=True,
        )
        eligible_count = int((row_word_ids >= 0).sum())
        target_count = round(eligible_count * target_ratio)
        targeted: list[Tensor] = []
        count = 0
        while scored and count < target_count:
            group = scored.pop(0)
            targeted.append(group)
            count += group.numel()

        replay_target = round(eligible_count * replay_ratio)
        replay: list[Tensor] = []
        count = 0
        if scored:
            order = torch.randperm(len(scored), generator=generator).tolist()
            for index in order:
                if count >= replay_target:
                    break
                replay.append(scored[index])
                count += scored[index].numel()
        selected_groups.append(targeted + replay)

    selected = _mask_from_groups(input_ids.shape, selected_groups, input_ids.device)
    corrupted, labels = corrupt(input_ids, selected, mask_token_id, vocab_size, generator)
    return MaskedBatch(corrupted, labels, selected, candidates=candidates, scores=scores)


def weight_ligm_targets(
    masked: MaskedBatch,
    word_ids: Tensor,
    scores: Tensor,
    target_ratio: float = 0.20,
    target_weight: float = 4.0,
) -> MaskedBatch:
    loss_weights = masked.selected.float()
    for batch_index, row_word_ids in enumerate(word_ids.cpu()):
        groups = [
            group
            for group in _groups(row_word_ids)
            if masked.selected[batch_index, group.to(masked.selected.device)].all()
        ]
        groups.sort(
            key=lambda group: float(scores[batch_index, group.to(scores.device)].mean()),
            reverse=True,
        )
        target_count = round(int((row_word_ids >= 0).sum()) * target_ratio)
        count = 0
        for group in groups:
            if count >= target_count:
                break
            loss_weights[batch_index, group.to(loss_weights.device)] = target_weight
            count += group.numel()

    return MaskedBatch(
        masked.input_ids,
        masked.labels,
        masked.selected,
        scores=scores,
        loss_weights=loss_weights,
    )
