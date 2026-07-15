import torch
from torch import Tensor, nn

from ligm.attention import all_local_attention
from ligm.masking import IGNORE_INDEX


def _true_log_probabilities(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    selected = labels != IGNORE_INDEX
    flat_labels = labels[selected]
    selected_logits = logits[selected] if logits.ndim == 3 else logits
    log_probs = selected_logits.float().log_softmax(dim=-1)
    true_log_probs = log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    return true_log_probs, true_log_probs.exp()


@torch.no_grad()
def information_gain_scores(
    teacher: nn.Module,
    candidate_input_ids: Tensor,
    attention_mask: Tensor,
    candidate_labels: Tensor,
    learnability: bool = True,
) -> Tensor:
    teacher.eval()
    global_output = teacher(
        input_ids=candidate_input_ids,
        attention_mask=attention_mask,
        labels=candidate_labels,
    )
    with all_local_attention(teacher):
        local_output = teacher(
            input_ids=candidate_input_ids,
            attention_mask=attention_mask,
            labels=candidate_labels,
        )
    global_log_p, global_p = _true_log_probabilities(global_output.logits, candidate_labels)
    local_log_p, _ = _true_log_probabilities(local_output.logits, candidate_labels)
    flat_scores = (global_log_p - local_log_p).clamp_min(0)
    if learnability:
        flat_scores = flat_scores * 4 * global_p * (1 - global_p)
    scores = torch.zeros_like(candidate_labels, dtype=torch.float32)
    scores[candidate_labels != IGNORE_INDEX] = flat_scores
    return scores


@torch.no_grad()
def entropy_scores(
    teacher: nn.Module,
    candidate_input_ids: Tensor,
    attention_mask: Tensor,
    candidate_labels: Tensor,
) -> Tensor:
    teacher.eval()
    output = teacher(
        input_ids=candidate_input_ids,
        attention_mask=attention_mask,
        labels=candidate_labels,
    )
    selected = candidate_labels != IGNORE_INDEX
    selected_logits = output.logits[selected] if output.logits.ndim == 3 else output.logits
    log_probabilities = selected_logits.float().log_softmax(dim=-1)
    flat_scores = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
    scores = torch.zeros_like(candidate_labels, dtype=torch.float32)
    scores[selected] = flat_scores
    return scores
