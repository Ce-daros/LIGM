from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def true_class_margins(logits: Tensor, labels: Tensor) -> Tensor:
    values = logits.float()
    true_values = values.gather(1, labels.unsqueeze(1)).squeeze(1)
    top_values, top_indices = values.topk(2, dim=-1)
    other_values = torch.where(
        top_indices[:, 0] == labels,
        top_values[:, 1],
        top_values[:, 0],
    )
    return true_values - other_values


def asymmetric_retention_loss(
    logits: Tensor,
    labels: Tensor,
    anchor_margins: Tensor,
    allowance: float,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    margins = true_class_margins(logits, labels)
    normalized_shortfall = (anchor_margins - allowance - margins) / temperature
    loss = temperature * F.softplus(normalized_shortfall).mean()
    risk = normalized_shortfall.sigmoid().mean()
    flip_rate = (margins < 0).float().mean()
    return loss, risk, flip_rate


def project_conflicting_gradients(
    remote_gradients: Sequence[Tensor],
    retention_gradients: Sequence[Tensor],
    block_ids: Sequence[str],
) -> tuple[list[Tensor], int, Tensor]:
    if not (
        len(remote_gradients) == len(retention_gradients) == len(block_ids)
    ):
        raise ValueError("Gradient and block collections must have equal lengths")

    dots: dict[str, Tensor] = {}
    retention_norms: dict[str, Tensor] = {}
    for remote, retention, block in zip(
        remote_gradients, retention_gradients, block_ids, strict=True
    ):
        dot = (remote.float() * retention.float()).sum()
        norm = retention.float().square().sum()
        dots[block] = dots.get(block, torch.zeros_like(dot)) + dot
        retention_norms[block] = retention_norms.get(
            block, torch.zeros_like(norm)
        ) + norm

    coefficients = {
        block: dot.clamp_max(0) / (retention_norms[block] + 1e-12)
        for block, dot in dots.items()
    }
    projected = [
        remote - coefficients[block].to(remote) * retention
        for remote, retention, block in zip(
            remote_gradients, retention_gradients, block_ids, strict=True
        )
    ]
    negative_blocks = sum(float(dot) < 0 for dot in dots.values())
    total_dot = sum(dots.values(), torch.zeros_like(next(iter(dots.values()))))
    remote_norm = torch.sqrt(
        sum(
            gradient.float().square().sum() for gradient in remote_gradients
        )
        + 1e-12
    )
    retention_norm = torch.sqrt(
        sum(
            gradient.float().square().sum() for gradient in retention_gradients
        )
        + 1e-12
    )
    cosine = total_dot / (remote_norm * retention_norm)
    return projected, negative_blocks, cosine
