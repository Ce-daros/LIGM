import torch


def apply_rotary_unpadded_torch(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    if cu_seqlens is None or max_seqlen is None:
        raise ValueError("ModernBERT unpadded rotary requires cu_seqlens and max_seqlen")

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    starts = torch.repeat_interleave(
        cu_seqlens[:-1],
        lengths,
        output_size=qkv.shape[0],
    )
    positions = torch.arange(qkv.shape[0], device=qkv.device) - starts
    rotary_half = cos.shape[-1]
    rotary_dim = rotary_half * 2

    qk = qkv[:, :2]
    qk_float = qk[..., :rotary_dim].float()
    first = qk_float[..., :rotary_half]
    second = qk_float[..., rotary_half:]
    cos_position = cos[positions].float()[:, None, None, :]
    sin_position = sin[positions].float()[:, None, None, :]
    rotated = torch.cat(
        (
            first * cos_position - second * sin_position,
            first * sin_position + second * cos_position,
        ),
        dim=-1,
    ).to(qkv.dtype)
    qk_output = torch.cat((rotated, qk[..., rotary_dim:]), dim=-1)
    return torch.cat((qk_output, qkv[:, 2:]), dim=1)


def use_torch_rotary() -> None:
    from transformers.models.modernbert import modeling_modernbert

    modeling_modernbert.apply_rotary_unpadded = apply_rotary_unpadded_torch
