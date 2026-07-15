from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def all_local_attention(model) -> Iterator[None]:
    """Restrict every layer to the local window without replacing its RoPE."""
    local_window = (model.config.local_attention // 2, model.config.local_attention // 2)
    original_windows = [layer.attn.local_attention for layer in model.model.layers]
    for layer in model.model.layers:
        layer.attn.local_attention = local_window
    try:
        yield
    finally:
        for layer, original_window in zip(model.model.layers, original_windows, strict=True):
            layer.attn.local_attention = original_window
