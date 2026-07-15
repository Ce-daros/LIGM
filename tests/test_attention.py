from ligm.attention import all_local_attention


class _Attention:
    def __init__(self, window):
        self.local_attention = window


class _Layer:
    def __init__(self, window):
        self.attn = _Attention(window)


class _Config:
    local_attention = 128


class _Backbone:
    layers = [_Layer((-1, -1)), _Layer((64, 64))]


class _Model:
    config = _Config()
    model = _Backbone()


def test_all_local_context_restores_layer_windows() -> None:
    model = _Model()
    original = [layer.attn.local_attention for layer in model.model.layers]
    with all_local_attention(model):
        assert [layer.attn.local_attention for layer in model.model.layers] == [
            (64, 64),
            (64, 64),
        ]
    assert [layer.attn.local_attention for layer in model.model.layers] == original
