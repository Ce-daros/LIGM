import torch

from ligm.rotary import apply_rotary_unpadded_torch


def test_unpadded_rotary_resets_positions_and_backpropagates():
    sequence = torch.arange(2 * 3 * 2 * 6, dtype=torch.float32).reshape(2, 3, 2, 6)
    qkv = torch.cat((sequence, sequence)).requires_grad_()
    angles = torch.tensor([[0.0, 0.0], [0.25, 0.5]])
    output = apply_rotary_unpadded_torch(
        qkv,
        angles.cos(),
        angles.sin(),
        torch.tensor([0, 2, 4], dtype=torch.int32),
        2,
    )

    torch.testing.assert_close(output[:2], output[2:])
    torch.testing.assert_close(output[:, 2], qkv[:, 2])
    torch.testing.assert_close(output[..., 4:], qkv[..., 4:])
    output.sum().backward()
    assert qkv.grad is not None
