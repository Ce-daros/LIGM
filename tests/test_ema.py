import torch
from torch import nn

from ligm.ema import EMATeacher


def test_ema_has_no_gradients_and_updates() -> None:
    student = nn.Linear(2, 2, bias=False)
    teacher = EMATeacher(student, decay=0.5)
    before = teacher.model.weight.detach().clone()
    with torch.no_grad():
        student.weight.add_(2)
    teacher.update(student)
    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())
    assert torch.allclose(teacher.model.weight, before + 1)
