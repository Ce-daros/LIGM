from copy import deepcopy

import torch
from torch import nn


class EMATeacher:
    def __init__(self, student: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between zero and one")
        self.decay = decay
        self.model = deepcopy(student).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        student_parameters = dict(student.named_parameters())
        for name, teacher_parameter in self.model.named_parameters():
            teacher_parameter.lerp_(student_parameters[name].detach(), 1.0 - self.decay)

        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in self.model.named_buffers():
            teacher_buffer.copy_(student_buffers[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "model": self.model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.model.load_state_dict(state["model"])
