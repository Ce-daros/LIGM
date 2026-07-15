import torch
from torch.optim import SGD

from ligm.config import TrainingConfig
from ligm.train import create_scheduler, rebase_scheduler, set_seed


def test_mask_sampling_does_not_change_data_crops() -> None:
    random_run = set_seed(11)
    ligm_run = set_seed(11)

    torch.rand(100, generator=ligm_run["mask"])

    assert torch.equal(
        torch.rand(10, generator=random_run["data"]),
        torch.rand(10, generator=ligm_run["data"]),
    )


def test_scheduler_rebase_immediately_sets_extended_schedule_rate() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=2e-5)
    scheduler = create_scheduler(optimizer, 100, TrainingConfig())
    optimizer.param_groups[0]["lr"] = 0.0

    rebase_scheduler(scheduler, 50)

    assert optimizer.param_groups[0]["lr"] == 2e-5
