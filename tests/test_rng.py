import torch

from ligm.train import set_seed


def test_mask_sampling_does_not_change_data_crops() -> None:
    random_run = set_seed(11)
    ligm_run = set_seed(11)

    torch.rand(100, generator=ligm_run["mask"])

    assert torch.equal(
        torch.rand(10, generator=random_run["data"]),
        torch.rand(10, generator=ligm_run["data"]),
    )
