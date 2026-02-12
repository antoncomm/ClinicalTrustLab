"""Calculate mean and standard deviation"""

import os
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf

# torch imports
import torch
from torch.utils.data import DataLoader

# neurone imports
from neurone.utils.configs import setup_engine
from neurone.train.procedures import setup_data
from neurone.utils.config_class import Config


def get_mean_std_for_dataloader(dataloader, engine):
    """Compute the mean and std value of a PyTorch dataloader.
    Examples:
        total_mean, total_std = get_mean_std_for_dataloader(dataloader, engine)

        # output
        print('mean = ' + str(total_mean))
        print('std = ' + str(total_std))
    """
    # placeholders
    psum = torch.zeros(3, device=engine.device)
    psum_sq = torch.zeros(3, device=engine.device)
    count = 0

    # loop through images
    for inputs in (pbar := tqdm(dataloader)):
        pbar.set_description("Calculate mean and std")
        inputs = inputs[0].to(device=engine.device)
        batch_size = inputs.shape[0]
        count += batch_size * inputs.shape[2] * inputs.shape[3]
        psum += inputs.sum(dim=(0, 2, 3))
        psum_sq += (inputs**2).sum(dim=(0, 2, 3))

    # mean and std
    total_mean = psum / count
    total_var = psum_sq / count - total_mean**2
    total_std = torch.sqrt(total_var)

    return total_mean, total_std


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Config):
    """
    Example:
        python tools/get_mean_std.py 'run.cuda_visible_devices="<cuda_number>"' run.engine_name=<engine_name>
    """

    if cfg["run"]["cuda_visible_devices"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["run"]["cuda_visible_devices"]

    print(OmegaConf.to_yaml(cfg))

    cfg = OmegaConf.to_object(cfg)
    datasets = setup_data(cfg)

    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=cfg["run"]["batch_size"],
        num_workers=cfg["run"]["workers"],
        shuffle=False,
        collate_fn=datasets["train"].collate_fn,
    )
    test_dataloader = DataLoader(
        datasets["test"],
        batch_size=cfg["run"]["batch_size"],
        num_workers=cfg["run"]["workers"],
        shuffle=False,
        collate_fn=datasets["test"].collate_fn,
    )

    engine = setup_engine(cfg["run"]["engine_name"])

    # train
    mean, std = get_mean_std_for_dataloader(train_dataloader, engine)
    print(f"Train\n{mean = }\t{std = }")
    print()

    # test
    mean, std = get_mean_std_for_dataloader(test_dataloader, engine)
    print(f"Test\n{mean = }\t{std = }")
    print()


if __name__ == "__main__":
    main()
