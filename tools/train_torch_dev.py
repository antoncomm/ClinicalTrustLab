"Run torch train with debugging."

import hydra
from omegaconf import OmegaConf
from ctl.utils.runners import get_runner_manual
from ctl.utils.config_class import Config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Config):
    """
    Main function that runs ctl.
    """

    # config_path = "configs/inbreast.yaml"
    cfg = OmegaConf.to_object(cfg)
    experiment_name = "test_run"
    verbose = True
    deterministic = False
    overwrite = True
    seed = 42
    device_name = "gpu"
    #cuda_visible_devices = "4"
    runner = get_runner_manual(
        cfg,
        experiment_name,
        verbose,
        deterministic,
        overwrite,
        seed,
        device_name,
        #cuda_visible_devices=cuda_visible_devices,
    )
    print(OmegaConf.to_yaml(cfg))  # should be after get_runner_manual
    runner.run()


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
