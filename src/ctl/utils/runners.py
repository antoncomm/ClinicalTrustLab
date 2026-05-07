"""Setup runners"""

import os
import datetime


import torch

# ctl imports
from ctl.utils.configs import setup_config, setup_engine

# ctl runners
from ctl import runners


def _setup_runner(config):
    if config["run"]["cuda_visible_devices"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["run"]["cuda_visible_devices"]
    if config["run"]["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # without that raise error
        torch.use_deterministic_algorithms(config["run"]["deterministic"])
    project_dir = os.path.join(
        config["run"]["save_dir"], config["run"]["experiment_name"]
    )
    engine = setup_engine(
        config["run"]["engine_name"],
        tracker=config["run"]["tracker"],
        project_dir=project_dir,
    )
    if config["attack"] and 'attack_runner' in config["attack"]:
        config["model"]["runner"] = config["attack"]["attack_runner"]
    runner = getattr(runners, config["model"]["runner"])(engine=engine, config=config)
    return runner


def get_runner(args):
    config = setup_config(args.config)
    config["run"]["experiment_name"] = args.name
    config["run"]["verbose"] = args.verbose
    config["run"]["overwrite"] = args.overwrite
    config["run"]["deterministic"] = args.deterministic
    config["run"]["seed"] = args.seed
    config["run"]["checkpoints_dir"] = "checkpoints"
    config["run"]["project_dir"] = "logs"
    config["run"]["tracker"] = args.tracker
    config["run"]["cuda_visible_devices"] = args.cuda
    config["run"]["engine_name"] = args.engine

    return _setup_runner(config)


def get_runner_manual(
    config,
    experiment_name=datetime.datetime.now(),
    verbose=True,
    deterministic=False,
    overwrite=False,
    seed=42,
    engine_name="cpu",
    cuda_visible_devices=None,
    tracker="tensorboard",
):
    # config = setup_config(config_path)
    config["run"]["experiment_name"] = experiment_name
    config["run"]["verbose"] = verbose
    config["run"]["checkpoints_dir"] = "checkpoints"
    config["run"]["project_dir"] = "logs"
    config["run"]["deterministic"] = deterministic
    config["run"]["overwrite"] = overwrite
    config["run"]["seed"] = seed
    config["run"]["engine_name"] = engine_name
    config["run"]["cuda_visible_devices"] = cuda_visible_devices
    config["run"]["tracker"] = tracker

    return _setup_runner(config)
