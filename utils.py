import torch
import numpy as np
import os
import random

import logging
import yaml

from model.config import AttentionConfig, ViTConfig


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_tensorboard_writer(log_dir):
    """Set the TensorBoard writer to log info in `log_dir`.

    Example:
    ```
    writer = set_tensorboard_writer("logs")
    writer.add_scalar("Loss/train", loss.item(), step)
    ```

    Args:
        log_dir: (string) where to log
    """
    from torch.utils.tensorboard import SummaryWriter  # type: ignore

    writer = SummaryWriter(log_dir=log_dir)
    return writer


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def get_device():
    """
    Get the device to be used for training.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def config_to_dict(config):
    from dataclasses import asdict, is_dataclass

    def convert(value):
        if is_dataclass(value):
            return convert(asdict(value))
        elif isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        else:
            return value

    return convert(config)


def load_config(config_path):

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    model_cfg_dict = config_dict["model_config"]

    attention_cfg = AttentionConfig(**model_cfg_dict["attention_config"])
    model_cfg_dict.pop("attention_config")
    model_cfg = ViTConfig(**model_cfg_dict)
    model_cfg.attention_config = attention_cfg

    return model_cfg
