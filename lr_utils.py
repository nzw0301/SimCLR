import numpy as np
from omegaconf import OmegaConf


def calculate_initial_lr(cfg: OmegaConf) -> float:
    """
    When the size of mini-batches is smaller, squared learning rate is better.
    :param cfg:
    :return:
    """
    if cfg["parameter"]["linear_schedule"]:
        scaled_lr = cfg["experiment"]["lr"] * cfg["experiment"]["batches"] / 256.
    else:
        scaled_lr = cfg["experiment"]["lr"] * np.sqrt(cfg["experiment"]["batches"])
    return scaled_lr


def calculate_lr(cfg, warmup_steps: int, current_steps: int) -> float:
    initial_lr = calculate_initial_lr(cfg)

    if warmup_steps > 0.:
        learning_rate = current_steps / warmup_steps * initial_lr
    else:
        learning_rate = initial_lr

    return learning_rate
