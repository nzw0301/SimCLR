import logging

import hydra
import numpy as np
import torch
import torchvision
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import SimCLRTransforms
from distributed_utils import init_ddp
from loss import NT_Xent
from lr_utils import calculate_lr, calculate_initial_lr
from model import ContrastiveModel


def exclude_from_wt_decay(named_params, weight_decay, skip_list=("bias", "bn")) -> list:
    # https://github.com/nzw0301/pytorch-lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L90-L105

    params = []
    excluded_params = []

    for name, param in named_params:

        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)

    return [
        {"params": params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0.}
    ]


def check_hydra_conf(cfg: OmegaConf) -> None:
    assert cfg["parameter"]["temperature"] > 0.
    assert cfg["parameter"]["epochs"] > 0
    assert cfg["experiment"]["batches"] > 0
    assert 1. > cfg["parameter"]["momentum"] >= 0
    assert cfg["parameter"]["warmup_epochs"] >= 0
    assert cfg["parameter"]["d"] > 0

    assert cfg["experiment"]["base_cnn"] in {"resnet18", "resnet50"}
    assert cfg["experiment"]["lr"] > 0.
    assert cfg["experiment"]["strength"] > 0.
    assert cfg["experiment"]["decay"] >= 0.


def validation(
        cfg: OmegaConf,
        model: ContrastiveModel,
        device: torch.device
):
    pass


def train(
        cfg: OmegaConf,
        training_data_loader: torch.utils.data.DataLoader,
        model: ContrastiveModel,
) -> None:
    """
    Training function
    :param cfg: Hydra's config instance
    :param training_data_loader: Training data loader for contrastive learning
    :param model: Contrastive model based on resnet
    :return: None
    """
    local_rank = cfg["distributed"]["local_rank"]
    num_gpus = cfg["distributed"]["world_size"]
    epochs = cfg["parameter"]["epochs"]
    num_training_samples = len(training_data_loader.dataset.data)
    steps_per_epoch = int(num_training_samples / (cfg["experiment"]["batches"] * num_gpus))  # because the drop=True
    total_steps = cfg["parameter"]["epochs"] * steps_per_epoch
    warmup_steps = cfg["parameter"]["warmup_epochs"] * steps_per_epoch
    current_step = 0

    model.train()
    nt_cross_entropy_loss = NT_Xent(temperature=cfg["parameter"]["temperature"], device=local_rank)

    optimizer = torch.optim.SGD(
        params=exclude_from_wt_decay(model.named_parameters(), weight_decay=cfg["experiment"]["decay"]),
        lr=calculate_initial_lr(cfg),
        momentum=cfg["parameter"]["momentum"],
        nesterov=False,
        weight_decay=0.
    )

    # https://github.com/google-research/simclr/blob/master/lars_optimizer.py#L26
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    cos_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optim,
        T_max=total_steps - warmup_steps,
    )

    for epoch in range(1, epochs + 1):
        training_data_loader.sampler.set_epoch(epoch)

        for (view0, view1), _ in training_data_loader:
            # adjust learning rate by applying linear warming
            if current_step <= warmup_steps:
                lr = calculate_lr(cfg, warmup_steps, current_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.zero_grad()
            z0 = model(view0.to(local_rank))
            z1 = model(view1.to(local_rank))
            loss = nt_cross_entropy_loss(z0, z1)
            loss.backward()
            optimizer.step()

            # adjust learning rate by applying cosine annealing
            if current_step > warmup_steps:
                cos_lr_scheduler.step()

            current_step += 1

        if local_rank == 0:
            logging.info("Epoch:{}/{} progress:{:.3f} loss:{:.3f}, lr:{:.7f}".format(
                epoch, epochs, epoch / epochs, loss.item(), optimizer.param_groups[0]["lr"]
            ))

            if epoch % cfg["experiment"]["save_model_epoch"] == 0:
                save_fname = "epoch={}-{}".format(epoch, cfg["experiment"]["output_model_name"])
                torch.save(model.state_dict(), save_fname)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    init_ddp(cfg)
    check_hydra_conf(cfg)

    seed = cfg["parameter"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank = cfg["distributed"]["local_rank"]
    logger.info("Using {}".format(rank))

    transform = SimCLRTransforms(strength=cfg["experiment"]["strength"])
    root = "~/pytorch_datasets"
    if cfg["experiment"]["name"] == "cifar10":
        training_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
    elif cfg["experiment"]["name"] == "cifar100":
        training_dataset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform
        )
    else:
        assert cfg["experiment"]["name"] in {"cifar10", "cifar100"}

    sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)
    training_data_loader = DataLoader(dataset=training_dataset, sampler=sampler,
                                      num_workers=cfg["parameter"]["num_workers"],
                                      batch_size=cfg["experiment"]["batches"], pin_memory=True, drop_last=True,
                                      )

    model = ContrastiveModel(base_cnn=cfg["experiment"]["base_cnn"], d=cfg["parameter"]["d"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train(cfg, training_data_loader, model)


if __name__ == "__main__":
    main()
