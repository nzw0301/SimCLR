import logging
import os

import hydra
import numpy as np
import torch
import torchvision
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import create_simclr_data_augmentation
from distributed_utils import init_ddp
from lr_utils import calculate_lr, calculate_initial_lr
from model import SupervisedModel


def check_hydra_conf(cfg: OmegaConf) -> None:
    assert cfg["parameter"]["epochs"] > 0
    assert cfg["experiment"]["batches"] > 0
    assert 1. > cfg["parameter"]["momentum"] >= 0
    assert cfg["parameter"]["warmup_epochs"] >= 0

    assert cfg["experiment"]["base_cnn"] in {"resnet18", "resnet50"}
    assert cfg["experiment"]["lr"] > 0.
    assert cfg["experiment"]["strength"] > 0.
    assert cfg["experiment"]["decay"] >= 0.


def validation(
        validation_data_loader: torch.utils.data.DataLoader,
        model: SupervisedModel,
        local_rank: int
) -> tuple:
    """
    :param validation_data_loader: Validation data loader
    :param model: ResNet based classifier.
    :param local_rank: local rank.
    :return: validation loss, the number of corrected samples, and the size of samples on a local
    """

    model.eval()

    sum_loss = torch.tensor([0.]).to(local_rank)
    num_corrects = torch.tensor([0.]).to(local_rank)

    with torch.no_grad():
        for data, targets in validation_data_loader:
            data, targets = data.to(local_rank), targets.to(local_rank)
            unnormalized_features = model(data)
            loss = torch.nn.functional.cross_entropy(unnormalized_features, targets, reduction="sum")

            predicted = torch.max(unnormalized_features.data, 1)[1]

            sum_loss += loss.item()
            num_corrects += (predicted == targets).sum()

    return sum_loss, num_corrects


def learning(
        cfg: OmegaConf,
        training_data_loader: torch.utils.data.DataLoader,
        validation_data_loader: torch.utils.data.DataLoader,
        model: SupervisedModel,
) -> None:
    """
    Learning function including evaluation

    :param cfg: Hydra's config instance
    :param training_data_loader: Training data loader
    :param validation_data_loader: Validation data loader
    :param model: Model
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

    best_metric = np.finfo(np.float64).max

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=calculate_initial_lr(cfg),
        momentum=cfg["parameter"]["momentum"],
        nesterov=False,
        weight_decay=cfg["experiment"]["decay"]
    )

    # https://github.com/google-research/simclr/blob/master/lars_optimizer.py#L26
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    cos_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optim,
        T_max=total_steps - warmup_steps,
    )

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        training_data_loader.sampler.set_epoch(epoch)

        for data, targets in training_data_loader:
            # adjust learning rate by applying linear warming
            if current_step <= warmup_steps:
                lr = calculate_lr(cfg, warmup_steps, current_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.zero_grad()
            data, targets = data.to(local_rank), targets.to(local_rank)
            unnormalized_features = model(data)
            loss = torch.nn.functional.cross_entropy(unnormalized_features, targets)
            loss.backward()
            optimizer.step()

            # adjust learning rate by applying cosine annealing
            if current_step > warmup_steps:
                cos_lr_scheduler.step()

            current_step += 1

        if local_rank == 0:
            logger_line = "Epoch:{}/{} progress:{:.3f} loss:{:.3f}, lr:{:.7f}".format(
                epoch, epochs, epoch / epochs, loss.item(), optimizer.param_groups[0]["lr"]
            )

        # During warmup phase, we skip validation
        sum_val_loss, num_val_corrects = validation(validation_data_loader, model, local_rank)

        torch.distributed.barrier()
        torch.distributed.reduce(sum_val_loss, dst=0)
        torch.distributed.reduce(num_val_corrects, dst=0)

        num_val_samples = len(validation_data_loader.dataset)

        # logging and save checkpoint
        if local_rank == 0:

            validation_loss = sum_val_loss.item() / num_val_samples
            validation_acc = num_val_corrects.item() / num_val_samples

            logging.info(logger_line + " val loss:{:.3f}, val acc:{:.2f}%".format(validation_loss, validation_acc * 100.))

            if cfg["parameter"]["metric"] == "loss":
                metric = validation_loss
            else:
                metric = 1. - validation_acc

            if metric <= best_metric:
                if "save_fname" in locals():
                    if os.path.exists(save_fname):
                        os.remove(save_fname)

                save_fname = "epoch={}-{}".format(epoch, cfg["experiment"]["output_model_name"])
                torch.save(model.state_dict(), save_fname)


@hydra.main(config_path="conf", config_name="supervised_config")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    check_hydra_conf(cfg)
    init_ddp(cfg)

    # fix seed
    seed = cfg["parameter"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank = cfg["distributed"]["local_rank"]
    logger.info("Using {}".format(rank))

    root = "~/pytorch_datasets"
    if cfg["experiment"]["name"].lower() == "cifar10":
        transform = create_simclr_data_augmentation(cfg["experiment"]["strength"], size=32)
        training_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        validation_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
        )
        num_classes = 10
    elif cfg["experiment"]["name"].lower() == "cifar100":
        transform = create_simclr_data_augmentation(cfg["experiment"]["strength"], size=32)
        training_dataset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform
        )
        validation_dataset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
        )
        num_classes = 100
    else:
        assert cfg["experiment"]["name"].lower() in {"cifar10", "cifar100"}

    sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)
    training_data_loader = DataLoader(dataset=training_dataset, sampler=sampler,
                                      num_workers=cfg["parameter"]["num_workers"],
                                      batch_size=cfg["experiment"]["batches"], pin_memory=True, drop_last=True,
                                      )

    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset, shuffle=False)
    validation_data_loader = DataLoader(dataset=validation_dataset, sampler=validation_sampler,
                                        num_workers=cfg["parameter"]["num_workers"],
                                        batch_size=cfg["experiment"]["batches"], pin_memory=True, drop_last=False,
                                        )

    model = SupervisedModel(base_cnn=cfg["experiment"]["base_cnn"], num_classes=num_classes)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    learning(cfg, training_data_loader, validation_data_loader, model)


if __name__ == "__main__":
    """
    To run this code,
    `python launch.py --nproc_per_node={The number of GPUs on a single machine} supervised.py`
    """
    main()
