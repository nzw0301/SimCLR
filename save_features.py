import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision
from dataset import SimCLRTransforms
from model import ContrastiveModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms


def check_hydra_conf(cfg: OmegaConf) -> None:
    assert cfg["parameter"]["epochs"] > 0
    assert cfg["experiment"]["base_cnn"] in {"resnet18", "resnet50"}


def convert_vectors(
        cfg: OmegaConf, data_loader, model: ContrastiveModel, device: torch.device
):
    """
    Convert experiment to feature representations.
    :param cfg: Hydra's config instance
    :param data_loader: Tata loader for raw experiment.
    :param model: Pre-trained instance
    :param device: PyTorch's device instance
    :return: Tuple of numpy array and labels.
    """
    model.eval()
    new_X = []
    new_y = []
    with torch.no_grad():
        for x_batches, y_batches in data_loader:
            if cfg["parameter"]["use_full_encoder"]:
                fs = model(x_batches.to(device))
            else:
                fs = model.encode(x_batches.to(device))

            new_X.append(fs)
            new_y.append(y_batches)

    X = torch.cat(new_X).cpu()
    y = torch.cat(new_y).cpu()

    return X.numpy(), y.numpy()


def convert_vectors_for_contrastive(
        cfg: OmegaConf, data_loader, model: ContrastiveModel, device: torch.device
):
    """
    Convert experiment to feature representations.
    :param cfg: Hydra's config instance
    :param data_loader: Tata loader for raw experiment.
    :param model: Pre-trained instance
    :param device: PyTorch's device instance
    :return: Tuple of numpy array and labels.
    """
    model.eval()
    new_X = []
    new_y = []
    with torch.no_grad():
        for (view0, _), y_batches in data_loader:
            if cfg["parameter"]["use_full_encoder"]:
                fs = model(view0.to(device))
            else:
                fs = model.encode(view0.to(device))

            new_X.append(fs)
            new_y.append(y_batches)

    X = torch.cat(new_X).cpu()
    y = torch.cat(new_y).cpu()

    return X.numpy(), y.numpy()


def get_data_loaders(cfg: OmegaConf, is_train: bool = True) -> tuple:

    if is_train:
        transform = SimCLRTransforms(strength=0.5)
    else:
        transform = transforms.Compose([transforms.ToTensor(), ])

    root = "~/pytorch_datasets"
    if cfg["experiment"]["name"] == "cifar10":
        training_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        )
    elif cfg["experiment"]["name"] == "cifar100":
        training_dataset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform
        )
    else:
        assert cfg["experiment"]["name"] in {"cifar10", "cifar100"}

    training_data_loader = DataLoader(
        dataset=training_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
    )

    return training_data_loader, val_data_loader


@hydra.main(config_path="conf", config_name="eval")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    check_hydra_conf(cfg)

    seed = cfg["parameter"]["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = cfg["parameter"]["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("Using {}".format(device))

    for weights_path in Path(cfg["experiment"]["target_dir"]).glob("*.pt"):
        key = str(weights_path).split("/")[-1]
        logger.info("Save features extracted by using {}".format(key))

        model = ContrastiveModel(base_cnn=cfg["experiment"]["base_cnn"], d=cfg["parameter"]["d"]).to(device)

        # load weights trained on self-supervised task
        if use_cuda:
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path, map_location=device))

        # no data-augmentation
        training_data_loader, val_data_loader = get_data_loaders(cfg, False)
        X_train, y_train = convert_vectors(cfg, training_data_loader, model, device)
        X_val, y_val = convert_vectors(cfg, val_data_loader, model, device)

        fname = "{}.feature.train.npy".format(key)
        np.save(fname, X_train)
        fname = "{}.label.train.npy".format(key)
        np.save(fname, y_train)
        fname = "{}.feature.val.npy".format(key)
        np.save(fname, X_val)
        fname = "{}.label.val.npy".format(key)
        np.save(fname, y_val)

        # average of data augmentation
        size_of_iterations = (1, 5, 20)
        training_data_loader, val_data_loader = get_data_loaders(cfg, True)

        X_trains = []
        X_vals = []
        for t in range(1, size_of_iterations[-1]+1):
            X_trains.append(convert_vectors_for_contrastive(cfg, training_data_loader, model, device)[0])
            X_vals.append(convert_vectors_for_contrastive(cfg, val_data_loader, model, device)[0])

            if t in size_of_iterations:
                fname = "{}.aug-{}.feature.train.npy".format(key, t)
                np.save(fname, np.array(X_trains).mean(axis=0))
                fname = "{}.aug-{}.feature.val.npy".format(key, t)
                np.save(fname, np.array(X_vals).mean(axis=0))


if __name__ == "__main__":
    main()
