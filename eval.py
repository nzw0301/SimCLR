import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DownstreamDataset
from lr_utils import calculate_initial_lr
from model import CentroidClassifier, LinearClassifier, NonLinearClassifier
from model import ContrastiveModel


def check_hydra_conf(cfg: OmegaConf) -> None:
    assert cfg["parameter"]["epochs"] > 0
    assert cfg["experiment"]["batches"] > 0
    assert 1. > cfg["parameter"]["momentum"] > 0.
    assert cfg["parameter"]["warmup_epochs"] >= 0

    assert cfg["experiment"]["base_cnn"] in {"resnet18", "resnet50"}
    assert cfg["experiment"]["lr"] > 0.
    assert cfg["experiment"]["decay"] >= 0.


def convert_vectors(
        cfg: OmegaConf, data_loader, model: ContrastiveModel, device: torch.device
):
    """
    Convert experiment to feature representations.
    :param cfg: Hydra's config instance
    :param data_loader: Tata loader for raw experiment.
    :param model: Pre-trained instance
    :param device: PyTorch's device instance
    :return: Tuple of tensors: features and labels.
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

    return X, y


def centroid_eval(
        data_loader: DataLoader, device: torch.device, classifier: CentroidClassifier, top_k: int = 5
) -> tuple:
    """
    :param data_loader: DataLoader of downstream task.
    :param device: PyTorch's device instance
    :param classifier: Instance of CentroidClassifier
    :param top_k: The number of top-k to calculate accuracy.
    :return: Tuple of top-1 accuracy and top-k accuracy.
    """
    num_samples = len(data_loader.dataset)
    classifier.eval()
    top_1_correct = 0
    top_k_correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            y = y.to(device)
            pred_top_k = torch.topk(classifier(x.to(device)), dim=1, k=top_k)[1]
            pred_top_1 = pred_top_k[:, 0]

            top_1_correct += pred_top_1.eq(y.view_as(pred_top_1)).sum().item()
            if top_k > 1:
                top_k_correct += (pred_top_k == y.view(len(y), 1)).sum().item()

    return top_1_correct / num_samples, top_k_correct / num_samples


def learnable_eval(
        cfg: OmegaConf,
        classifier,
        training_data_loader: DataLoader,
        val_data_loader: DataLoader,
        device: torch.device
) -> tuple:
    """
    :param cfg: Hydra's config instance
    :param classifier: Instance of classifier. Either linear or nonlinear
    :param training_data_loader: Training data loader for a downstream task
    :param val_data_loader: Validation data loader for a downstream task
    :param device: PyTorch's device instance
    :return: tuple of train acc, train top-k acc, train loss, val acc, val top-k acc, and val loss.
    """

    def calculate_accuracies_loss(classifier, data_loader: DataLoader, device: torch.device, top_k: int = 5) -> tuple:
        """
        Auxiliary function to calculate accuracies and loss.
        :param classifier: Instance of classifier. Either linear or nonlinear
        :param data_loader: Data loader for a downstream task
        :param device: PyTorch's device instance
        :param top_k: The number of top-k to calculate accuracy. Note `top_k <= 1` is same to top1.
        :return: Tuple of top 1 acc, top k acc, and loss.
        """

        classifier.eval()
        total_loss = 0.
        top_1_correct = 0
        top_k_correct = 0
        num_samples = len(data_loader.dataset)

        with torch.no_grad():
            for x, y in data_loader:
                optimizer.zero_grad()
                y = y.to(device)
                outputs = classifier(x.to(device))
                total_loss += torch.nn.functional.cross_entropy(outputs, y, reduction="sum").item()

                pred_top_k = torch.topk(outputs, dim=1, k=top_k)[1]
                pred_top_1 = pred_top_k[:, 0]

                top_1_correct += pred_top_1.eq(y.view_as(pred_top_1)).sum().item()
                if top_k > 1:
                    top_k_correct += (pred_top_k == y.view(len(y), 1)).sum().item()
                else:
                    top_k_correct += top_1_correct

        return top_1_correct / num_samples, top_k_correct / num_samples, total_loss / num_samples

    epochs = cfg["parameter"]["epochs"]
    num_training_samples = len(training_data_loader.dataset)
    total_steps = cfg["parameter"]["epochs"] * int(np.ceil(num_training_samples / cfg["experiment"]["batches"]))

    classifier.train()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        params=classifier.parameters(),
        lr=calculate_initial_lr(cfg),
        momentum=cfg["parameter"]["momentum"],
        nesterov=True,
        weight_decay=cfg["experiment"]["decay"]
    )

    cos_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    train_accuracies = []
    train_top_k_accuracies = []
    val_accuracies = []
    val_top_k_accuracies = []
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        sum_loss = 0.
        for x, y in training_data_loader:
            optimizer.zero_grad()

            outputs = classifier(x.to(device))
            loss = cross_entropy_loss(outputs, y.to(device))

            loss.backward()
            optimizer.step()

            cos_lr_scheduler.step()
            sum_loss += loss.item() * len(y)

        average_loss = sum_loss / num_training_samples
        logging.info("Epoch:{}/{} progress:{:.3f} loss:{:.3f}, lr:{:.7f}".format(
            epoch, epochs, epoch / epochs, average_loss, optimizer.param_groups[0]["lr"]
        ))

        train_acc, train_top_k_acc, train_loss = calculate_accuracies_loss(classifier, training_data_loader, device)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        train_top_k_accuracies.append(train_top_k_acc)

        val_acc, val_top_k_acc, val_loss = calculate_accuracies_loss(classifier, val_data_loader, device)
        val_accuracies.append(val_acc)
        val_top_k_accuracies.append(val_top_k_acc)
        val_losses.append(val_loss)

    return train_accuracies, train_top_k_accuracies, train_losses, val_accuracies, val_top_k_accuracies, val_losses


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

    transform = transforms.Compose([transforms.ToTensor(), ])

    root = "~/pytorch_datasets"
    if cfg["experiment"]["name"] == "cifar10":
        training_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        )
        num_classes = 10
    elif cfg["experiment"]["name"] == "cifar100":
        training_dataset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform
        )
        num_classes = 100
    else:
        assert cfg["experiment"]["name"] in {"cifar10", "cifar100"}

    training_data_loader = DataLoader(
        dataset=training_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=True,
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
    )
    classification_results = {}

    top_k = cfg["parameter"]["top_k"]
    for weights_path in Path(cfg["experiment"]["target_dir"]).glob("*.pt"):
        key = str(weights_path).split("/")[-1]
        logger.info("Evaluation by using {}".format(key))

        model = ContrastiveModel(base_cnn=cfg["experiment"]["base_cnn"], d=cfg["parameter"]["d"])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)

        state_dict = torch.load(weights_path)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # load weights trained on self-supervised task
        if use_cuda:
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False, map_location=device)

        downstream_training_dataset = DownstreamDataset(*convert_vectors(cfg, training_data_loader, model, device))
        downstream_val_dataset = DownstreamDataset(*convert_vectors(cfg, val_data_loader, model, device))

        downstream_training_data_loader = DataLoader(
            dataset=downstream_training_dataset,
            batch_size=cfg["experiment"]["batches"],
            shuffle=True,
        )
        downstream_val_data_loader = DataLoader(
            dataset=downstream_val_dataset,
            batch_size=cfg["experiment"]["batches"],
            shuffle=False,
        )

        if cfg["parameter"]["classifier"] == "centroid":
            classifier = CentroidClassifier(
                weights=CentroidClassifier.create_weights(downstream_training_dataset, num_classes=num_classes).to(
                    device)
            )
            train_acc, train_top_k_acc = centroid_eval(downstream_training_data_loader, device, classifier, top_k)
            val_acc, val_top_k_acc = centroid_eval(downstream_val_data_loader, device, classifier, top_k)

            classification_results[key] = {
                "train_acc": train_acc,
                "train_top_{}_acc".format(top_k): train_top_k_acc,
                "val_acc": val_acc,
                "val_top_{}_acc".format(top_k): val_top_k_acc
            }
            logger.info("train acc: {}, val acc: {}".format(train_acc, val_acc))

        else:
            if cfg["parameter"]["use_full_encoder"]:
                num_last_units = model.g[-1].out_features
            else:
                num_last_units = model.g[0].in_features

            if cfg["parameter"]["classifier"] == "linear":
                classifier = LinearClassifier(num_last_units, num_classes).to(device)
            elif cfg["parameter"]["classifier"].replace("-", "") == "nonlinear":
                classifier = NonLinearClassifier(num_last_units, num_classes).to(device)

            train_accuracies, train_top_k_accuracies, train_losses, val_accuracies, val_top_k_accuracies, val_losses = \
                learnable_eval(cfg, classifier, downstream_training_data_loader, downstream_val_data_loader, device)

            classification_results[key] = {
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_top_{}_accuracies".format(top_k): train_top_k_accuracies,
                "val_top_{}_accuracies".format(top_k): val_top_k_accuracies,
                "lowest_val_loss": min(val_losses),
                "highest_val_acc": max(val_accuracies),
                "highest_val_top_k_acc": max(val_top_k_accuracies)
            }
            logger.info("train acc: {}, val acc: {}".format(max(train_accuracies), max(val_accuracies)))

    fname = cfg["parameter"]["classification_results_json_fname"]

    with open(fname, "w") as f:
        json.dump(classification_results, f)


if __name__ == "__main__":
    main()
