from collections import OrderedDict

import torch
from torchvision.models import resnet18, resnet50


class LinearClassifier(torch.nn.Module):
    def __init__(self, num_features: int = 128, num_classes: int = 10):
        """
        Linear classifier for linear evaluation protocol.

        :param num_features: The dimensionality of feature representation
        :param num_classes: The number of supervised class
        """

        super(LinearClassifier, self).__init__()
        self.classifier = torch.nn.Linear(num_features, num_classes)


    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return self.classifier(inputs)  # N x num_classes


class CentroidClassifier(torch.nn.Module):

    def __init__(self, weights: torch.FloatTensor):
        """
        :param weights: The pre-computed weights of the classifier.
        """
        super(CentroidClassifier, self).__init__()
        self.weights = weights  # d x num_classes

    def forward(self, inputs) -> torch.FloatTensor:
        return torch.matmul(inputs, self.weights)  # N x num_classes

    @staticmethod
    def create_weights(data_loader, num_classes: int) -> torch.FloatTensor:
        """
        :param data_loader: Data loader of feature representation to create weights.
        :param num_classes: The number of classes.
        :return: FloatTensor contains weights.
        """

        X = data_loader.data
        Y = data_loader.targets

        weights = []
        for k in range(num_classes):
            ids = torch.where(Y == k)[0]
            weights.append(torch.mean(X[ids], dim=0))

        weights = torch.stack(weights, dim=1)  # d x num_classes
        return weights


class ProjectionHead(torch.nn.Module):
    def __init__(self, num_last_hidden_units: int, d: int):
        """
        :param num_last_hidden_units: the dimensionality of the encoder's output representation.
        :param d: the dimensionality of output.

        """
        super(ProjectionHead, self).__init__()

        self.projection_head = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(num_last_hidden_units, num_last_hidden_units)),
            ('bn1', torch.nn.BatchNorm1d(num_last_hidden_units)),
            ('relu1', torch.nn.ReLU()),
            ('linear2', torch.nn.Linear(num_last_hidden_units, d, bias=False))
        ]))

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return self.projection_head(inputs)


class ContrastiveModel(torch.nn.Module):

    def __init__(self, base_cnn: str = "resnet18", d: int = 128, is_cifar: bool = True):
        """
        :param base_cnn: The backbone's model name. resnet18 or resnet50.
        :param d: The dimensionality of the output feature.
        :param is_cifar:
            model is for CIFAR10/100 or not.
            If it is `True`, network is modified by following SimCLR's experiments.
        """

        assert base_cnn in {"resnet18", "resnet50"}
        super(ContrastiveModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
            num_last_hidden_units = 2048
        elif base_cnn == "resnet18":
            self.f = resnet18()
            num_last_hidden_units = 512

            if is_cifar:
                # replace the first conv2d with smaller conv
                self.f.conv1 = torch.nn.Conv2d(
                    in_channels=3, out_channels=64, stride=1, kernel_size=3, padding=3, bias=False
                )

                # remove the first max pool
                self.f.maxpool = torch.nn.Identity()
        else:
            raise ValueError(
                "`base_cnn` must be either `resnet18` or `resnet50`. `{}` is unsupported.".format(base_cnn)
            )

        # drop the last classification layer
        self.f.fc = torch.nn.Identity()

        # non-linear projection head
        self.g = ProjectionHead(num_last_hidden_units, d)

    def encode(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        return features before projection head.
        :param inputs: FloatTensor that contains images.
        :return: feature representations.
        """

        return self.f(inputs)  # N x num_last_hidden_units

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        h = self.encode(inputs)
        z = self.g(h)
        return z  # N x d


class SupervisedModel(torch.nn.Module):

    def __init__(self, base_cnn: str = "resnet18", num_classes: int = 10, is_cifar: bool = True):
        """
        :param base_cnn: name of backbone model.
        :param num_classes: the number of supervised classes.
        :param is_cifar: Whether CIFAR10/100 or not.
        """

        assert base_cnn in {"resnet18", "resnet50"}
        super(SupervisedModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
            num_last_hidden_units = 2048
        elif base_cnn == "resnet18":
            self.f = resnet18()
            num_last_hidden_units = 512

            if is_cifar:
                # replace the first conv2d with smaller conv
                self.f.conv1 = torch.nn.Conv2d(
                    in_channels=3, out_channels=64, stride=1, kernel_size=3, padding=3, bias=False
                )

                # remove the first max pool
                self.f.maxpool = torch.nn.Identity()
        else:
            raise ValueError(
                "`base_cnn` must be either `resnet18` or `resnet50`. `{}` is unsupported.".format(base_cnn)
            )

        self.f.fc = torch.nn.Linear(num_last_hidden_units, num_classes)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        return self.f(inputs)
