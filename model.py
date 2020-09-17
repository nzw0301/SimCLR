import torch
import torchvision
from torchvision.models import resnet18, resnet50


class NonLinearClassifier(torch.nn.Module):
    def __init__(self, num_features: int = 128, num_hidden: int = 128, num_classes: int = 10):
        super(NonLinearClassifier, self).__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_hidden),
            torch.nn.BatchNorm1d(num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_classes),
        )

    def forward(self, inputs):
        """
        Unnormalized probabilities
        :param inputs: Mini-batches of images.
        :return: Unnormalized probabilities.
        """

        return self.classifier(inputs)  # N x num_classes


class LinearClassifier(torch.nn.Module):

    def __init__(self, num_features: int = 128, num_classes: int = 10):
        super(LinearClassifier, self).__init__()
        self.classifier = torch.nn.Linear(num_features, num_classes)

    def forward(self, inputs):
        return self.classifier(inputs)  # N x num_classes


class CentroidClassifier(torch.nn.Module):

    def __init__(self, weights: torch.FloatTensor):
        super(CentroidClassifier, self).__init__()
        self.weights = weights  # d x num_classes

    def forward(self, inputs):
        return torch.matmul(inputs, self.weights)  # N x num_classes

    @staticmethod
    def create_weights(data_loader, num_classes: int) -> torch.FloatTensor:
        X = data_loader.data
        Y = data_loader.targets

        weights = []
        for k in range(num_classes):
            ids = torch.where(Y == k)[0]
            weights.append(torch.mean(X[ids], dim=0))

        weights = torch.stack(weights, 1)
        return weights


class ContrastiveModel(torch.nn.Module):

    def __init__(self, base_cnn="resnet18", d=128):

        assert base_cnn in {"resnet18", "resnet50"}
        super(ContrastiveModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
            num_last_hidden_units = 2048
        elif base_cnn == "resnet18":
            self.f = resnet18()
            num_last_hidden_units = 512

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

        # remove the last linear layer
        self.f.fc = torch.nn.Identity()

        # projection head
        self.g = torch.nn.Sequential(
            torch.nn.Linear(num_last_hidden_units, num_last_hidden_units),
            torch.nn.BatchNorm1d(num_last_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(num_last_hidden_units, d, bias=False),
        )

    def encode(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        return self.f(inputs)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        h = self.encode(inputs)
        z = self.g(h)
        return z


class SupervisedModel(torch.nn.Module):

    def __init__(self, base_cnn="resnet18", num_classes: int = 10) -> torchvision.models.resnet.ResNet:

        assert base_cnn in {"resnet18", "resnet50"}
        super(SupervisedModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
            num_last_hidden_units = 2048
        elif base_cnn == "resnet18":
            self.f = resnet18()
            num_last_hidden_units = 512

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
