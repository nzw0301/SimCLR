import torch


class NT_Xent(torch.nn.Module):
    """
    Normalised temperature-scaled cross entropy loss.
    """

    def __init__(self, temperature: float = 0.1, reduction: str = "mean",
                 device: torch.device = torch.device("cpu")) -> None:
        """
        :param temperature: Temperature parameter. The value must be positive.
        :param reduction: Same to PyTorch's `reduction` in losses.
        :param device: PyTorch's device instance.
        """
        assert temperature > 0.
        assert reduction in {"none", "mean", "sum"}

        super(NT_Xent, self).__init__()
        self.cross_entropy = torch.nn.functional.cross_entropy
        self.temperature = temperature
        self.reduction = reduction
        self.device = device

    def forward(self, view0: torch.FloatTensor, view1: torch.FloatTensor) -> torch.FloatTensor:
        """
        :param view0: Feature representation. The shape is (N, D), where `N` is the size of mini-batches,
            and `D` is the dimensionality of features.
        :param view1: The other feature representations. The shape is same to `view0`'s shape.
        :return: Loss value. The shape depends on `reduction`: (2, N) or a scaler.
        """

        view0 = torch.nn.functional.normalize(view0, p=2, dim=1)
        view1 = torch.nn.functional.normalize(view1, p=2, dim=1)

        size_mini_batches = len(view0)
        targets = torch.arange(size_mini_batches).to(self.device)  # == indices for positive pairs
        mask = ~torch.eye(size_mini_batches, dtype=torch.bool).to(self.device)  # to remove similarity to themselves

        size_mini_batches = len(view0)

        sim00 = torch.matmul(view0, view0.t()) / self.temperature  # N x N
        sim11 = torch.matmul(view1, view1.t()) / self.temperature  # N x N

        # remove own similarities
        sim00 = sim00[mask].view(size_mini_batches, -1)  # N x (N-1)
        sim11 = sim11[mask].view(size_mini_batches, -1)  # N x (N-1)

        sim01 = torch.matmul(view0, view1.t()) / self.temperature  # N x N

        sim0 = torch.cat([sim01, sim00], dim=1)  # N x (N+(N-1))
        sim1 = torch.cat([sim01.t(), sim11], dim=1)  # N x (N+(N-1))

        if self.reduction == "none":
            return torch.stack([
                self.cross_entropy(sim0, targets, reduction="none"),
                self.cross_entropy(sim1, targets, reduction="none")
            ])  # (2, N)
        else:
            loss = self.cross_entropy(sim0, targets, reduction="sum") + self.cross_entropy(sim1, targets,
                                                                                           reduction="sum")
            if self.reduction == "sum":
                return loss  # (2, N)
            else:
                return loss / size_mini_batches * 0.5  # shape: scaler
