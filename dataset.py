import torch
from torchvision import transforms


class DownstreamDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        assert len(data) == len(targets)

        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> tuple:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)


def create_simclr_data_augmentation(strength: float, size: int) -> transforms.Compose:
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * strength,
        contrast=0.8 * strength,
        saturation=0.8 * strength,
        hue=0.2 * strength
    )

    rnd_color_jitter = transforms.RandomApply(transforms=[color_jitter], p=0.8)

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            # the following two are `color_distort`
            rnd_color_jitter,
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
        ],
    )


class SimCLRTransforms(object):
    def __init__(self, strength: float = 0.5, size: int = 32) -> None:
        # Definition is from Appendix A. of SimCLRv1 paper:
        # https://arxiv.org/pdf/2002.05709.pdf
        # Note that `clip` comes from the tensorflow code in the Appendix.

        self.transform = create_simclr_data_augmentation(strength, size)

    def __call__(self, x):
        return self.transform(x), self.transform(x)
