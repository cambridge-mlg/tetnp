from abc import ABC

import torchvision

from .image import ImageGenerator
from .image_datasets import TranslatedImageGenerator


class CIFAR10(ABC):
    def __init__(self, data_dir: str, train: bool = True, download: bool = False):
        self.n = 32 * 32
        self.dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize(
                    #     (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    # ),
                ]
            ),
        )


class CIFAR10Generator(CIFAR10, ImageGenerator):
    def __init__(
        self, *, data_dir: str, train: bool = True, download: bool = False, **kwargs
    ):
        CIFAR10.__init__(self, data_dir, train, download)
        ImageGenerator.__init__(self, dataset=self.dataset, n=self.n, **kwargs)


class TranslatedCIFAR10Generator(CIFAR10, TranslatedImageGenerator):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        **kwargs,
    ):
        CIFAR10.__init__(self, data_dir, train, download)
        TranslatedImageGenerator.__init__(
            self, dataset=self.dataset, train=train, **kwargs
        )
