from abc import ABC

import torchvision

from .image import ImageGenerator
from .image_datasets import TranslatedImageGenerator


class MNIST(ABC):
    def __init__(self, data_dir: str, train: bool = True, download: bool = False):
        self.n = 28 * 28
        self.dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )


class MNISTGenerator(MNIST, ImageGenerator):
    def __init__(
        self, *, data_dir: str, train: bool = True, download: bool = False, **kwargs
    ):
        MNIST.__init__(self, data_dir, train, download)
        ImageGenerator.__init__(self, dataset=self.dataset, n=self.n, **kwargs)


class TranslatedMNISTGenerator(MNIST, TranslatedImageGenerator):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        **kwargs,
    ):
        MNIST.__init__(self, data_dir, train, download)
        TranslatedImageGenerator.__init__(
            self, dataset=self.dataset, train=train, **kwargs
        )
