from typing import Optional, Tuple

import numpy as np
import torch
import torchvision

from .image import ImageGenerator


class TranslationImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_translation: Tuple[int, int],
        stationary_image_size: Optional[Tuple[int, int, int]] = None,
        translated_image_size: Optional[Tuple[int, int, int]] = None,
        train: bool = True,
        zero_shot: bool = True,
        seed: int = 0,
    ):
        self.seed = seed
        # (height, width, num_channels).
        self.image_size = dataset[0][0].permute(1, 2, 0).shape
        self.max_translation = max_translation

        self.stationary_image_size = (
            self.image_size if stationary_image_size is None else stationary_image_size
        )
        self.translated_image_size = (
            [
                dim + 2 * shift
                for dim, shift in zip(self.image_size[:-1], max_translation)
            ]
            + [self.image_size[-1]]
            if translated_image_size is None
            else translated_image_size
        )

        # Make transforms.
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor()]
        )

        # Should be scaled between 0 and 1.
        data = torch.stack([img[0] for img in dataset], dim=0).permute(0, 2, 3, 1)
        if train and zero_shot:
            self.data = self.make_stationary_images(data)
        else:
            self.data = self.make_translated_images(data)

    def make_stationary_images(self, dataset: torch.Tensor) -> torch.Tensor:
        # (num_images, height, width, num_channels).
        background = np.zeros((dataset.shape[0], *self.stationary_image_size)).astype(
            np.uint8
        )

        borders = (
            np.array(self.stationary_image_size[:-1]) - np.array(self.image_size[:-1])
        ) // 2
        background[
            :,
            borders[0] : (background.shape[1] - borders[0]),
            borders[1] : (background.shape[2] - borders[1]),
        ] = dataset
        return torch.from_numpy(background)

    def make_translated_images(self, dataset: torch.Tensor) -> torch.Tensor:
        # (num_images, height, width, num_channels).
        background = torch.from_numpy(
            (np.zeros((dataset.shape[0], *self.translated_image_size)).astype(np.uint8))
        )

        st = np.random.get_state()
        np.random.seed(self.seed)
        vertical_shifts = np.random.randint(
            low=-self.max_translation[0],
            high=self.max_translation[0],
            size=dataset.shape[0],
        )
        horizontal_shifts = np.random.randint(
            low=-self.max_translation[1],
            high=self.max_translation[1],
            size=dataset.shape[0],
        )
        np.random.set_state(st)
        borders = (
            np.array(self.translated_image_size[:-1]) - np.array(self.image_size[:-1])
        ) // 2

        for i, (vshift, hshift) in enumerate(zip(vertical_shifts, horizontal_shifts)):
            img = dataset[i, ...]

            # Trim original image to fit within background.
            if vshift < -borders[0]:
                # Trim bottom.
                img = img[-(vshift + borders[0]) :, :]
            elif vshift > borders[0]:
                # Trim top.
                img = img[: -(vshift + borders[0] - self.image_size[0]), :]

            if hshift + borders[1] < 0:
                # Trim left.
                img = img[:, -(hshift + borders[1]) :]
            elif hshift > borders[1]:
                # Trim right.
                img = img[:, : -(hshift + borders[1] - self.image_size[1])]

            vslice = slice(
                max(0, vshift + borders[0]), max(0, vshift + borders[1]) + img.shape[0]
            )
            hslice = slice(
                max(0, hshift + borders[1]), max(0, hshift + borders[1]) + img.shape[1]
            )
            background[i, vslice, hslice] = torch.as_tensor(img)

        return background

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Move channel dim to first dimension for compatability with torchvision.datasets.
        img = self.transforms(self.data[idx].permute(2, 0, 1)).float()
        return img, 0


class TranslatedImageGenerator(ImageGenerator):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        train: bool = True,
        max_translation: Tuple[int, int] = (14, 14),
        stationary_image_size: Optional[Tuple[int, int, int]] = None,
        translated_image_size: Optional[Tuple[int, int, int]] = None,
        zero_shot: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        self.dataset = TranslationImageDataset(
            dataset=dataset,
            max_translation=max_translation,
            stationary_image_size=stationary_image_size,
            translated_image_size=translated_image_size,
            train=train,
            zero_shot=zero_shot,
            seed=seed,
        )
        self.n = self.dataset.data.shape[1] * self.dataset.data.shape[2]
        super().__init__(dataset=self.dataset, n=self.n, **kwargs)
