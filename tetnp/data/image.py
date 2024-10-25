from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import torch
import torchvision
from tnp.data.base import Batch, ImageBatch


@dataclass
class BatchWithMask(Batch):
    mc: torch.Tensor


class ImageGenerator(torch.utils.data.IterableDataset, ABC):
    def __init__(
        self,
        *,
        dataset: torchvision.datasets.VisionDataset,
        n: int,
        batch_size: int,
        min_pc: float,
        max_pc: float,
        nt: Optional[int] = None,
        samples_per_epoch: Optional[int] = None,
        x_mean: Optional[Tuple[float, float]] = None,
        x_std: Optional[Tuple[float, float]] = None,
        return_as_gridded: bool = False,
    ):
        assert (
            len(dataset[0][0].shape) == 3
        ), "Images must be shape (num_channels, height, width)."

        self.batch_size = batch_size
        self.min_pc = min_pc
        self.max_pc = max_pc
        self.nt = nt
        self.n = n
        self.dataset = dataset

        if samples_per_epoch is None:
            samples_per_epoch = len(self.dataset)

        self.samples_per_epoch = min(samples_per_epoch, len(self.dataset))
        self.num_batches = samples_per_epoch // batch_size
        self.return_as_gridded = return_as_gridded

        # Set input mean and std.
        if x_mean is None or x_std is None:
            x = torch.stack(
                torch.meshgrid(
                    *[
                        torch.range(0, dim - 1)
                        for dim in self.dataset[0][0][0, ...].shape
                    ]
                ),
                dim=-1,
            )
            x = einops.rearrange(x, "n1 n2 d -> (n1 n2) d")
            self.x_mean = x.mean(dim=0)
            self.x_std = x.std(dim=0)
        else:
            self.x_mean = torch.as_tensor(x_mean)
            self.x_std = torch.as_tensor(x_std)

        # Set the batch counter.
        self.batches = 0

        # These will be used when creating an iterable.
        self.batch_sampler: Optional[torch.utils.data.BatchSampler] = None

    def __iter__(self):
        """Reset epoch counter and batch sampler and return self."""
        self.batches = 0

        # Create batch sampler.
        sampler = torch.utils.data.RandomSampler(
            self.dataset, num_samples=self.samples_per_epoch
        )
        self.batch_sampler = iter(
            torch.utils.data.BatchSampler(
                sampler, batch_size=self.batch_size, drop_last=True
            )
        )
        return self

    def __next__(self) -> ImageBatch:
        """Generate next batch of data, using the `generate_batch` method.
        The `generate_batch` method should be implemented by the derived class.
        """

        if self.batches >= self.num_batches:
            raise StopIteration

        self.batches += 1
        return self.generate_batch()

    def generate_batch(self) -> ImageBatch:
        """Generate batch of data.

        Returns:
            Batch: Tuple of tensors containing the context and target data.
        """

        batch_shape = torch.Size((self.batch_size,))

        # Sample context masks.
        pc = torch.rand(size=()) * (self.max_pc - self.min_pc) + self.min_pc
        mc = self.sample_masks(prop=pc, batch_shape=batch_shape)

        # Sample batch of data.
        batch = self.sample_batch(mc=mc)

        return batch

    def sample_masks(self, prop: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
        """Sample context masks.

        Returns:
            mc: Context mask.
        """

        # Sample proportions to mask.
        num_mask = self.n * prop
        rand = torch.rand(size=(*batch_shape, self.n))
        randperm = rand.argsort(dim=-1)
        mc = randperm < num_mask

        return mc

    def sample_batch(self, mc: torch.Tensor) -> ImageBatch:
        """Sample batch of data.

        Args:
            mc: Context mask.

        Returns:
            batch: Batch of data.
        """

        # Sample batch of data.
        if self.batch_sampler is None:
            raise ValueError("Batch sampler not set.")

        batch_idx = next(self.batch_sampler)

        # (batch_size, num_channels, height, width).
        y_grid = torch.stack([self.dataset[idx][0] for idx in batch_idx], dim=0)

        # Input grid.
        x = torch.stack(
            torch.meshgrid(
                *[torch.range(0, dim - 1) for dim in y_grid[0, 0, ...].shape]
            ),
            dim=-1,
        )

        # Rearrange.
        y = einops.rearrange(y_grid, "m d n1 n2 -> m (n1 n2) d")
        x = einops.rearrange(x, "n1 n2 d -> (n1 n2) d")

        # Normalise inputs.
        x = (x - self.x_mean) / self.x_std
        x = einops.repeat(x, "n p -> m n p", m=len(batch_idx))

        xc = torch.stack([x_[mask] for x_, mask in zip(x, mc)])
        yc = torch.stack([y_[mask] for y_, mask in zip(y, mc)])
        xt = torch.stack([x_[~mask] for x_, mask in zip(x, mc)])
        yt = torch.stack([y_[~mask] for y_, mask in zip(y, mc)])

        mt_grid = None
        if self.nt is not None:
            # Only use nt randomly sampled target points.
            rand = torch.rand(size=(xt.shape[0], xt.shape[1]))
            randperm = rand.argsort(dim=-1)
            mt = randperm < self.nt
            xt = torch.stack([xt_[mt_] for xt_, mt_ in zip(xt, mt)])
            yt = torch.stack([yt_[mt_] for yt_, mt_ in zip(yt, mt)])

            if self.return_as_gridded:
                mt_grid = torch.full(mc.shape, False)
                mt_idx_orig = [torch.nonzero(~mc_).squeeze(-1) for mc_ in mc]
                mt_idx_orig_mask = [torch.nonzero(mt_).squeeze(-1) for mt_ in mt]
                mt_idx = [mt_idx_orig[i][mt_idx_orig_mask[i]] for i in range(len(mc))]

                for i, mt_idx_ in enumerate(mt_idx):
                    mt_grid[i][mt_idx_] = True

        if self.return_as_gridded:
            # Restructure mask.
            y_grid = einops.rearrange(y_grid, "m d n1 n2 -> m n1 n2 d")
            mc_grid = einops.rearrange(
                mc,
                "m (n1 n2) -> m n1 n2",
                n1=y_grid[0, ...].shape[-3],
                n2=y_grid[0, ...].shape[-2],
            )
            if mt_grid is None:
                mt_grid = ~mc_grid
            else:
                mt_grid = einops.rearrange(
                    mt_grid,
                    "m (n1 n2) -> m n1 n2",
                    n1=y_grid[0, ...].shape[-3],
                    n2=y_grid[0, ...].shape[-2],
                )

            return ImageBatch(
                y_grid=y_grid,
                mc_grid=mc_grid,
                mt_grid=mt_grid,
                yt=yt,
            )

        return BatchWithMask(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt, mc=mc)
