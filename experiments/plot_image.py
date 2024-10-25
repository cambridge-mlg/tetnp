import copy
import os
from typing import Callable, List, Tuple

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tnp.data.base import ImageBatch
from tnp.utils.np_functions import np_pred_fn
from torch import nn

from tetnp.data.image import BatchWithMask

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_image(
    model: nn.Module,
    batches: List[ImageBatch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (24.0, 8.0),
    name: str = "plot",
    subplots: bool = True,
    savefig: bool = False,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
):
    for i in range(num_fig):
        batch = batches[i]

        for key, value in vars(batch).items():
            if isinstance(value, torch.Tensor):
                setattr(batch, key, value[:1])

        plot_batch = copy.deepcopy(batch)
        if isinstance(batch, ImageBatch):
            plot_batch.mt_grid = torch.full(batch.mt_grid.shape, True)
        else:
            plot_batch.xt = batch.x

        with torch.no_grad():
            y_plot_pred_dist = pred_fn(model, plot_batch)
            yt_pred_dist = pred_fn(model, batch)

            mean, std = (
                y_plot_pred_dist.mean.cpu().numpy(),
                y_plot_pred_dist.stddev.cpu().numpy(),
            )
            model_nll = (
                -yt_pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
            )

        # Reorganise into grid.
        if isinstance(batch, ImageBatch):
            mc = batch.mc_grid.flatten(start_dim=1)
            y = batch.y_grid.flatten(start_dim=1, end_dim=-2)
            prop_ctx = batch.mc_grid[0].sum() / batch.y_grid[0, ..., 0].numel()
        else:
            assert isinstance(batch, BatchWithMask)
            mc = batch.mc
            y = batch.y
            prop_ctx = batch.xc.shape[-2] / batch.x.shape[-2]

        if y.shape[-1] == 1:
            # Single channel.
            mc_ = einops.repeat(mc[:1], "m n -> m n d", d=y.shape[-1])
            yc_ = np.ma.masked_where(
                ~mc_.cpu().numpy(),
                y[:1, :].cpu().numpy(),
            )
        else:
            # Three channels.
            # Masking does not work for RGB images.
            # Use mask to control alpha values instead.
            mc_ = einops.rearrange(mc[:1], "m n -> m n 1")
            yc_ = torch.cat((y[:1], mc_), dim=-1).cpu().numpy()

        # Assumes same height and width.
        w = int(yc_.shape[-2] ** 0.5)
        yc_ = einops.rearrange(yc_, "1 (n m) d -> n m d", n=w, m=w)
        y_ = einops.rearrange(y[:1, :].cpu().numpy(), "1 (n m) d -> n m d", n=w, m=w)
        mean = einops.rearrange(mean, "1 (n m) d -> n m d", n=w, m=w)
        std = einops.rearrange(std, "1 (n m) d -> n m d", n=w, m=w)

        if subplots:
            # Make figure for plotting
            fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=1)

            axes[0].imshow(yc_, cmap="gray", vmax=1, vmin=0)
            axes[1].imshow(mean, cmap="gray", vmax=1, vmin=0)
            axes[2].imshow(std, cmap="gray", vmax=std.max(), vmin=std.min())

            axes[0].set_title("Context set", fontsize=18)
            axes[1].set_title("Mean prediction", fontsize=18)
            axes[2].set_title("Std prediction", fontsize=18)

            plt.suptitle(
                f"prop_ctx = {prop_ctx:.2f}, NLL = {model_nll:.3f}",
                fontsize=24,
            )

            fname = f"fig/{name}/{i:03d}"
            if wandb.run is not None and logging:
                wandb.log({fname: wandb.Image(fig)})
            elif savefig:
                if not os.path.isdir(f"fig/{name}"):
                    os.makedirs(f"fig/{name}")
                plt.savefig(fname)
            else:
                plt.show()

            plt.close()

        else:
            for fig_name, y_plot in zip(
                ("context", "ground_truth", "pred_mean"), (yc_, y_, mean)
            ):
                fig = plt.figure(figsize=figsize)

                plt.imshow(y_plot, vmax=1, vmin=0)
                plt.tight_layout()

                fname = f"fig/{name}/{i:03d}/{fig_name}"
                if wandb.run is not None and logging:
                    wandb.log({fname: wandb.Image(fig)})
                elif savefig:
                    if not os.path.isdir(f"fig/{name}/{i:03d}"):
                        os.makedirs(f"fig/{name}/{i:03d}")
                    plt.savefig(fname)
                else:
                    plt.show()

                plt.close()
