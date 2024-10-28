import copy
from typing import List, Optional, Tuple, Type, Union

import einops
import torch
from check_shapes import check_shapes
from tnp.networks.cnn import CONV, POOL, UPSAMPLE_MODE, ConvBlock
from torch import nn

from tetnp.utils.dropout import dropout_all


class ATECNN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_channels: Union[List[int], int],
        num_basis: int,
        num_blocks: Optional[int] = None,
        basis_fn_connector: nn.Module = None,
        **kwargs,
    ):
        super().__init__()

        if num_blocks is None:
            assert isinstance(num_channels, list)
            num_blocks = len(num_channels) - 1

        self.dim = dim
        self.num_basis = num_basis
        self.num_blocks = num_blocks
        self.in_out_channels = self._get_in_out_channels(num_channels, num_blocks)
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_c, out_c, CONV[dim], **kwargs)
                for in_c, out_c in self.in_out_channels
            ]
        )

        # Modify conv_blocks to account for
        self.num_basis = num_basis

        # Basis functions on the grid
        self.basis_fn_blocks = _get_clones(
            basis_fn_connector, len(self.conv_blocks) - 1
        )

    @check_shapes("x: [m, ..., c]", "dropout: [m]", "return: [m, ...]")
    def forward(self, x: torch.Tensor, dropout: torch.Tensor) -> torch.Tensor:
        # Move channels to after batch dimension.
        x = torch.movedim(x, -1, 1)

        x_conv, x_basis = torch.split(
            x, [x.shape[1] - self.num_basis, self.num_basis], dim=1
        )

        for conv_block, basis_fn_block in zip(
            self.conv_blocks[:-1], self.basis_fn_blocks
        ):
            # Update conv part.
            x_conv = conv_block(x)

            # Update basis part.
            x_basis = torch.movedim(x_basis, 1, -1)
            x_basis = basis_fn_block(x_basis)
            x_basis = dropout_all(x_basis, dropout=dropout)
            x_basis = torch.movedim(x_basis, -1, 1)

            # Merge back together.
            x = torch.cat((x_conv, x_basis), dim=1)

        # Final conv block.
        x_conv = self.conv_blocks[-1](x)

        # Move channels to final dimension.
        x_conv = torch.movedim(x_conv, 1, -1)

        return x_conv

    def _get_in_out_channels(
        self, num_channels: Union[List[int], int], num_blocks: int
    ) -> List[Tuple[int, int]]:
        """Return a list of tuple of input and output channels."""
        if isinstance(num_channels, int):
            channel_list = [num_channels + self.num_basis] * (num_blocks + 1)
        else:
            channel_list = [n + self.num_basis for n in num_channels]

        assert len(channel_list) == (
            num_blocks + 1
        ), f"{len(channel_list)} != {num_blocks}."

        out_channel_list = [n - self.num_basis for n in channel_list[1:]]

        return list(zip(channel_list, out_channel_list))


class RelaxedConvBlock(nn.Module):
    """Implementation of https://arxiv.org/pdf/2201.11969.pdf."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Conv: nn.Module,
        num_basis: int,
        kernel_size: int = 5,
        basis_fn: Optional[nn.Module] = None,
        grid_size: Optional[Tuple[int, ...]] = None,
        activation: nn.Module = nn.ReLU(),
        **kwargs,
    ):
        super().__init__()

        self.activation = activation
        self.out_channels = out_channels
        padding = kernel_size // 2

        # Conv = make_depth_sep_conv(Conv)
        self.conv = Conv(
            in_channels,
            num_basis * out_channels,
            kernel_size,
            padding=padding,
            **kwargs,
        )

        self.num_basis = num_basis

        if basis_fn is None:
            assert grid_size is not None
            basis_fn = nn.Parameter(torch.randn(*grid_size, num_basis))

        self.basis_fn = basis_fn

    @check_shapes(
        "z: [m, cin, ...]",
        "x: [m, ..., dx]",
        "dropout: [m]",
        "return: [m, cout, ...]",
    )
    def forward(
        self,
        z: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        dropout: Optional[torch.Tensor] = None,
    ):
        # (m, cin, ...).
        z = self.activation(z)
        # (m, num_basis * cout, n1, n2, ..., ndim).
        conv_out = self.conv(z)
        # (m, cout, n1, n2, ..., ndim, num_basis).
        conv_out = torch.stack(torch.split(conv_out, self.out_channels, dim=1), dim=-1)

        # (m, n1, n2, ..., ndim, num_basis).
        if isinstance(self.basis_fn, nn.Module):
            if x is None:
                # Assumes all data always lies on same grid.
                x = torch.stack(
                    torch.meshgrid(
                        torch.range(0, z.shape[-2] - 1), torch.range(0, z.shape[-1] - 1)
                    ),
                    dim=-1,
                ).to(z)
                if x.shape[-2] == 1:
                    x = torch.zeros_like(x)
                else:
                    x = (x - x.mean()) / x.std()
                x = einops.repeat(x, "n1 n2 d -> m n1 n2 d", m=z.shape[0])

            basis_out = self.basis_fn(x)
        else:
            basis_out = einops.repeat(
                self.basis_fn, "n1 n2 d -> m n1 n2 d", m=z.shape[0]
            )

        basis_out = dropout_all(
            basis_out,
            dropout=dropout,
        )

        out = (conv_out * (1 + basis_out[:, None, ...])).sum(-1) / (self.num_basis * 3)
        return out


class RelaxedCNN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_channels: Union[List[int], int],
        basis_fn: Optional[Union[nn.Module, Tuple[nn.Module, ...]]] = None,
        num_blocks: Optional[int] = None,
        conv_block: Type[RelaxedConvBlock] = RelaxedConvBlock,
        grid_size: Optional[Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]] = None,
        **kwargs,
    ):
        super().__init__()

        if num_blocks is None:
            assert isinstance(num_channels, list)
            num_blocks = len(num_channels) - 1

        self.dim = dim
        self.num_blocks = num_blocks
        self.in_out_channels = self._get_in_out_channels(num_channels, num_blocks)

        if isinstance(basis_fn, nn.Module):
            basis_fns = [
                copy.deepcopy(basis_fn) for _ in range(len(self.in_out_channels))
            ]
        elif basis_fn is None:
            basis_fns = [None] * len(self.in_out_channels)
        else:
            assert len(basis_fn) == len(self.in_out_channels)
            basis_fns = basis_fn

        if grid_size is None or isinstance(grid_size[0], int):
            grid_size = [grid_size] * len(self.in_out_channels)

        self.conv_blocks = nn.ModuleList(
            [
                conv_block(
                    in_c,
                    out_c,
                    CONV[dim],
                    basis_fn=basis_fn_,
                    grid_size=grid_size_,
                    **kwargs,
                )
                for basis_fn_, (in_c, out_c), grid_size_ in zip(
                    basis_fns, self.in_out_channels, grid_size
                )
            ]
        )

    @check_shapes(
        "z: [m, ..., cin]", "x: [m, ..., dx]", "dropout: [m]", "return: [m, ..., cout]"
    )
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        dropout: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Move channels to after batch dimension.
        z = torch.movedim(z, -1, 1)

        for conv_block in self.conv_blocks:
            z = conv_block(z, x, dropout)

        # Move channels to final dimension.
        z = torch.movedim(z, 1, -1)

        return z

    def _get_in_out_channels(
        self, num_channels: Union[List[int], int], num_blocks: int
    ) -> List[Tuple[int, int]]:
        """Return a list of tuple of input and output channels."""
        if isinstance(num_channels, int):
            channel_list = [num_channels] * (num_blocks + 1)
        else:
            channel_list = list(num_channels)

        assert len(channel_list) == (
            num_blocks + 1
        ), f"{len(channel_list)} != {num_blocks}."

        return list(zip(channel_list, channel_list[1:]))


class RelaxedUNet(RelaxedCNN):
    def __init__(
        self,
        dim: int,
        num_channels: Union[int, List[int]],
        num_blocks: Optional[int] = None,
        max_num_channels: int = 256,
        pooling_size: int = 2,
        factor_chan: int = 2,
        grid_size: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ):
        self.max_num_channels = max_num_channels
        self.factor_chan = factor_chan

        if grid_size is not None:
            if num_blocks is None:
                assert isinstance(num_channels, list)
                num_blocks = len(num_channels) - 1

            # Compute grid size for each conv block...
            grid_sizes = [grid_size]
            for _ in range(num_blocks // 2):
                grid_size = tuple(s // pooling_size for s in grid_size)
                grid_sizes.append(grid_size)

            grid_sizes = grid_sizes + grid_sizes[::-1][1:]
        else:
            grid_sizes = grid_size

        super().__init__(
            dim=dim,
            num_channels=num_channels,
            num_blocks=num_blocks,
            grid_size=grid_sizes,
            **kwargs,
        )

        self.pooling_size = pooling_size
        self.pooling = POOL[dim](pooling_size)
        self.upsample_mode = UPSAMPLE_MODE[dim]

    @check_shapes(
        "z: [m, cin, ...]",
        "x: [m, ..., dx]",
        "dropout: [m]",
        "return: [m, cout, ...]",
    )
    def forward(
        self,
        z: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        dropout: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Move channels to after batch dimension.
        z = torch.movedim(z, -1, 1)

        num_down_blocks = self.num_blocks // 2
        residuals = []
        xs = []

        # Downwards convolutions.
        for i in range(num_down_blocks):
            z = self.conv_blocks[i](z, x, dropout)
            residuals.append(z)
            xs.append(x)
            z = self.pooling(z)

            if x is not None:
                x = torch.movedim(x, -1, 1)
                x = self.pooling(x)
                x = torch.movedim(x, 1, -1)

        # Bottleneck.
        z = self.conv_blocks[num_down_blocks](z, x, dropout)

        # Upwards convolutions.
        for i in range(num_down_blocks + 1, self.num_blocks):
            z = nn.functional.interpolate(
                z,
                size=residuals[num_down_blocks - i].shape[-self.dim :],
                mode=self.upsample_mode,
                align_corners=True,
            )

            z = torch.cat((z, residuals[num_down_blocks - i]), dim=1)
            z = self.conv_blocks[i](z, xs[num_down_blocks - i], dropout)

        z = torch.movedim(z, 1, -1)
        return z

    def _get_in_out_channels(
        self, num_channels: Union[List[int], int], num_blocks: int
    ) -> List[Tuple[int, int]]:
        # Doubles at every down layer, as in vanila UNet.
        factor_chan = self.factor_chan

        assert num_blocks % 2 == 1, f"n_blocks={num_blocks} not odd."

        if isinstance(num_channels, int):
            # e.g. if n_channels=16, n_blocks=5: [16, 32, 64].
            channel_list = [
                factor_chan**i * num_channels for i in range(num_blocks // 2 + 1)
            ]
        else:
            channel_list = list(num_channels)

        # e.g.: [16, 32, 64, 64, 32, 16].
        channel_list = channel_list + channel_list[::-1]

        # Bound max number of channels by self.max_nchannels (besides first and
        # last dim as this is input / output should not be changed).
        channel_list = (
            channel_list[:1]
            + [min(c, self.max_num_channels) for c in channel_list[1:-1]]
            + channel_list[-1:]
        )

        # e.g.: [(16, 32), (32, 64), (64, 64), (64, 32), (32, 16)].
        in_out_channels = super()._get_in_out_channels(channel_list, num_blocks)
        # e.g.: [(16, 32), (32, 64), (64, 64), (128, 32), (64, 16)] due to concat.
        idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels))
        in_out_channels[idcs] = [
            (in_chan * 2, out_chan) for in_chan, out_chan in in_out_channels[idcs]
        ]
        return in_out_channels


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
