from functools import partial
from typing import List, Optional, Tuple, Type, Union

import escnn
import torch
from check_shapes import check_shapes
from torch import nn

ESCNN_ACT = {
    2: partial(escnn.gspaces.flipRot2dOnR2, N=8),
    3: escnn.gspaces.flipRot3dOnR3,
}

ESCNN_CONV = {
    2: escnn.nn.R2Conv,
    3: escnn.nn.R3Conv,
}


class SymmetricConv1d(nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 0.5 * (self.weight + torch.flip(self.weight, dims=(-1,)))
        return self._conv_forward(input, weight, self.bias)


class EuclideanConvBlock(nn.Module):
    def __init__(
        self,
        in_type: escnn.nn.FieldType,
        out_type: escnn.nn.FieldType,
        Conv: escnn.nn.modules.conv.rd_convolution._RdConv,
        kernel_size: int = 5,
        activation: Type[
            escnn.nn.modules.equivariant_module.EquivariantModule
        ] = escnn.nn.ReLU,
        **kwargs,
    ):
        super().__init__()

        self.activation = activation(in_type)
        padding = kernel_size // 2

        # Conv = make_depth_sep_conv(Conv)
        self.conv = Conv(
            in_type=in_type,
            out_type=out_type,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs,
        )

    @check_shapes("x: [m, c, ...]")
    def forward(self, x: torch.Tensor):
        return self.conv(self.activation(x))


class EuclideanCNN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_channels: Union[List[int], int],
        num_blocks: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if num_blocks is None:
            assert isinstance(num_channels, list)
            num_blocks = len(num_channels) - 1

        self.dim = dim
        self.num_blocks = num_blocks
        self.in_out_channels = self._get_in_out_channels(num_channels, num_blocks)

        act = ESCNN_ACT[dim]()
        self.in_out_field_types = self._in_out_field_types(act, self.in_out_channels)
        self.conv_blocks = nn.ModuleList(
            [
                EuclideanConvBlock(
                    in_ft,
                    out_ft,
                    ESCNN_CONV[dim],
                    **kwargs,
                )
                for in_ft, out_ft in self.in_out_field_types
            ]
        )

        # self.reset_parameters()

    @check_shapes("x: [m, ..., c]", "return: [m, ...]")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move channels to after batch dimension.
        x = torch.movedim(x, -1, 1)
        x = self.in_out_field_types[0][0](x)

        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Get tensor.
        x = x.tensor

        # Move channels to final dimension.
        x = torch.movedim(x, 1, -1)

        return x

    def _in_out_field_types(
        self, act: escnn.gspaces.GSpace, in_out_channels: List[Tuple[int, int]]
    ) -> List[Tuple[escnn.nn.FieldType, escnn.nn.FieldType]]:
        """Return a list of tuple of input and output channels."""
        in_out_field_types = []
        for i, (in_c, out_c) in enumerate(in_out_channels):
            if i == 0:
                in_rep = in_c * [act.trivial_repr]
            else:
                # in_rep = in_c * [act.trivial_repr]
                in_rep = in_c * [act.regular_repr]

            if i == len(in_out_channels) - 1:
                out_rep = out_c * [act.trivial_repr]
            else:
                # out_rep = out_c * [act.trivial_repr]
                out_rep = out_c * [act.regular_repr]

            in_out_field_types.append(
                (escnn.nn.FieldType(act, in_rep), escnn.nn.FieldType(act, out_rep))
            )

        return in_out_field_types

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
