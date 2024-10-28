import einops
import torch
from check_shapes import check_shapes
from tnp.models.convcnp import ConvCNPEncoder, GriddedConvCNPEncoder
from torch import nn

from tetnp.networks.atecnn import ATECNN
from tetnp.utils.dropout import dropout_all


class ATEConvCNPEncoder(ConvCNPEncoder):
    force_dropout = False

    def __init__(
        self,
        *,
        basis_fn: nn.Module,
        p_basis_dropout: float = 0.5,
        sum_basis: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.basis_fn = basis_fn
        self.p_basis_dropout = p_basis_dropout
        self.sum_basis = sum_basis

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dz]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        # Add density.
        yc = torch.cat((yc, torch.ones(yc.shape[:-1] + (1,)).to(yc)), dim=-1)

        # Encode to grid.
        x_grid, z_grid = self.grid_encoder(xc, yc)

        # Encode to z.
        z_grid = self.z_encoder(z_grid)

        # Get basis_fn values.
        z_grid_basis = self.basis_fn(x_grid)

        if self.force_dropout:
            z_grid_basis, dropout = dropout_all(
                z_grid_basis, 1.0, True, return_dropout=True
            )
        else:
            z_grid_basis, dropout = dropout_all(
                z_grid_basis, self.p_basis_dropout, self.training, return_dropout=True
            )

        # Whether we add or concatenate basis functions
        if self.sum_basis:
            z_grid = z_grid + z_grid_basis
        else:
            z_grid = torch.cat((z_grid, z_grid_basis), dim=-1)

        # Pass through conv_net.
        if isinstance(self.conv_net, ATECNN):
            z_grid = self.conv_net(z_grid, dropout=dropout)
        else:
            z_grid = self.conv_net(z_grid)

        # Decode.
        zt = self.grid_decoder(x_grid, z_grid, xt)

        return zt


class RelaxedConvCNPEncoder(ConvCNPEncoder):
    force_dropout = False

    def __init__(
        self,
        *,
        p_basis_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.p_basis_dropout = p_basis_dropout

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dz]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        # Add density.
        yc = torch.cat((yc, torch.ones(yc.shape[:-1] + (1,)).to(yc)), dim=-1)

        # Encode to grid.
        x_grid, z_grid = self.grid_encoder(xc, yc)

        # Encode to z.
        z_grid = self.z_encoder(z_grid)

        # Whether or not to use dropout.
        if self.force_dropout:
            _, dropout = dropout_all(z_grid, 1.0, True, return_dropout=True)
        else:
            _, dropout = dropout_all(
                z_grid, self.p_basis_dropout, self.training, return_dropout=True
            )

        # Convolve.
        z_grid = self.conv_net(z_grid, x_grid, dropout=dropout)

        # Decode.
        zt = self.grid_decoder(x_grid, z_grid, xt)
        return zt


class GriddedRelaxedConvCNPEncoder(GriddedConvCNPEncoder):
    force_dropout = False

    def __init__(
        self,
        *,
        p_basis_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.p_basis_dropout = p_basis_dropout

    @check_shapes(
        "mc: [m, ...]",
        "y: [m, ..., dy]",
        "mt: [m, ...]",
        "return: [m, dt, dz]",
    )
    def forward(
        self, mc: torch.Tensor, y: torch.Tensor, mt: torch.Tensor
    ) -> torch.Tensor:
        mc_ = einops.repeat(mc, "m n1 n2 -> m n1 n2 d", d=y.shape[-1])
        yc = y * mc_
        z_grid = torch.cat((yc, mc_), dim=-1)
        z_grid = self.z_encoder(z_grid)

        # Whether or not to use dropout.
        if self.force_dropout:
            _, dropout = dropout_all(z_grid, 1.0, True, return_dropout=True)
        else:
            _, dropout = dropout_all(
                z_grid, self.p_basis_dropout, self.training, return_dropout=True
            )

        z_grid = self.conv_net(z_grid, dropout=dropout)
        zt = torch.stack([z_grid[i][mt[i]] for i in range(mt.shape[0])])
        return zt


class GriddedATEConvCNPEncoder(GriddedConvCNPEncoder):
    force_dropout = False

    def __init__(
        self,
        *,
        basis_fn: nn.Module,
        sum_basis: bool = True,
        p_basis_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.basis_fn = basis_fn
        self.p_basis_dropout = p_basis_dropout
        self.sum_basis = sum_basis

    @check_shapes(
        "mc: [m, ...]",
        "y: [m, ..., dy]",
        "mt: [m, ...]",
        "return: [m, dt, dz]",
    )
    def forward(
        self, mc: torch.Tensor, y: torch.Tensor, mt: torch.Tensor
    ) -> torch.Tensor:
        mc_ = einops.repeat(mc, "m n1 n2 -> m n1 n2 d", d=y.shape[-1])
        yc = y * mc_
        z_grid = torch.cat((yc, mc_), dim=-1)
        z_grid = self.z_encoder(z_grid)

        # Construct x_grid to pass through basis_fn.
        x_grid = torch.stack(
            torch.meshgrid(
                torch.range(0, y.shape[-3] - 1), torch.range(0, y.shape[-2] - 1)
            ),
            dim=-1,
        ).to(y)
        x_grid = (x_grid - x_grid.mean()) / x_grid.std()

        z_grid_basis = self.basis_fn(x_grid)
        if self.force_dropout:
            z_grid_basis, dropout = dropout_all(
                z_grid_basis, 1.0, True, return_dropout=True
            )
        else:
            z_grid_basis, dropout = dropout_all(
                z_grid_basis, self.p_basis_dropout, self.training, return_dropout=True
            )

        # Whether we add or concatenate basis functions
        if self.sum_basis:
            z_grid = z_grid + z_grid_basis
        else:
            z_grid = torch.cat((z_grid, z_grid_basis), dim=-1)

        if isinstance(self.conv_net, ATECNN):
            z_grid = self.conv_net(z_grid, dropout=dropout)
        else:
            z_grid = self.conv_net(z_grid)

        zt = torch.stack([z_grid[i][mt[i]] for i in range(mt.shape[0])])
        return zt
