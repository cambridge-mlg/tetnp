import warnings
from abc import ABC
from typing import Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from tnp.utils.grids import construct_grid, flatten_grid
from torch import nn

from tetnp.networks.tetransformer import (
    TEISTEncoder,
    TEPerceiverEncoder,
    TETNPTransformerEncoder,
    TETransformerEncoder,
)
from tetnp.utils.dropout import dropout_all


class BaseATETransformerEncoder(nn.Module, ABC):
    force_dropout = False

    def __init__(
        self,
        basis_fn: nn.Module,
        p_basis_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.basis_fn = basis_fn
        self.p_basis_dropout = p_basis_dropout


class ATETransformerEncoder(BaseATETransformerEncoder, TETransformerEncoder):
    @check_shapes(
        "z: [m, n, d]", "x: [m, n, dx]", "mask: [m, n, n]", "return: [m, n, d]"
    )
    def forward(
        self, z: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Obtain basis functions and concatenate.
        z_basis = self.basis_fn(x)
        if self.force_dropout:
            z_basis = dropout_all(z_basis, 1.0, True)
        else:
            z_basis = dropout_all(z_basis, self.p_basis_dropout, self.training)

        z = z + z_basis
        return super().forward(z, x, mask)


class ATETNPTransformerEncoder(BaseATETransformerEncoder, TETNPTransformerEncoder):
    @check_shapes(
        "zc: [m, nc, dz]",
        "zt: [m, nt, dz]",
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "mask: [m, nt, nc]",
        "return: [m, nt, dz]",
    )
    def forward(
        self,
        zc: torch.Tensor,
        zt: torch.Tensor,
        xc: torch.Tensor,
        xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Obtain basis functions and concatenate.
        zc_basis = self.basis_fn(xc)
        if self.force_dropout:
            zc_basis = dropout_all(zc_basis, 1.0, True)
        else:
            zc_basis = dropout_all(zc_basis, self.p_basis_dropout, self.training)

        # Try just summing.
        zc = zc + zc_basis

        return super().forward(zc, zt, xc, xt, mask)


class ATEPerceiverEncoder(BaseATETransformerEncoder, TEPerceiverEncoder):
    def __init__(
        self,
        *,
        gridded_pseudo_tokens: bool = False,
        grid_range: Optional[Tuple[Tuple[float, float], ...]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if gridded_pseudo_tokens:
            assert (
                grid_range is not None
            ), "Must specify grid_range if gridded_pseudo_tokens."
            # Reinitialise pseudo-tokens to be on the grid, assuming num_pseudo is num_pseudo per dimension.
            points_per_dim = self.latent_inputs.shape[0]
            latent_inputs = construct_grid(
                grid_range=grid_range,
                points_per_dim=(points_per_dim,) * len(grid_range),
            )
            latent_inputs, _ = flatten_grid(latent_inputs, start_dim=0)

            self.latent_inputs = nn.Parameter(latent_inputs, requires_grad=False)
            self.latent_tokens = nn.Parameter(
                torch.randn(self.latent_inputs.shape[0], self.embed_dim)
            )

            # Keep grid fixed throughout.
            self.pseudo_token_initialiser = lambda zq, zc, xq, xc: (zq, xq)

    @check_shapes(
        "zc: [m, nc, dz]",
        "zt: [m, nt, dz]",
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "mask: [m, nq, n]",
        "return: [m, nq, dz]",
    )
    def forward(
        self,
        zc: torch.Tensor,
        zt: torch.Tensor,
        xc: torch.Tensor,
        xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        zq = einops.repeat(self.latent_tokens, "l e -> m l e", m=zc.shape[0])
        xq = einops.repeat(self.latent_inputs, "l d -> m l d", m=zc.shape[0])

        # Now initialise pseudo-tokens.
        zq, xq = self.pseudo_token_initialiser(zq, zc, xq, xc)

        # Obtain basis functions and concatenate.
        zq_basis = self.basis_fn(xq)
        if self.force_dropout:
            zq_basis = dropout_all(zq_basis, 1.0, True)
        else:
            zq_basis = dropout_all(zq_basis, self.p_basis_dropout, self.training)

        # Try just summing.
        zq = zq + zq_basis

        for mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer in zip(
            self.mhsa_layers, self.mhca_ctoq_layers, self.mhca_qtot_layers
        ):
            zq, xq = mhca_ctoq_layer(zq, zc, xq, xc)
            zq, xq = mhsa_layer(zq, xq)
            zt, xt = mhca_qtot_layer(zt, zq, xt, xq)

        return zt


class ATEISTEncoder(BaseATETransformerEncoder, TEISTEncoder):
    def __init__(
        self,
        *,
        gridded_pseudo_tokens: bool = False,
        grid_range: Optional[Tuple[Tuple[float, float], ...]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if gridded_pseudo_tokens:
            assert (
                grid_range is not None
            ), "Must specify grid_range if gridded_pseudo_tokens."
            # Reinitialise pseudo-tokens to be on the grid, assuming num_pseudo is num_pseudo per dimension.
            points_per_dim = self.latent_inputs.shape[0]
            latent_inputs = construct_grid(
                grid_range=grid_range,
                points_per_dim=(points_per_dim,) * len(grid_range),
            )
            latent_inputs, _ = flatten_grid(latent_inputs, start_dim=0)

            self.latent_inputs = nn.Parameter(latent_inputs, requires_grad=False)
            self.latent_tokens = nn.Parameter(
                torch.randn(self.latent_inputs.shape[0], self.embed_dim)
            )

            # Keep grid fixed throughout.
            self.pseudo_token_initialiser = lambda zq, zc, xq, xc: (zq, xq)

    @check_shapes(
        "zc: [m, nc, dz]",
        "zt: [m, nt, dz]",
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "mask: [m, nq, n]",
        "return: [m, nq, dz]",
    )
    def forward(
        self,
        zc: torch.Tensor,
        zt: torch.Tensor,
        xc: torch.Tensor,
        xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        zq = einops.repeat(self.latent_tokens, "l e -> m l e", m=zc.shape[0])
        xq = einops.repeat(self.latent_inputs, "l d -> m l d", m=zc.shape[0])

        # Now initialise pseudo-tokens.
        zq, xq = self.pseudo_token_initialiser(zq, zc, xq, xc)

        # Obtain basis functions and concatenate.
        zq_basis = self.basis_fn(xq)
        if self.force_dropout:
            zq_basis = dropout_all(zq_basis, 1.0, True)
        else:
            zq_basis = dropout_all(zq_basis, self.p_basis_dropout, self.training)

        # Try just summing.
        zq = zq + zq_basis

        for mhca_ctoq_layer, mhca_qtoc_layer, mhca_qtot_layer in zip(
            self.mhca_ctoq_layers, self.mhca_qtoc_layers, self.mhca_qtot_layers
        ):
            zq, xq = mhca_ctoq_layer(zq, zc, xq, xc)
            zc, xc = mhca_qtoc_layer(zc, zq, xc, xq)
            zt, xt = mhca_qtot_layer(zt, zq, xt, xq)

        return zt
