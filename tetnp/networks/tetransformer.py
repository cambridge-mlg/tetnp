import copy
import warnings
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from .teattention_layers import (
    MultiHeadCrossTEAttentionLayer,
    MultiHeadSelfTEAttentionLayer,
)
from .tept_init import PseudoTokenInitialiser


class TETransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: MultiHeadSelfTEAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)

    @check_shapes(
        "z: [m, n, d]", "x: [m, n, dx]", "mask: [m, n, n]", "return: [m, n, d]"
    )
    def forward(
        self, z: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            z, x = layer(z, x, mask)

        return z


class TETNPTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossTEAttentionLayer,
        mhsa_layer: Optional[MultiHeadSelfTEAttentionLayer] = None,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = (
            self.mhca_layers
            if mhsa_layer is None
            else _get_clones(mhsa_layer, num_layers)
        )

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
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            if isinstance(mhsa_layer, MultiHeadSelfTEAttentionLayer):
                zc, xc = mhsa_layer(zc, xc)
            elif isinstance(mhsa_layer, MultiHeadCrossTEAttentionLayer):
                zc, xc = mhsa_layer(zc, zc, xc, xc)
            else:
                raise TypeError("Unknown layer type.")

            zt, xt = mhca_layer(zt, zc, xt, xc)

        return zt


class BaseTEPerceiverEncoder(nn.Module, ABC):
    def __init__(
        self,
        dim: int,
        num_latents: int,
        mhsa_layer: MultiHeadSelfTEAttentionLayer,
        mhca_ctoq_layer: MultiHeadCrossTEAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossTEAttentionLayer,
        num_layers: int,
        pseudo_token_initialiser: Optional[PseudoTokenInitialiser] = None,
    ):
        super().__init__()

        # Initialise pseudo-tokens and pseudo-locations.
        self.embed_dim = mhsa_layer.embed_dim
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, self.embed_dim))
        self.latent_inputs = nn.Parameter(torch.randn(num_latents, dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)

        if pseudo_token_initialiser is None:
            self.pseudo_token_initialiser = lambda zq, zc, xq, xc: (
                zq,
                xq + xc.mean(-2, keepdim=True),
            )
        else:
            self.pseudo_token_initialiser = pseudo_token_initialiser


class TEPerceiverEncoder(BaseTEPerceiverEncoder):
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

        # Initialise pseudo-tokens.
        zq, xq = self.pseudo_token_initialiser(zq, zc, xq, xc)
        for mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer in zip(
            self.mhsa_layers, self.mhca_ctoq_layers, self.mhca_qtot_layers
        ):
            zq, xq = mhca_ctoq_layer(zq, zc, xq, xc)
            zq, xq = mhsa_layer(zq, xq)
            zt, xt = mhca_qtot_layer(zt, zq, xt, xq)

        return zt


class BaseTEISTEncoder(nn.Module, ABC):
    def __init__(
        self,
        dim: int,
        num_latents: int,
        mhca_ctoq_layer: MultiHeadSelfTEAttentionLayer,
        mhca_qtoc_layer: MultiHeadCrossTEAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossTEAttentionLayer,
        num_layers: int,
        pseudo_token_initialiser: Optional[PseudoTokenInitialiser] = None,
    ):
        super().__init__()

        # Initialise pseudo-tokens and pseudo-locations.
        embed_dim = mhca_ctoq_layer.embed_dim
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, embed_dim))
        self.latent_inputs = nn.Parameter(torch.randn(num_latents, dim))

        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtoc_layers = _get_clones(mhca_qtoc_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)

        if pseudo_token_initialiser is None:
            self.pseudo_token_initialiser = lambda zq, zc, xq, xc: (
                zq,
                xq + xc.mean(-2, keepdim=True),
            )
        else:
            self.pseudo_token_initialiser = pseudo_token_initialiser


class TEISTEncoder(BaseTEISTEncoder):
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

        # Initialise pseudo-tokens.
        zq, xq = self.pseudo_token_initialiser(zq, zc, xq, xc)

        for mhca_ctoq_layer, mhca_qtoc_layer, mhca_qtot_layer in zip(
            self.mhca_ctoq_layers, self.mhca_qtoc_layers, self.mhca_qtot_layers
        ):
            zq, xq = mhca_ctoq_layer(zq, zc, xq, xc)
            zc, xc = mhca_qtoc_layer(zc, zq, xc, xq)
            zt, xt = mhca_qtot_layer(zt, zq, xt, xq)

        return zt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
