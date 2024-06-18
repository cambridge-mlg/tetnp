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
        "x: [m, n, d]", "t: [m, n, dt]", "mask: [m, n, n]", "return: [m, n, d]"
    )
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x, t = layer(x, t, mask)

        return x


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
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "tc: [m, nc, dt]",
        "tt: [m, nt, dt]",
        "mask: [m, nt, nc]",
        "return: [m, nt, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        xt: torch.Tensor,
        tc: torch.Tensor,
        tt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            if isinstance(mhsa_layer, MultiHeadSelfTEAttentionLayer):
                xc, tc = mhsa_layer(xc, tc)
            elif isinstance(mhsa_layer, MultiHeadCrossTEAttentionLayer):
                xc, tc = mhsa_layer(xc, xc, tc, tc)
            else:
                raise TypeError("Unknown layer type.")

            xt, tt = mhca_layer(xt, xc, tt, tc)

        return xt


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

        assert mhsa_layer.embed_dim == mhca_ctoq_layer.embed_dim, "embed_dim mismatch."
        assert mhsa_layer.embed_dim == mhca_qtot_layer.embed_dim, "embed_dim mismatch."

        self.embed_dim = mhsa_layer.embed_dim
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, self.embed_dim))
        self.latent_inputs = nn.Parameter(torch.randn(num_latents, dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)

        if pseudo_token_initialiser is None:
            self.pseudo_token_initialiser = lambda xq, xc, tq, tc: (
                xq,
                tq + tc.mean(-2, keepdim=True),
            )
        else:
            self.pseudo_token_initialiser = pseudo_token_initialiser


class TEPerceiverEncoder(BaseTEPerceiverEncoder):
    tq_cache: Optional[torch.Tensor] = None

    @check_shapes(
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "tc: [m, nc, dt]",
        "tt: [m, nt, dt]",
        "mask: [m, nq, n]",
        "return: [m, nq, dx]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        xt: torch.Tensor,
        tc: torch.Tensor,
        tt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        xq = einops.repeat(self.latent_tokens, "l e -> m l e", m=xc.shape[0])
        tq = einops.repeat(self.latent_inputs, "l d -> m l d", m=xc.shape[0])

        # Now initialise pseudo-tokens.
        xq, tq = self.pseudo_token_initialiser(xq, xc, tq, tc)

        # Add mean of context input-locations to make translation equivariant.
        for mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer in zip(
            self.mhsa_layers, self.mhca_ctoq_layers, self.mhca_qtot_layers
        ):
            xq, tq = mhca_ctoq_layer(xq, xc, tq, tc)
            xq, tq = mhsa_layer(xq, tq)
            xt, tt = mhca_qtot_layer(xt, xq, tt, tq)

        return xt


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

        assert (
            mhca_ctoq_layer.embed_dim == mhca_qtoc_layer.embed_dim
        ), "embed_dim mismatch."
        assert (
            mhca_ctoq_layer.embed_dim == mhca_qtot_layer.embed_dim
        ), "embed_dim mismatch."

        embed_dim = mhca_ctoq_layer.embed_dim
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, embed_dim))
        self.latent_inputs = nn.Parameter(torch.randn(num_latents, dim))

        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtoc_layers = _get_clones(mhca_qtoc_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)

        if pseudo_token_initialiser is None:
            self.pseudo_token_initialiser = lambda xq, xc, tq, tc: (
                xq,
                tq + tc.mean(-2, keepdim=True),
            )
        else:
            self.pseudo_token_initialiser = pseudo_token_initialiser


class TEISTEncoder(BaseTEISTEncoder):
    @check_shapes(
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "tc: [m, nc, dt]",
        "tt: [m, nt, dt]",
        "mask: [m, nq, n]",
        "return: [m, nq, dx]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        xt: torch.Tensor,
        tc: torch.Tensor,
        tt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        xq = einops.repeat(self.latent_tokens, "l e -> m l e", m=xc.shape[0])
        tq = einops.repeat(self.latent_inputs, "l d -> m l d", m=xc.shape[0])

        # Now initialise pseudo-tokens.
        xq, tq = self.pseudo_token_initialiser(xq, xc, tq, tc)

        # Cache for plotting.
        for mhca_ctoq_layer, mhca_qtoc_layer, mhca_qtot_layer in zip(
            self.mhca_ctoq_layers, self.mhca_qtoc_layers, self.mhca_qtot_layers
        ):
            xq, tq = mhca_ctoq_layer(xq, xc, tq, tc)
            xc, tc = mhca_qtoc_layer(xc, xq, tc, tq)
            xt, tt = mhca_qtot_layer(xt, xq, tt, tq)

        return xt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
