from abc import ABC
from functools import partial
from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from .teattention import (
    BaseMultiHeadTEAttention,
    MultiHeadCrossTEAttention,
    MultiHeadSelfTEAttention,
    MultiHeadTEAttention,
)


class MultiHeadTEAttentionLayer(nn.Module, ABC):
    def __init__(
        self,
        embed_dim: int,
        attention: Union[MultiHeadTEAttention, partial[BaseMultiHeadTEAttention]],
        feedforward_dim: Optional[int] = None,
        p_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm_first: bool = False,
        **kwargs
    ):
        super().__init__()
        feedforward_dim = embed_dim if feedforward_dim is None else feedforward_dim

        self.embed_dim = embed_dim
        self.attn = attention(
            **kwargs,
        )

        # Feedforward model.
        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation,
            nn.Dropout(p_dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(p_dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_first = norm_first

        self.attn_dropout = nn.Dropout(p_dropout)


class MultiHeadSelfTEAttentionLayer(MultiHeadTEAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(MultiHeadSelfTEAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes(
        "z: [m, n, dz]",
        "x: [m, n, dx]",
        "mask: [m, n, n]",
        "return[0]: [m, n, dz]",
        "return[1]: [m, n, dx]",
    )
    def attn_block(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z, x = self.attn(z, x, mask=mask)
        return self.attn_dropout(z), x

    @check_shapes(
        "z: [m, n, dz]",
        "x: [m, n, dx]",
        "mask: [m, n, n]",
        "return[0]: [m, n, dz]",
        "return[1]: [m, n, dx]",
    )
    def forward(
        self, z: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_first:
            z_update, x = self.attn_block(self.norm1(z), x, mask)
            z = z + z_update
            z = z + self.ff_block(self.norm2(z))
        else:
            z_update, x = self.attn_block(z, x, mask)
            z = self.norm1(z + z_update)
            z = z + self.ff_block(self.norm2(z))

        return z, x


class MultiHeadCrossTEAttentionLayer(MultiHeadTEAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(MultiHeadCrossTEAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes(
        "zq: [m, nq, dz]",
        "zk: [m, nk, dz]",
        "xq: [m, nq, dx]",
        "xk: [m, nk, dx]",
        "mask: [m, nq, nk]",
        "return[0]: [m, nq, dz]",
        "return[1]: [m, nq, dx]",
    )
    def attn_block(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        xq: torch.Tensor,
        xk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        zq, xq = self.attn(zq, zk, xq, xk, mask=mask)
        return self.attn_dropout(zq), xq

    @check_shapes(
        "zq: [m, nq, dz]",
        "zk: [m, nk, dz]",
        "xq: [m, nq, dx]",
        "xk: [m, nk, dx]",
        "mask: [m, nq, nk]",
        "return[0]: [m, nq, dz]",
        "return[1]: [m, nq, dx]",
    )
    def forward(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        xq: torch.Tensor,
        xk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.norm_first:
            zq_update, xq = self.attn_block(
                self.norm1(zq), self.norm1(zk), xq, xk, mask
            )
            zq = zq + zq_update
            zq = zq + self.ff_block(self.norm2(zq))
        else:
            zq_update, xq = self.attn_block(zq, zk, xq, xk, mask)
            zq = self.norm1(zq + zq_update)
            zq = zq + self.ff_block(self.norm2(zq))

        return zq, xq
