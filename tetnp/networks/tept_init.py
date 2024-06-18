from typing import Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn


class PseudoTokenInitialiser(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == embed_dim)

        self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(p_dropout) if project_out else nn.Identity(),
        )

        # Pre-softmax weighting of location attention weights.
        self.raw_head_weights = nn.Parameter(torch.ones((num_heads,)))

    @property
    def head_weights(self):
        return self.raw_head_weights.softmax(dim=0)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "tq: [m, nq, dt]",
        "tkv: [m, nkv, dt]",
        "return[0]: [m, nq, dx]",
        "return[1]: [m, nq, dt]",
    )
    def forward(
        self,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        tq: torch.Tensor,
        tkv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.to_q(xq)
        k = self.to_k(xkv)

        # Each of shape (m, num_heads, n, head_dim).
        q, k = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q, k),
        )

        tkv_ = einops.repeat(tkv, "m n d -> m h n d", h=self.num_heads)

        tq_update = (
            nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                q, k, tkv_, scale=self.scale
            )
        )

        # Now do weighted sum over heads.
        tq_update = einops.rearrange(tq_update, "m h n d -> m n d h")
        tq_update = tq_update @ self.head_weights

        tq_out = tq + tq_update

        return xq, tq_out
