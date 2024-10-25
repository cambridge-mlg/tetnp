from abc import ABC
from typing import Callable, Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.group_actions import translation


class BaseMultiHeadTEAttention(nn.Module, ABC):
    def __init__(
        self,
        qk_dim: int,
        v_dim: int,
        num_heads: int,
        head_dim: int,
        kernel: nn.Module,
        p_dropout: float = 0.0,
        group_action: Callable = translation,
        phi: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == v_dim)

        self.kernel = kernel
        self.to_k = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_q = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(v_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, v_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

        # Group action on inputs prior to kernel.
        self.group_action = group_action

        # Additional transformation on spatio-temporal locations.
        self.phi = phi

    @check_shapes(
        "zq: [m, nq, dz]",
        "zk: [m, nkv, dz]",
        "zv: [m, nkv, dz]",
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, dz]",
        "return[1]: [m, nq, dx]",
    )
    def propagate(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes multi-head translation equivariant attention.

        Args:
            zq (torch.Tensor): Query token.
            zk (torch.Tensor): Key token.
            zv (torch.Tensor): Value token.
            xq (torch.Tensor): Query input locations.
            xkv (torch.Tensor): Key input locations.
            mask (Optional[torch.Tensor], optional): Query-key mask. Defaults to None.

        Returns:
            torch.Tensor: Output of attention mechanism.
        """
        # Compute output of group action.
        # (m, nq, nkv, dx).
        diff = self.group_action(xq, xkv)

        # Compute token attention.
        q = self.to_q(zq)
        k = self.to_k(zk)
        v = self.to_v(zv)

        # Each of shape (m, {num_heads, qk_dim}, n, head_dim).
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q, k, v),
        )

        # (m, h, nq, nk).
        token_dots = (q @ k.transpose(-1, -2)) * self.scale
        token_dots = einops.rearrange(token_dots, "m h nq nk -> m nq nk h")
        kernel_input = torch.cat((diff, token_dots), dim=-1)
        dots = self.kernel(kernel_input)
        dots = einops.rearrange(dots, "m nq nk h -> m h nq nk")

        if mask is not None:
            mask = einops.repeat(mask, "m n p -> m h n p", h=self.num_heads)
            dots = torch.masked_fill(dots, mask, -float("inf"))

        # (m, num_heads, nq, nk).
        attn = dots.softmax(dim=-1)

        out = attn @ v
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # Also update spatio-temporal locations if necessary.
        if self.phi:
            phi_input = einops.rearrange(attn, "m h n p -> m n p h")
            x_dots = self.phi(phi_input)
            xq_new = xq + (diff * x_dots).mean(-2)
        else:
            xq_new = xq

        return out, xq_new


class MultiHeadTEAttention(BaseMultiHeadTEAttention):
    @check_shapes(
        "zq: [m, nq, dz]",
        "zk: [m, nkv, dz]",
        "zv: [m, nkv, dz]",
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, dz]",
        "return[1]: [m, nq, dx]",
    )
    def forward(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().propagate(zq, zk, zv, xq, xkv, mask)


class MultiHeadSelfTEAttention(BaseMultiHeadTEAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes(
        "z: [m, n, dz]",
        "x: [m, n, dx]",
        "mask: [m, n, n]",
        "return[0]: [m, n, dz]",
        "return[1]: [m, n, dx]",
    )
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().propagate(z, z, z, x, x, mask)


class MultiHeadCrossTEAttention(BaseMultiHeadTEAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().propagate(zq, zk, zk, xq, xk, mask)
