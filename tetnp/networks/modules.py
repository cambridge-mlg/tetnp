from typing import Tuple

import numpy as np
import torch
from torch import nn


class ModuleOnPreSpecifiedDomain(nn.Module):
    def __init__(
        self,
        *,
        module: nn.Module,
        x_range: Tuple[Tuple[float, float]],
        default_val: float = 0.0,
    ):
        super().__init__()

        self.module = module
        self.x_range = torch.as_tensor(x_range)
        self.default_val = default_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)

        in_domain = torch.all(x < self.x_range[..., 1].to(out), dim=-1) & torch.all(
            x > self.x_range[..., 0].to(out), dim=-1
        )
        out = torch.where(
            in_domain[..., None], out, torch.ones_like(out) * self.default_val
        )

        return out


class ModuleOnFourierExpandedInput(nn.Module):
    def __init__(
        self,
        *,
        module: nn.Module,
        x_range: Tuple[Tuple[float, float]],
        num_wavelengths: int = 10,
    ):
        super().__init__()

        self.module = module
        self.x_range = torch.as_tensor(x_range)
        max_wavelengths = self.x_range[..., 1] - self.x_range[..., 0]
        min_wavelengths = 10 ** (torch.log10(max_wavelengths) / num_wavelengths)

        # Get wavelengths. Shape (num_wavelengths // 2, x_dim).
        self.wavelengths = torch.stack(
            [
                torch.logspace(
                    torch.log10(min_wavelengths[dim]),
                    torch.log10(max_wavelengths[dim]),
                    num_wavelengths // 2,
                    base=10,
                )
                for dim in range(len(x_range))
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform expansion with float64 to avoid numerical instability.
        old_dtype = x.dtype
        x = x.double()

        wavelengths = self.wavelengths.to(x)
        prod = x * 2 * np.pi / wavelengths
        fourier = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        fourier = fourier.to(old_dtype)

        in_domain = torch.all(x < self.x_range[..., 1].to(fourier), dim=-1) & torch.all(
            x > self.x_range[..., 0].to(fourier), dim=-1
        )
        out = torch.where(in_domain[..., None], fourier, torch.zeros_like(fourier))

        return out
