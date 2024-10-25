from typing import Union

import torch
from check_shapes import check_shapes
from tnp.models.base import ConditionalNeuralProcess
from tnp.models.tnp import TNPDecoder
from tnp.utils.helpers import preprocess_observations
from torch import nn

from ..networks.tetransformer import (
    TEISTEncoder,
    TEPerceiverEncoder,
    TETNPTransformerEncoder,
)


class TETNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[
            TETNPTransformerEncoder, TEPerceiverEncoder, TEISTEncoder
        ],
        y_encoder: nn.Module,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        zc = self.y_encoder(yc)
        zt = self.y_encoder(yt)

        zt = self.transformer_encoder(zc, zt, xc, xt)
        return zt


class TETNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: TETNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
