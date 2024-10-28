import random
from abc import ABC
from typing import Dict, Optional, Tuple

import einops
import gpytorch
import torch
from tnp.data.base import GroundTruthPredictor
from tnp.data.gp import GPRegressionModel
from tnp.data.synthetic import SyntheticGeneratorUniformInput


class FiniteMixtureGPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
        self,
        kernels: Tuple[gpytorch.kernels.Kernel, ...],
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        self.kernels = kernels
        self.likelihood = likelihood

        self._result_cache: Optional[Dict[str, torch.Tensor]] = None

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Move devices.
        old_device = xc.device
        device = self.kernels[0].device
        xc = xc.to(device)
        yc = yc.to(device)
        xt = xt.to(device)
        if yt is not None:
            yt = yt.to(device)

        if yt is not None and self._result_cache is not None:
            # Return cached results.
            return (
                self._result_cache["mean"],
                self._result_cache["std"],
                self._result_cache["gt_loglik"],
            )

        mean_list = []
        std_list = []
        gt_loglik_list = []

        # Compute posterior.
        for i, (xc_, yc_, xt_) in enumerate(zip(xc, yc, xt)):
            inner_mll_list = []
            inner_mean_list = []
            inner_std_list = []
            inner_gt_loglik_list = []
            for kernel in self.kernels:
                gp_model = GPRegressionModel(
                    likelihood=self.likelihood,
                    kernel=kernel,
                    train_inputs=xc_,
                    train_targets=yc_[..., 0],
                )
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                    likelihood=self.likelihood, model=gp_model
                )
                mll.eval()
                gp_model.eval()
                gp_model.likelihood.eval()
                with torch.no_grad():
                    # Get marginal log likelihood for training data.
                    yc_pred_ = gp_model(xc_)
                    kernel_mll = mll(yc_pred_, yc_[..., 0])
                    inner_mll_list.append(kernel_mll)

                    # Get posterior predictive distribution for test data.
                    dist = gp_model(xt_)
                    pred_dist = gp_model.likelihood.marginal(dist)
                    if yt is not None:
                        gt_loglik = pred_dist.to_data_independent_dist().log_prob(
                            yt[i, ..., 0]
                        )
                        inner_gt_loglik_list.append(gt_loglik)

                    inner_mean_list.append(pred_dist.mean)
                    try:
                        inner_std_list.append(pred_dist.stddev)
                    except RuntimeError:
                        inner_std_list.append(
                            pred_dist.covariance_matrix.diagonal() ** 0.5
                        )

            # Comptue marginal predictions and gt_logliks.
            posterior_probs = torch.stack(inner_mll_list, dim=0).softmax(dim=0)
            means = torch.stack(inner_mean_list, dim=0)
            variances = torch.stack(inner_std_list, dim=0) ** 2
            marginal_mean = (means * posterior_probs[:, None]).sum(dim=0)
            marginal_variance = (
                (variances + means**2) * posterior_probs[:, None]
            ).sum(dim=0) - marginal_mean**2
            if yt is not None:
                marginal_gt_loglik = torch.logsumexp(
                    torch.stack(inner_gt_loglik_list, dim=0), dim=0
                )
            else:
                marginal_gt_loglik = None

            mean_list.append(marginal_mean)
            std_list.append(marginal_variance.sqrt())
            gt_loglik_list.append(marginal_gt_loglik)

        mean = torch.stack(mean_list, dim=0)
        std = torch.stack(std_list, dim=0)

        if yt is not None:
            gt_loglik = torch.stack(gt_loglik_list, dim=0)
        else:
            gt_loglik = None

        # Cache for deterministic validation batches.
        # Note yt is not specified when passing x_plot.
        if yt is not None:
            self._result_cache = {
                "mean": mean,
                "std": std,
                "gt_loglik": gt_loglik,
            }

        # Move back.
        xc = xc.to(old_device)
        yc = yc.to(old_device)
        xt = xt.to(old_device)
        if yt is not None:
            yt = yt.to(old_device)

        mean = mean.to(old_device)
        std = std.to(old_device)
        if gt_loglik is not None:
            gt_loglik = gt_loglik.to(old_device)

        return mean, std, gt_loglik

    def sample_outputs(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:

        # Sample kernel.
        kernel = random.choice(self.kernels)

        # Construct GP model.
        gp_model = GPRegressionModel(
            likelihood=self.likelihood,
            kernel=kernel,
        )
        gp_model.eval()
        gp_model.likelihood.eval()

        # Sample from prior.
        with torch.no_grad():
            dist = gp_model.forward(x)
            f = dist.sample(sample_shape=sample_shape)
            dist = gp_model.likelihood(f)
            y = dist.sample()
            return y[..., None]


class FiniteMixtureGPGenerator(ABC):
    def __init__(
        self,
        *,
        kernels: Tuple[gpytorch.kernels.Kernel, ...],
        noise_std: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernels = kernels
        self.noise_std = noise_std

    def set_up_gp(self) -> FiniteMixtureGPGroundTruthPredictor:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = self.noise_std**2
        return FiniteMixtureGPGroundTruthPredictor(
            kernels=self.kernels,
            likelihood=likelihood,
        )

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, GroundTruthPredictor]:
        gp = self.set_up_gp()
        return gp.sample_outputs(x), gp


class FiniteMixtureGPGeneratorUniformInput(
    FiniteMixtureGPGenerator, SyntheticGeneratorUniformInput
):
    pass


class FiniteMixtureGPGeneratorUniformSameInputs(FiniteMixtureGPGeneratorUniformInput):
    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        x = super().sample_inputs(nc=nc, batch_shape=torch.Size(), nt=nt)
        x = einops.repeat(x, "n d -> b n d", b=batch_shape[0])
        return x

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gt_pred = self.set_up_gp()
        sample_shape = x.shape[:-2]
        return gt_pred.sample_outputs(x[0], sample_shape=sample_shape), gt_pred
