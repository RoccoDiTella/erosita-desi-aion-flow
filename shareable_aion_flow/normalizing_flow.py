"""Conditional normalizing flow, target standardizer, and KDE prior.

A one-dimensional Zuko neural spline flow (NSF) conditioned on the attention
context predicts a full posterior over the standardized target. The KDE prior is
the unconditional baseline used to define information gain (posterior minus prior
log-likelihood).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import gaussian_kde
from torch import nn


def split_normal_log_kernel(
    residual: torch.Tensor, sig_lo: torch.Tensor, sig_hi: torch.Tensor
) -> torch.Tensor:
    """Log density of a two-piece (split) normal measurement model.

    Represents an asymmetric error bar: the observed value ``y`` around a true
    value ``t`` is split-normal with lower width ``sig_lo`` (mass where y<t) and
    upper width ``sig_hi`` (y>t). ``residual = y - t``; the branch uses ``sig_lo``
    when the observation is below the candidate true value and ``sig_hi`` above.
    Normalisation ``sqrt(2/pi)/(sig_lo+sig_hi)`` makes it integrate to 1, so the
    symmetric limit ``sig_lo==sig_hi`` reduces to a plain Gaussian.
    """
    sig = torch.where(residual < 0, sig_lo, sig_hi)
    log_norm = 0.5 * math.log(2.0 / math.pi) - torch.log(sig_lo + sig_hi)
    return -0.5 * (residual / sig) ** 2 + log_norm


def sample_split_normal(
    sig_lo: torch.Tensor, sig_hi: torch.Tensor, max_sigma: float = 1.5
) -> torch.Tensor:
    """Draw ``eps ~ split-normal(mode 0; widths sig_lo/sig_hi)``, truncated at
    ``max_sigma`` per side.

    Used by ``--error-mode inject``: pick the lower half with probability
    ``sig_lo/(sig_lo+sig_hi)`` (truncation removes the same mass fraction from
    each side, so the side odds are unchanged), then draw the magnitude from a
    truncated half-normal via the inverse CDF:
    ``|eps| = sigma * sqrt(2) * erfinv(u * erf(max_sigma/sqrt(2)))``.
    Truncation keeps single draws out of the far tail (relevant for the
    huge-sigma HR rows) at the cost of a slightly narrower effective kernel.
    """
    u = torch.rand_like(sig_lo)
    cap = math.erf(max_sigma / math.sqrt(2.0))
    z = math.sqrt(2.0) * torch.erfinv(u * cap)
    take_lower = torch.rand_like(sig_lo) < (sig_lo / (sig_lo + sig_hi))
    return torch.where(take_lower, -sig_lo * z, sig_hi * z)


def convolve_logprob(
    logp_fn,
    y: torch.Tensor,
    sig_lo: torch.Tensor,
    sig_hi: torch.Tensor,
    context: torch.Tensor,
    n_nodes: int = 41,
    span: float = 5.0,
) -> torch.Tensor:
    """Error-aware log-likelihood ``log ∫ p(t|x)·K(y|t) dt`` via 1-D quadrature.

    ``logp_fn(t_flat, context_flat) -> [.]`` is the (intrinsic) flow log-density;
    ``y, sig_lo, sig_hi`` are per-sample observation + asymmetric errors in the
    SAME (standardized) units the flow models. Nodes span ``y ± span·max(sig)`` so
    a tiny error recovers the plain ``logp_fn(y)`` (kernel → delta) and a large
    error correctly down-weights the source. Returns one value per sample.
    """
    if not (y.shape == sig_lo.shape == sig_hi.shape) or y.dim() != 1:
        raise ValueError("y, sig_lo, sig_hi must be 1-D tensors of the same shape.")
    sig_lo = sig_lo.clamp_min(1e-6)
    sig_hi = sig_hi.clamp_min(1e-6)
    sig_ref = torch.maximum(sig_lo, sig_hi)  # [B]
    z = torch.linspace(-span, span, n_nodes, device=y.device, dtype=y.dtype)  # [N]
    t = y.unsqueeze(1) + z.unsqueeze(0) * sig_ref.unsqueeze(1)  # [B, N]
    residual = y.unsqueeze(1) - t  # [B, N]
    log_kernel = split_normal_log_kernel(residual, sig_lo.unsqueeze(1), sig_hi.unsqueeze(1))
    dz = (2.0 * span) / (n_nodes - 1)
    log_dt = torch.log(sig_ref * dz)  # [B] rectangular-rule node width
    context_rep = context.repeat_interleave(n_nodes, dim=0)  # [B*N, C]
    logp = logp_fn(t.reshape(-1), context_rep).reshape(y.shape[0], n_nodes)  # [B, N]
    integrand = logp + log_kernel + log_dt.unsqueeze(1)
    return torch.logsumexp(integrand, dim=1)  # [B]


@dataclass(frozen=True)
class TargetStandardizer:
    """Frozen z-score standardizer for the one-dimensional regression target."""

    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> TargetStandardizer:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 2:
            raise ValueError("At least two finite target values are required to fit target standardization.")
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-12:
            raise ValueError("Target values must have finite mean and non-zero variance.")
        return cls(mean=mean, std=float(std + 1e-8))

    def state_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, float]) -> TargetStandardizer:
        return cls(mean=float(state["mean"]), std=float(state["std"]))

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse_numpy(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean


class KDEPrior:
    """Unconditional Gaussian-KDE baseline over the standardized target.

    Fit on the training targets only, it stands in for "predict the marginal" and
    anchors the information-gain metric: IG = posterior log-likelihood minus this
    prior's log-likelihood, in nats.
    """

    def __init__(
        self,
        standardized_train_values: np.ndarray,
        bw_method: str | float = "scott",
        metadata: dict[str, object] | None = None,
    ) -> None:
        values = np.asarray(standardized_train_values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 2:
            raise ValueError("At least two finite standardized target values are required for the KDE prior.")
        if not np.isfinite(values).all() or float(np.std(values)) <= 1e-12:
            raise ValueError("KDE prior values must be finite with non-zero variance.")
        self.values = values
        self.bw_method = bw_method
        self.metadata = metadata or {}
        self.kde = gaussian_kde(values, bw_method=bw_method)

    def log_prob_numpy(self, standardized_values: np.ndarray) -> np.ndarray:
        density = self.kde.evaluate(np.asarray(standardized_values, dtype=np.float64))
        return np.log(np.clip(density, 1e-300, None))

    def log_prob_tensor(self, standardized_values: torch.Tensor) -> torch.Tensor:
        device = standardized_values.device
        log_prob = self.log_prob_numpy(standardized_values.detach().cpu().numpy())
        return torch.from_numpy(log_prob.astype(np.float32)).to(device)

    def save(self, path: Path, metadata: dict[str, object] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload_metadata = self.metadata.copy()
        if metadata is not None:
            payload_metadata.update(metadata)
        np.savez_compressed(
            path,
            values=self.values,
            bw_method=np.asarray(str(self.bw_method)),
            metadata_json=np.asarray(json.dumps(payload_metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> KDEPrior:
        payload = np.load(path, allow_pickle=False)
        bw_raw = str(payload["bw_method"].item())
        bw_method: str | float = float(bw_raw) if bw_raw.replace(".", "", 1).isdigit() else bw_raw
        metadata = {}
        if "metadata_json" in payload.files:
            metadata = json.loads(str(payload["metadata_json"].item()))
        return cls(payload["values"], bw_method=bw_method, metadata=metadata)


class ConditionalNSFFlow(nn.Module):
    """Conditional 1-D neural spline flow (Zuko NSF) over the standardized target.

    Conditioned on the attention context, it models the full posterior
    ``p(target | inputs)`` rather than a point estimate. ``zuko`` is imported lazily
    so the rest of the package works without it installed.
    """

    def __init__(
        self,
        *,
        context_dim: int = 256,
        transforms: int = 8,
        hidden_features: tuple[int, ...] = (256, 256),
        features: int = 1,
    ) -> None:
        super().__init__()
        try:
            from zuko.flows import NSF
        except ImportError as exc:
            raise ImportError("Install zuko>=1.5.0 to train or evaluate the normalizing flow.") from exc
        self.context_dim = context_dim
        self.features = int(features)
        self.flow = NSF(
            features=self.features,
            context=context_dim,
            transforms=transforms,
            hidden_features=list(hidden_features),
        )

    def distribution(self, context: torch.Tensor):
        if context.dim() != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(f"Expected context shape [B, {self.context_dim}], got {tuple(context.shape)}.")
        return self.flow(context)

    def log_prob(self, standardized_target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if standardized_target.dim() == 1 and self.features == 1:
            target = standardized_target.unsqueeze(-1)
        elif standardized_target.dim() == 2 and standardized_target.shape[-1] == self.features:
            target = standardized_target
        else:
            raise ValueError(
                f"Expected target shape [B] or [B, {self.features}], got {tuple(standardized_target.shape)}."
            )
        if target.shape[0] != context.shape[0]:
            raise ValueError(
                f"Target batch size {target.shape[0]} does not match context batch size {context.shape[0]}."
            )
        return self.distribution(context).log_prob(target).reshape(-1)

    def log_prob_draws(self, y_draws: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Log-density of k perturbed targets per source in ONE broadcasted call.

        ``y_draws``: [k, B] (features=1) or [k, B, D]; ``context``: [B, ctx].
        The context-conditioned distribution is built once; the k draws are
        evaluated under it in a single batched pass (for 1-D flows the spline
        parameters are point-independent, so extra draws are ~free).
        Returns [k, B].
        """
        if y_draws.dim() == 2 and self.features == 1:
            x = y_draws.unsqueeze(-1)
        elif y_draws.dim() == 3 and y_draws.shape[-1] == self.features:
            x = y_draws
        else:
            raise ValueError(
                f"Expected draws [k, B] or [k, B, {self.features}], got {tuple(y_draws.shape)}."
            )
        if x.shape[1] != context.shape[0]:
            raise ValueError(
                f"Draws batch dim {x.shape[1]} does not match context batch {context.shape[0]}."
            )
        return self.distribution(context).log_prob(x)

    def log_prob_convolved(
        self,
        standardized_target: torch.Tensor,
        sig_lo: torch.Tensor,
        sig_hi: torch.Tensor,
        context: torch.Tensor,
        n_nodes: int = 41,
        span: float = 5.0,
    ) -> torch.Tensor:
        """Error-aware log-likelihood of a noisy standardized observation.

        Convolves the intrinsic flow density with a split-normal measurement
        kernel (``sig_lo``/``sig_hi`` are the standardized asymmetric errors).
        Deconvolves measurement noise; the ``sig->0`` limit equals ``log_prob``.
        """
        y = standardized_target.reshape(-1)
        return convolve_logprob(self.log_prob, y, sig_lo.reshape(-1), sig_hi.reshape(-1), context, n_nodes, span)

    def sample(self, context: torch.Tensor, num_samples: int) -> torch.Tensor:
        samples = self.distribution(context).sample((int(num_samples),))
        return samples.squeeze(-1)
