from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import gaussian_kde
from torch import nn


@dataclass(frozen=True)
class TargetStandardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "TargetStandardizer":
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
    def from_state_dict(cls, state: dict[str, float]) -> "TargetStandardizer":
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
    """One-dimensional KDE prior on standardized targets."""

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
    def load(cls, path: Path) -> "KDEPrior":
        payload = np.load(path, allow_pickle=False)
        bw_raw = str(payload["bw_method"].item())
        bw_method: str | float = float(bw_raw) if bw_raw.replace(".", "", 1).isdigit() else bw_raw
        metadata = {}
        if "metadata_json" in payload.files:
            metadata = json.loads(str(payload["metadata_json"].item()))
        return cls(payload["values"], bw_method=bw_method, metadata=metadata)


class ConditionalNSFFlow(nn.Module):
    """Conditional one-dimensional neural spline flow."""

    def __init__(
        self,
        *,
        context_dim: int = 256,
        transforms: int = 8,
        hidden_features: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        try:
            from zuko.flows import NSF
        except ImportError as exc:
            raise ImportError("Install zuko>=1.5.0 to train or evaluate the normalizing flow.") from exc
        self.context_dim = context_dim
        self.flow = NSF(
            features=1,
            context=context_dim,
            transforms=transforms,
            hidden_features=list(hidden_features),
        )

    def distribution(self, context: torch.Tensor):
        if context.dim() != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(f"Expected context shape [B, {self.context_dim}], got {tuple(context.shape)}.")
        return self.flow(context)

    def log_prob(self, standardized_target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if standardized_target.dim() == 1:
            target = standardized_target.unsqueeze(-1)
        elif standardized_target.dim() == 2 and standardized_target.shape[-1] == 1:
            target = standardized_target
        else:
            raise ValueError(f"Expected target shape [B] or [B, 1], got {tuple(standardized_target.shape)}.")
        if target.shape[0] != context.shape[0]:
            raise ValueError(f"Target batch size {target.shape[0]} does not match context batch size {context.shape[0]}.")
        return self.distribution(context).log_prob(target).reshape(-1)

    def sample(self, context: torch.Tensor, num_samples: int) -> torch.Tensor:
        samples = self.distribution(context).sample((int(num_samples),))
        return samples.squeeze(-1)
