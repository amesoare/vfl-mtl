"""privacy/adaptive_dpsgd.py — DP clip-and-noise for VFL cut-layer gradients."""

from __future__ import annotations

import torch
from torch import Tensor

from fl.client import VFLClient

# Site → primary task assignment (fixed by MIMIC-III vertical split protocol)
SITE_TASK_MAP: dict[str, str] = {"A": "ihm", "B": "decomp", "C": "pheno"}

_TASKS = ("ihm", "decomp", "pheno")


class AdaptiveDPSGD:
    """Per-sample L2 clip + Gaussian noise for VFL cut-layer gradients (Abadi et al. 2016)."""

    def __init__(self, max_grad_norm: float = 1.0) -> None:
        self.max_grad_norm = max_grad_norm
        self._sigma: dict[str, float] = {}


    def set_uniform(self, sigma: float) -> None:
        """All tasks use the same noise multiplier σ."""
        for task in _TASKS:
            self._sigma[task] = sigma

    def set_stratified(
        self,
        sigma_ihm: float,
        sigma_decomp: float,
        sigma_pheno: float,
    ) -> None:
        """Per-task σ allocation: σ_IHM < σ_Decomp < σ_Pheno (clinical risk hierarchy)."""
        self._sigma = {
            "ihm":    sigma_ihm,
            "decomp": sigma_decomp,
            "pheno":  sigma_pheno,
        }


    def clip_and_noise(self, grad: Tensor, task: str) -> Tensor:
        """Clip per-row L2 norm to C/B and add Gaussian noise; returns (B, embed_dim)."""
        sigma = self._sigma.get(task, 0.0)
        if sigma == 0.0:
            return grad

        B = grad.shape[0]

        # Effective per-row sensitivity: C/B (rows are batch-averaged gradients)
        effective_C = self.max_grad_norm / B

        # Per-sample L2 norm clipping to effective_C
        norms = grad.norm(2, dim=-1, keepdim=True)                    # (B, 1)
        scale = (effective_C / (norms + 1e-8)).clamp(max=1.0)
        clipped = grad * scale                                         # (B, embed_dim)

        # Gaussian noise: std = σ·(C/B)/√B → summed-row noise std = σ·(C/B)
        noise_std = sigma * effective_C / (B ** 0.5)
        noise = torch.randn_like(clipped) * noise_std
        return clipped + noise


    @property
    def is_enabled(self) -> bool:
        """True when at least one task has σ > 0 configured."""
        return any(v > 0.0 for v in self._sigma.values())

    def sigma_for(self, task: str) -> float:
        """Return σ for the given task (0.0 if not set = no noise)."""
        return self._sigma.get(task, 0.0)

    def __repr__(self) -> str:
        return (
            f"AdaptiveDPSGD(max_grad_norm={self.max_grad_norm}, sigma={self._sigma})"
        )



class DPVFLClient(VFLClient):
    """VFLClient with DP at the cut layer: overrides receive_gradient() to apply clip_and_noise()."""

    def __init__(
        self,
        dp_mechanism: AdaptiveDPSGD,
        site: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if site not in SITE_TASK_MAP:
            raise ValueError(f"Unknown site {site!r}. Expected one of {list(SITE_TASK_MAP)}")
        self._dp = dp_mechanism
        self._task = SITE_TASK_MAP[site]

    def receive_gradient(self, grad: Tensor) -> None:
        """Apply DP noise to grad then delegate to VFLClient.receive_gradient()."""
        if self._dp.is_enabled:
            grad = self._dp.clip_and_noise(grad, self._task)
        super().receive_gradient(grad)
