"""
fl/server.py — VFL server: aggregation, loss, and gradient distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.mmoe import MMoEServer


_BCE = nn.BCELoss()


def _weighted_bce(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """BCE with positive-sample upweighting on Sigmoid outputs. weight = pos_weight·y + (1−y)."""
    w = pos_weight * target + (1.0 - target)
    return nn.functional.binary_cross_entropy(pred, target, weight=w)


class VFLServer:

    SITES = ("A", "B", "C")

    TASKS = ("ihm", "decomp", "pheno")

    def __init__(
        self,
        embed_dim: int = 64,
        num_experts: int = 4,
        expert_hidden: int = 128,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
        task_weights: dict[str, float] | None = None,
        n_sites: int = 3,
        decomp_pos_weight: float = 1.0,
        use_mmoe: bool = True,
        uniform_gating: bool = False,
        uncertainty_weighting: bool = False,
        task_types: dict | None = None,
    ):
        self.device             = torch.device(device)
        self.embed_dim          = embed_dim
        self.SITES              = self.__class__.SITES[:n_sites]
        self.decomp_pos_weight  = decomp_pos_weight
        self.uncertainty_weighting = uncertainty_weighting
        self._task_types        = task_types or {}

        self.model = MMoEServer(
            input_dim=n_sites * embed_dim,
            num_experts=num_experts,
            expert_hidden=expert_hidden,
            expert_out=embed_dim,
            use_mmoe=use_mmoe,
            uniform_gating=uniform_gating,
            task_types=task_types,
        ).to(self.device)

        # Kendall et al. (2018): learn log(σ_i²) per task, init=0 → σ_i=1 at start.
        # Optimised jointly with MMoE via the same Adam instance.
        if uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                t: nn.Parameter(torch.zeros(1, device=self.device))
                for t in self.TASKS
            })
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.log_vars.parameters()), lr=lr
            )
        else:
            self.log_vars  = None
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.task_weights = task_weights or {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0}

        # Stored between aggregate_embeddings() and get_embedding_gradients()
        self._concat_embedding: Tensor | None = None

    def aggregate_embeddings(self, embeddings: dict[str, Tensor]) -> Tensor:
        """Concatenate per-site embeddings → (B, n_sites×embed_dim). Stored for gradient slicing."""
        parts = [embeddings[s].to(self.device) for s in self.SITES]
        self._concat_embedding = torch.cat(parts, dim=-1)  # (B, 192)
        self._concat_embedding.retain_grad()  # non-leaf: opt in to keep .grad after backward
        return self._concat_embedding

    def forward_and_loss(
        self,
        labels: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run MMoEServer and return (total_loss, per-task loss dict)."""
        assert self._concat_embedding is not None, "call aggregate_embeddings() first"

        self.model.train()
        preds = self.model(self._concat_embedding)  # dict[str, Tensor]

        task_losses = {}
        for t, pred in preds.items():
            y  = labels[t].to(self.device).float()
            tt = self._task_types.get(t, "binary")
            if tt == "regression":
                task_losses[t] = nn.functional.mse_loss(pred.squeeze(-1), y)
            elif tt == "multilabel":
                task_losses[t] = _BCE(pred, y)
            elif t == "decomp":
                task_losses[t] = _weighted_bce(pred.squeeze(-1), y, self.decomp_pos_weight)
            else:
                task_losses[t] = _BCE(pred.squeeze(-1), y)

        if self.uncertainty_weighting:
            # Kendall et al. (2018): L = Σ_i [ exp(-s_i)/2 · L_i + s_i/2 ]
            # where s_i = log(σ_i²). Precision exp(-s_i) down-weights high-variance tasks;
            # s_i/2 regularises to prevent σ_i → ∞.
            # Only include tasks with weight > 0 so zero-weight tasks are truly inactive.
            total_loss = sum(
                0.5 * torch.exp(-self.log_vars[t]) * loss + 0.5 * self.log_vars[t]
                for t, loss in task_losses.items()
                if self.task_weights.get(t, 1.0) > 0
            )
        else:
            total_loss = sum(
                self.task_weights[t] * loss for t, loss in task_losses.items()
            )
        return total_loss, task_losses

    def backward_and_step(self, total_loss: Tensor) -> None:
        """Backprop through MMoE and update server weights."""
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def get_embedding_gradients(self) -> dict[str, Tensor]:
        """Slice concatenated-embedding grad into per-site pieces after backward_and_step()."""
        assert self._concat_embedding is not None and \
               self._concat_embedding.grad is not None, \
            "call backward_and_step() before get_embedding_gradients()"

        grad = self._concat_embedding.grad  # (B, 192)
        slices = grad.split(self.embed_dim, dim=-1)  # three (B, 64) tensors
        return dict(zip(self.SITES, slices))

    def compute_task_gradient_similarity(self, labels: dict[str, Tensor]) -> dict[str, float]:
        """Pairwise cosine similarity between per-task gradients at shared expert parameters (Yu et al. 2020)."""
        assert self._concat_embedding is not None, "call aggregate_embeddings() first"
        emb = self._concat_embedding.detach()

        if self.model.use_mmoe:
            shared_params = (
                list(self.model.mmoe.experts.parameters())
                + list(self.model.mmoe.gates.parameters())
            )
        else:
            shared_params = list(self.model.shared_bottom.parameters())

        task_grads: dict[str, Tensor | None] = {}
        self.model.train()

        for task in self.TASKS:
            if self.task_weights.get(task, 1.0) == 0.0:
                task_grads[task] = None
                continue

            self.model.zero_grad()
            preds = self.model(emb)

            if task == "ihm":
                loss = _BCE(preds["ihm"].squeeze(-1), labels["ihm"].to(self.device))
            elif task == "decomp":
                if self._task_types.get("decomp") == "regression":
                    loss = nn.functional.mse_loss(
                        preds["decomp"].squeeze(-1),
                        labels["decomp"].to(self.device).float(),
                    )
                else:
                    loss = _weighted_bce(
                        preds["decomp"].squeeze(-1),
                        labels["decomp"].to(self.device).float(),
                        self.decomp_pos_weight,
                    )
            else:
                loss = _BCE(preds["pheno"], labels["pheno"].to(self.device))

            loss.backward()

            g_parts = [p.grad.detach().flatten() for p in shared_params if p.grad is not None]
            task_grads[task] = torch.cat(g_parts) if g_parts else None

        self.model.zero_grad()

        def _cos(a: Tensor | None, b: Tensor | None) -> float:
            if a is None or b is None:
                return float("nan")
            return float(
                torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            )

        return {
            "grad_sim_ihm_decomp":   _cos(task_grads.get("ihm"),   task_grads.get("decomp")),
            "grad_sim_ihm_pheno":    _cos(task_grads.get("ihm"),   task_grads.get("pheno")),
            "grad_sim_decomp_pheno": _cos(task_grads.get("decomp"), task_grads.get("pheno")),
        }

    @torch.no_grad()
    def predict(self, embeddings: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run inference (no gradient tracking); returns per-task prediction tensors."""
        self.model.eval()
        concat = torch.cat(
            [embeddings[s].to(self.device) for s in self.SITES], dim=-1
        )
        return self.model(concat)
