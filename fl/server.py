"""
fl/server.py — VFL server: aggregation, loss, and gradient distribution.

Training protocol per batch:
  1. aggregate_embeddings()  — concatenate the three site embeddings → (B, 192)
  2. forward_and_loss()      — run MMoEServer, compute per-task losses
  3. backward_and_step()     — backprop through MMoE, update server weights
  4. get_embedding_gradients() — slice the gradient of the concatenated embedding
                                  back into per-site pieces and return to clients

DP hook (Paper 2): subclass and override get_embedding_gradients() to add noise
before returning gradients to clients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.mmoe import MMoEServer


# Loss functions (instantiated once, reused every batch)
_BCE  = nn.BCELoss()
_CE   = nn.CrossEntropyLoss()


class VFLServer:
    """
    Parameters
    ----------
    embed_dim    : per-site embedding size (default 64); concat input = 3 × embed_dim
    num_experts  : MMoE expert count (default 4)
    expert_hidden: hidden size inside each expert MLP (default 128)
    lr           : Adam learning rate for MMoE + task heads
    device       : cpu or cuda
    task_weights : relative loss weights for {'ihm', 'los', 'pheno'} (default equal)
    """

    SITES = ("A", "B", "C")

    def __init__(
        self,
        embed_dim: int = 64,
        num_experts: int = 4,
        expert_hidden: int = 128,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
        task_weights: dict[str, float] | None = None,
    ):
        self.device    = torch.device(device)
        self.embed_dim = embed_dim

        self.model = MMoEServer(
            input_dim=len(self.SITES) * embed_dim,
            num_experts=num_experts,
            expert_hidden=expert_hidden,
            expert_out=embed_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.task_weights = task_weights or {"ihm": 1.0, "los": 1.0, "pheno": 1.0}

        # Stored between aggregate_embeddings() and get_embedding_gradients()
        self._concat_embedding: Tensor | None = None

    # ------------------------------------------------------------------

    def aggregate_embeddings(self, embeddings: dict[str, Tensor]) -> Tensor:
        """
        Concatenate per-site embeddings into a single vector.

        Parameters
        ----------
        embeddings : {'A': (B, 64), 'B': (B, 64), 'C': (B, 64)}

        Returns
        -------
        (B, 192)  — stored internally; also returned for convenience
        """
        parts = [embeddings[s].to(self.device) for s in self.SITES]
        self._concat_embedding = torch.cat(parts, dim=-1)  # (B, 192)
        self._concat_embedding.retain_grad()  # non-leaf: opt in to keep .grad after backward
        return self._concat_embedding

    def forward_and_loss(
        self,
        labels: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Run MMoEServer and compute the weighted sum of per-task losses.

        Parameters
        ----------
        labels : {
            'ihm'  : (B,)    float32  — binary mortality label
            'los'  : (B,)    int64    — LOS bin index 0–9
            'pheno': (B, 25) float32  — multi-label phenotype flags
        }

        Returns
        -------
        total_loss : scalar Tensor
        task_losses: {'ihm': scalar, 'los': scalar, 'pheno': scalar}
        """
        assert self._concat_embedding is not None, "call aggregate_embeddings() first"

        self.model.train()
        preds = self.model(self._concat_embedding)  # dict[str, Tensor]

        task_losses = {
            "ihm":   _BCE(preds["ihm"].squeeze(-1),
                          labels["ihm"].to(self.device)),
            "los":   _CE(preds["los"],
                         labels["los"].to(self.device)),
            "pheno": _BCE(preds["pheno"],
                          labels["pheno"].to(self.device)),
        }

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
        """
        Slice the gradient of the concatenated embedding back into per-site pieces.

        Called after backward_and_step(). Each client receives its own 64-dim slice.

        DP hook (Paper 2): subclass and add noise to grad slices here.

        Returns
        -------
        {'A': (B, 64), 'B': (B, 64), 'C': (B, 64)}
        """
        assert self._concat_embedding is not None and \
               self._concat_embedding.grad is not None, \
            "call backward_and_step() before get_embedding_gradients()"

        grad = self._concat_embedding.grad  # (B, 192)
        slices = grad.split(self.embed_dim, dim=-1)  # three (B, 64) tensors
        return dict(zip(self.SITES, slices))

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, embeddings: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Run inference (no gradient tracking).

        Returns raw model outputs: probabilities for IHM/pheno, logits for LOS.
        """
        self.model.eval()
        concat = torch.cat(
            [embeddings[s].to(self.device) for s in self.SITES], dim=-1
        )
        return self.model(concat)
