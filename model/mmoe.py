"""model/mmoe.py — Server-side Mixture-of-Experts for VFL-MTL (Ma et al., 2018 KDD)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor



# Expert MLP


class ExpertMLP(nn.Module):
    """Single shared expert: input_dim → hidden_dim → output_dim with ReLU."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)



# MMoE Layer


class MMoELayer(nn.Module):
    """Mixture-of-Experts layer with per-task softmax gating (Ma et al., 2018)."""

    def __init__(
        self,
        input_dim: int = 192,
        num_experts: int = 4,
        expert_hidden: int = 128,
        expert_out: int = 64,
        num_tasks: int = 3,
        uniform_gating: bool = False,
    ):
        super().__init__()
        self.num_experts    = num_experts
        self.num_tasks      = num_tasks
        self.uniform_gating = uniform_gating

        self.experts = nn.ModuleList([
            ExpertMLP(input_dim, expert_hidden, expert_out)
            for _ in range(num_experts)
        ])
        # One softmax gating network per task (unused when uniform_gating=True)
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts, bias=False)
            for _ in range(num_tasks)
        ])

    def forward(self, x: Tensor) -> list[Tensor]:
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, num_experts, expert_out)

        task_outputs = []
        for gate in self.gates:
            if self.uniform_gating:
                # Ablation 4: fixed equal weights — no learned routing
                weights = torch.full(
                    (x.size(0), self.num_experts),
                    1.0 / self.num_experts,
                    device=x.device,
                )
            else:
                weights = torch.softmax(gate(x), dim=-1)  # (B, num_experts)
            out = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, expert_out)
            task_outputs.append(out)

        return task_outputs  # 3 × (B, expert_out)



# Task Heads


class TaskHead(nn.Module):
    """Per-task output head: binary/regression → Linear(in,1); multilabel → Linear(in,n)+Sigmoid."""

    def __init__(self, in_dim: int, task_type: str, n_classes: int = 1):
        super().__init__()
        assert task_type in ("binary", "regression", "multilabel")
        self.task_type = task_type

        if task_type == "binary":
            self.head = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        elif task_type == "regression":
            self.head = nn.Linear(in_dim, 1)
        else:  # multilabel
            self.head = nn.Sequential(nn.Linear(in_dim, n_classes), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)



# MMoEServer


class MMoEServer(nn.Module):
    """Server-side model: MMoELayer + per-task heads over [h_A; h_B; h_C]."""

    # Task order is fixed; names are used as dict keys throughout the codebase.
    TASK_ORDER = ("ihm", "decomp", "pheno")

    # Default task types for MIMIC; override via task_types kwarg for other datasets.
    _DEFAULT_TASK_TYPES = {"ihm": "binary", "decomp": "binary", "pheno": "multilabel"}
    _N_CLASSES          = {"ihm": 1,        "decomp": 1,        "pheno": 25}

    def __init__(
        self,
        input_dim: int = 192,
        num_experts: int = 4,
        expert_hidden: int = 128,
        expert_out: int = 64,
        use_mmoe: bool = True,
        uniform_gating: bool = False,
        task_types: dict | None = None,
    ):
        super().__init__()
        self.use_mmoe = use_mmoe
        _tt = {**self._DEFAULT_TASK_TYPES, **(task_types or {})}

        if use_mmoe:
            self.mmoe = MMoELayer(
                input_dim=input_dim,
                num_experts=num_experts,
                expert_hidden=expert_hidden,
                expert_out=expert_out,
                num_tasks=len(self.TASK_ORDER),
                uniform_gating=uniform_gating,
            )
        else:
            self.shared_bottom = nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, expert_out),
                nn.ReLU(),
            )

        self.heads = nn.ModuleDict({
            t: TaskHead(expert_out, _tt[t], self._N_CLASSES[t])
            for t in self.TASK_ORDER
        })

    def forward(self, concat_embedding: Tensor) -> dict[str, Tensor]:
        if self.use_mmoe:
            task_vecs = self.mmoe(concat_embedding)  # 3 × (B, expert_out)
        else:
            shared = self.shared_bottom(concat_embedding)  # (B, expert_out)
            task_vecs = [shared] * len(self.TASK_ORDER)
        return {
            task: self.heads[task](vec)
            for task, vec in zip(self.TASK_ORDER, task_vecs)
        }
