"""fl/client.py — VFL client: local LSTM encoder for one hospital site."""

from __future__ import annotations

import torch
from torch import Tensor

from model.encoder import SiteEncoder


class VFLClient:

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        embed_dim: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.encoder = SiteEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embed_dim=embed_dim,
            dropout=dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self._local_embedding: Tensor | None = None  # kept for backward pass

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Run encoder; return detached embedding (B, embed_dim) for the server."""
        self.encoder.train()
        self._local_embedding = self.encoder(x.to(self.device), mask.to(self.device))
        return self._local_embedding.detach().requires_grad_(True)

    def receive_gradient(self, grad: Tensor) -> None:
        """Backprop server gradient through local encoder and update weights."""
        assert self._local_embedding is not None, "call forward() first"
        self.optimizer.zero_grad()
        self._local_embedding.backward(grad.to(self.device))
        self.optimizer.step()
        self._local_embedding = None

    @torch.no_grad()
    def eval_forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Encode in eval mode (no gradient tracking). Returns (B, embed_dim)."""
        self.encoder.eval()
        return self.encoder(x.to(self.device), mask.to(self.device))


    def get_encoder_params(self) -> dict:
        return {k: v.clone() for k, v in self.encoder.state_dict().items()}

    def set_encoder_params(self, state_dict: dict) -> None:
        self.encoder.load_state_dict(state_dict)
