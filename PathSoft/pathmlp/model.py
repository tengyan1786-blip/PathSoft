from __future__ import annotations

import torch
import torch.nn as nn


class PathScorer(nn.Module):
    def __init__(self, emb_dim: int = 1024, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, q_emb: torch.Tensor, path_emb: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                q_emb,
                path_emb,
                q_emb - path_emb,
                q_emb * path_emb,
            ],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)

