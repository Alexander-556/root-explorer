# src/neutrino/clf/model.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Sequence
from neutrino.clf.config.model_config import ClfModelConfig


class MLPBCE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_sizes: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # single logit (BCEWithLogitsLoss)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # logits [N]

    @classmethod
    def from_config(
        cls,
        in_dim: int,
        cfg: ClfModelConfig,
    ) -> "MLPBCE":
        """
        Build MLPBCE from a ClfModelConfig.
        Expects cfg.type == "torch_mlp_bce" and cfg.params with:
        - hidden_sizes: list[int]
        - dropout: float
        """
        assert cfg.type == "torch_mlp_bce", f"Unexpected model type: {cfg.type}"
        params = cfg.params
        hidden = params.get("hidden_sizes", [64, 32])
        dropout = float(params.get("dropout", 0.0))
        return cls(
            in_dim=in_dim,
            hidden_sizes=hidden,
            dropout=dropout,
        )
