from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LinearConnector(nn.Module):
    def __init__(self, clip_dim: int, lm_dim: int, num_visual_tokens: int, **_: object):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.lm_dim = lm_dim
        self.proj = nn.Linear(clip_dim, num_visual_tokens * lm_dim)

    def forward(self, clip_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.proj(clip_features).view(-1, self.num_visual_tokens, self.lm_dim)
        return out, {"energy": out.new_zeros(())}


class MLPConnector(nn.Module):
    def __init__(self, clip_dim: int, lm_dim: int, num_visual_tokens: int, hidden_dim: int, depth: int = 3, dropout: float = 0.0, **_: object):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.lm_dim = lm_dim
        layers = []
        in_dim = clip_dim
        for _ in range(max(depth - 1, 1)):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_visual_tokens * lm_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, clip_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.net(clip_features).view(-1, self.num_visual_tokens, self.lm_dim)
        return out, {"energy": out.new_zeros(())}


class PCNConnector(nn.Module):
    """3-layer predictive-coding connector with iterative latent settling."""

    def __init__(
        self,
        clip_dim: int,
        lm_dim: int,
        num_visual_tokens: int,
        hidden_dim: int,
        depth: int = 3,
        settle_steps: int = 6,
        state_lr: float = 0.2,
        dropout: float = 0.0,
        **_: object,
    ):
        super().__init__()
        if depth != 3:
            raise ValueError("PCNConnector currently expects depth=3 for architectural parity.")
        self.num_visual_tokens = num_visual_tokens
        self.lm_dim = lm_dim
        self.settle_steps = settle_steps
        self.state_lr = state_lr
        self.bottom_init = nn.Linear(clip_dim, hidden_dim)
        self.h1_init = nn.Linear(clip_dim, hidden_dim)
        self.h2_init = nn.Linear(clip_dim, hidden_dim)
        self.pred_h1_to_h0 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.pred_h2_to_h1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.bottom_to_h1 = nn.Linear(hidden_dim, hidden_dim)
        self.h1_to_h2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_visual_tokens * lm_dim)

    def forward(self, clip_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h0_target = F.gelu(self.bottom_init(clip_features))
        h1 = F.gelu(self.h1_init(clip_features))
        h2 = F.gelu(self.h2_init(clip_features))
        energies = []

        for _ in range(self.settle_steps):
            pred_h0 = self.pred_h1_to_h0(h1)
            pred_h1 = self.pred_h2_to_h1(h2)
            err_h0 = h0_target - pred_h0
            err_h1 = h1 - pred_h1
            energy = 0.5 * (err_h0.pow(2).mean() + err_h1.pow(2).mean())
            energies.append(energy)

            bu_h1 = self.bottom_to_h1(err_h0)
            td_h1 = -err_h1
            bu_h2 = self.h1_to_h2(err_h1)
            h1 = h1 + self.state_lr * (bu_h1 + td_h1)
            h2 = h2 + self.state_lr * bu_h2
            h1 = F.layer_norm(h1, h1.shape[-1:])
            h2 = F.layer_norm(h2, h2.shape[-1:])

        visual = self.decoder(h2).view(-1, self.num_visual_tokens, self.lm_dim)
        energy = torch.stack(energies).mean() if energies else visual.new_zeros(())
        settle_delta = energies[-1] - energies[0] if len(energies) > 1 else energy.new_zeros(())
        return visual, {"energy": energy, "settle_delta": settle_delta}


def build_connector(kind: str, clip_dim: int, lm_dim: int, cfg: Dict[str, object]) -> nn.Module:
    common = dict(
        clip_dim=clip_dim,
        lm_dim=lm_dim,
        num_visual_tokens=int(cfg["num_visual_tokens"]),
        hidden_dim=int(cfg.get("hidden_dim", lm_dim)),
        depth=int(cfg.get("depth", 3)),
        settle_steps=int(cfg.get("settle_steps", 6)),
        state_lr=float(cfg.get("state_lr", 0.2)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    if kind == "linear":
        return LinearConnector(**common)
    if kind == "mlp":
        return MLPConnector(**common)
    if kind == "pcn":
        return PCNConnector(**common)
    raise ValueError(f"Unknown connector type: {kind}")
