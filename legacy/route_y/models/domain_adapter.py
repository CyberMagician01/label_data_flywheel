"""RGB/IR prototype expert adapter for BeePoseTrack-Y."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .route_context import add_aux_loss, append_route_probs


class DomainPrototypeBank(nn.Module):
    """Maintains RGB, IR and shared foreground prototypes."""

    def __init__(self, channels: int, momentum: float = 0.95) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer("rgb", F.normalize(torch.randn(channels), dim=0))
        self.register_buffer("ir", F.normalize(torch.randn(channels), dim=0))
        self.register_buffer("shared", F.normalize(torch.randn(channels), dim=0))

    @torch.no_grad()
    def update(self, features: torch.Tensor, domain_ids: torch.Tensor) -> None:
        pooled = F.normalize(F.adaptive_avg_pool2d(features, 1).flatten(1), dim=1)
        for domain_value, name in ((0, "rgb"), (1, "ir")):
            mask = domain_ids == domain_value
            if mask.any():
                proto = F.normalize(pooled[mask].mean(0), dim=0)
                old = getattr(self, name)
                old.mul_(self.momentum).add_(proto * (1.0 - self.momentum))
                old.copy_(F.normalize(old, dim=0))
        self.shared.copy_(F.normalize(0.5 * (self.rgb + self.ir), dim=0))


class RGBIRDomainAdapter(nn.Module):
    """Shared branch plus RGB and IR depthwise experts with soft routing."""

    def __init__(self, channels: int, stats_dim: int = 6) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.rgb_expert = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False), nn.BatchNorm2d(channels))
        self.ir_expert = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False), nn.BatchNorm2d(channels))
        self.router = nn.Sequential(nn.Linear(channels + stats_dim, max(channels // 8, 16)), nn.SiLU(inplace=True), nn.Linear(max(channels // 8, 16), 3))
        self.project = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        self.prototypes = DomainPrototypeBank(channels)

    def forward(self, x: torch.Tensor, stats: torch.Tensor | None = None, domain_ids: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        if stats is None:
            stats = x.new_zeros((b, 6))
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        route = torch.softmax(self.router(torch.cat([pooled, stats], dim=1)), dim=1)
        append_route_probs(route)
        shared = self.shared(x)
        rgb = self.rgb_expert(x)
        ir = self.ir_expert(x)
        fused = route[:, 0, None, None, None] * shared + route[:, 1, None, None, None] * rgb + route[:, 2, None, None, None] * ir
        if self.training and domain_ids is not None:
            self.prototypes.update(x.detach(), domain_ids.detach())
            domain_term = route[:, 2] if domain_ids.float().mean() > 0.5 else route[:, 1]
            target_expert = torch.where(domain_ids.long() == 1, route[:, 2], route[:, 1])
            expert_margin = F.relu(0.5 - target_expert).mean()
            entropy = -(route * route.clamp_min(1e-6).log()).sum(1).mean()
            balance = torch.square(route.mean(0) - route.new_tensor([0.34, 0.33, 0.33])).mean()
            add_aux_loss("prototype", expert_margin + 0.1 * balance - 0.01 * entropy + 0.0 * domain_term.mean())
        return x + self.project(fused)
