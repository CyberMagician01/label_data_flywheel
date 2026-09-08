"""Per-forward Route-Best context shared by custom YOLO modules and losses."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RouteContext:
    clip: torch.Tensor | None = None
    domain_ids: torch.Tensor | None = None
    stats: torch.Tensor | None = None
    density_maps: list[torch.Tensor] = field(default_factory=list)
    route_probs: list[torch.Tensor] = field(default_factory=list)
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


_CTX: ContextVar[RouteContext | None] = ContextVar("bee_route_context", default=None)


def begin_route_context(
    clip: torch.Tensor | None = None,
    domain_ids: torch.Tensor | None = None,
    stats: torch.Tensor | None = None,
    **meta: Any,
) -> RouteContext:
    ctx = RouteContext(clip=clip, domain_ids=domain_ids, stats=stats, meta=dict(meta))
    _CTX.set(ctx)
    return ctx


def get_route_context() -> RouteContext | None:
    return _CTX.get()


def clear_route_context() -> None:
    _CTX.set(None)


def append_density_map(density: torch.Tensor) -> None:
    ctx = get_route_context()
    if ctx is not None:
        ctx.density_maps.append(density)


def append_route_probs(route: torch.Tensor) -> None:
    ctx = get_route_context()
    if ctx is not None:
        ctx.route_probs.append(route)


def add_aux_loss(name: str, value: torch.Tensor) -> None:
    ctx = get_route_context()
    if ctx is None:
        return
    ctx.aux_losses[name] = ctx.aux_losses.get(name, value.new_zeros(())) + value
