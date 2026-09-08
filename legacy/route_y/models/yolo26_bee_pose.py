"""Custom YOLO26 BeePose modules and Ultralytics registration."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .density_head import DensityHead
from .domain_adapter import RGBIRDomainAdapter
from .endpoint_refine import EndpointRefine
from .freq_gate import DynamicFrequencyGate
from .rep_tdc import RepTDCFuse
from .route_context import append_density_map, begin_route_context, get_route_context


class BeeRouteBlock(nn.Module):
    """Route-Best feature block: TDC, frequency gate, RGB/IR experts and endpoint refinement."""

    def __init__(self, channels: int, use_tdc: bool = True, use_frequency: bool = True, use_domain_experts: bool = True, use_endpoint_refine: bool = True) -> None:
        super().__init__()
        self.use_frequency = use_frequency
        self.use_domain_experts = use_domain_experts
        self.tdc = RepTDCFuse(channels) if use_tdc else nn.Identity()
        self.freq = DynamicFrequencyGate(channels) if use_frequency else nn.Identity()
        self.domain = RGBIRDomainAdapter(channels) if use_domain_experts else nn.Identity()
        self.endpoint = EndpointRefine(channels) if use_endpoint_refine else nn.Identity()
        self.raw_motion = nn.Sequential(
            nn.Conv2d(4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.motion_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_route_context()
        x = self.tdc(x)
        if ctx is not None and ctx.clip is not None and ctx.clip.shape[0] == x.shape[0]:
            raw_motion = self._raw_clip_motion(ctx.clip, x.shape[-2:])
            raw_motion = self.raw_motion(raw_motion.to(device=x.device, dtype=x.dtype))
            x = x + self.motion_gate(raw_motion) * raw_motion
        stats = ctx.stats if ctx is not None else None
        domain_ids = ctx.domain_ids if ctx is not None else None
        if stats is not None:
            stats = stats.to(device=x.device, dtype=x.dtype)
        if domain_ids is not None:
            domain_ids = domain_ids.to(device=x.device)
        x = self.freq(x, stats=stats) if self.use_frequency else self.freq(x)
        x = self.domain(x, stats=stats, domain_ids=domain_ids) if self.use_domain_experts else self.domain(x)
        return self.endpoint(x)

    @staticmethod
    def _raw_clip_motion(clip: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        b, t, c, h, w = clip.shape
        gray = clip.mean(2)
        current = gray[:, -1:]
        diffs = []
        for lag in (1, 2, 4):
            ref = gray[:, max(t - 1 - lag, 0) : max(t - lag, 1)]
            diffs.append(current - ref)
        raw = torch.cat([current, *diffs], dim=1)
        return F.interpolate(raw, size=size, mode="bilinear", align_corners=False)


class FiveFrameInputAdapter(nn.Module):
    """Convert [B, 5*3, H, W] causal clips to current-frame-compatible RGB tensors."""

    def __init__(self, clip_len: int = 5) -> None:
        super().__init__()
        self.clip_len = clip_len
        self.fuse = nn.Conv2d(clip_len * 3, 3, 1, bias=True)
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.bias.zero_()
            start = (clip_len - 1) * 3
            for c in range(3):
                self.fuse.weight[c, start + c, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FiveFrameInputAdapter expects 4D tensor, got {tuple(x.shape)}")
        if x.shape[1] == 3:
            begin_route_context(clip=x[:, None], source="single_frame")
            return x
        expected = self.clip_len * 3
        if x.shape[1] != expected:
            raise ValueError(f"expected 3 or {expected} channels, got {x.shape[1]}")
        clip = x.view(x.shape[0], self.clip_len, 3, x.shape[-2], x.shape[-1])
        ctx = get_route_context()
        domain_ids = ctx.domain_ids if ctx is not None else None
        stats = ctx.stats if ctx is not None else None
        begin_route_context(clip=clip, domain_ids=domain_ids, stats=stats, source="five_frame")
        return self.fuse(x)


class BeeDensityTap(nn.Module):
    """Feature-preserving density tap.

    The density map is cached for auxiliary losses/export metadata while the
    original feature tensor continues through the YOLO neck/head.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head = DensityHead(channels)
        self.last_density: torch.Tensor | None = None
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        density = self.head(x)
        self.last_density = None if self.training else density.detach()
        append_density_map(density)
        return x


def register_ultralytics_modules() -> None:
    """Register custom modules in Ultralytics' model parser namespace."""

    import ultralytics.nn.tasks as tasks

    tasks.BeeRouteBlock = BeeRouteBlock
    tasks.BeeDensityTap = BeeDensityTap
    tasks.FiveFrameInputAdapter = FiveFrameInputAdapter
