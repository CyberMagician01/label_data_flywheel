"""RGB/IR balanced sampling utilities."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Iterator


class BalancedDomainOrder:
    """Builds a one-epoch order with equal frame quota for RGB and IR."""

    def __init__(self, domains: Iterable[str], seed: int = 2026) -> None:
        self.domains = [str(x).upper() for x in domains]
        self.seed = seed

    def epoch_indices(self, epoch: int) -> list[int]:
        buckets: dict[str, list[int]] = defaultdict(list)
        for idx, domain in enumerate(self.domains):
            buckets[domain].append(idx)
        rng = random.Random(self.seed + epoch)
        for values in buckets.values():
            rng.shuffle(values)
        if not buckets:
            return []
        quota = max(len(v) for v in buckets.values())
        expanded: list[int] = []
        for domain in sorted(buckets):
            values = buckets[domain]
            if not values:
                continue
            expanded.extend(values[i % len(values)] for i in range(quota))
        rng.shuffle(expanded)
        return expanded

    def __iter__(self) -> Iterator[int]:
        yield from self.epoch_indices(0)
