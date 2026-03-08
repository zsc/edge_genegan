from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class UniformGapPairSampler:
    clip_len: int
    max_gap: int

    def __call__(self) -> tuple[int, int]:
        if self.clip_len < 2:
            raise ValueError("clip_len must be >= 2")
        max_gap = max(1, min(self.max_gap, self.clip_len - 1))
        t = random.randint(0, self.clip_len - 1)
        if t - max_gap < 0:
            s = t + random.randint(1, max_gap)
        elif t + max_gap >= self.clip_len:
            s = t - random.randint(1, min(max_gap, t))
        else:
            s = t + random.choice([*range(-max_gap, 0), *range(1, max_gap + 1)])
            if s == t:
                s = t + 1
        if s < 0 or s >= self.clip_len:
            s = max(0, min(s, self.clip_len - 1))
            if s == t:
                s = (s + 1) % self.clip_len
        return min(t, s), max(t, s)
