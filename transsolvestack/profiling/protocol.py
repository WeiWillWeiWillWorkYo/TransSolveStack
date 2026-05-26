"""Benchmark protocol configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Controls warmup and repeated measurements for a benchmark run."""

    warmup_runs: int = 1
    measurement_repeats: int = 3

    def validate(self) -> None:
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.measurement_repeats < 1:
            raise ValueError("measurement_repeats must be >= 1")

