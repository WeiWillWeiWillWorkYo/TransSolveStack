"""Taichi backend capability wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class TaichiBackendInfo:
    available: bool
    arch: str = "gpu"
    reason: str | None = None


def probe_taichi_backend() -> TaichiBackendInfo:
    if find_spec("taichi") is None:
        return TaichiBackendInfo(
            available=False,
            reason="taichi package is not installed",
        )
    return TaichiBackendInfo(available=True)

