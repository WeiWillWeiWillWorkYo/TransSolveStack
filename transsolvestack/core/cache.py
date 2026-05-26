"""Device-aware cache contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolverCache:
    """Cache for device buffers, operator state, and repeated-solve metadata."""

    cache_id: str
    backend: str = "taichi_gpu"
    entries: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.entries.get(key, default)

    def put(self, key: str, value: Any) -> None:
        self.entries[key] = value

