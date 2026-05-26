"""Linear operator interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from transsolvestack.core.types import LinearOperatorSpec


class LinearOperator(ABC):
    spec: LinearOperatorSpec

    @abstractmethod
    def apply(self, x: Any, y: Any | None = None) -> Any:
        """Compute y = A x without requiring CPU matrix materialization."""

