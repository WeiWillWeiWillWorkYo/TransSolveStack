"""Preconditioner interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from transsolvestack.operators.base import LinearOperator


class Preconditioner(ABC):
    name: str

    @abstractmethod
    def setup(self, operator: LinearOperator) -> None:
        """Prepare preconditioner state."""

    @abstractmethod
    def apply(self, r: Any, z: Any | None = None) -> Any:
        """Apply z = M^{-1} r."""

