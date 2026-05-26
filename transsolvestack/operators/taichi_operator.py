"""Taichi operator base classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from transsolvestack.core.types import LinearOperatorSpec
from transsolvestack.operators.base import LinearOperator


@dataclass
class TaichiLinearOperator(LinearOperator):
    """Matrix-free Taichi operator wrapper.

    The apply kernel is injected by concrete operator families.
    """

    spec: LinearOperatorSpec
    apply_kernel: Callable[[Any, Any | None], Any]

    def apply(self, x: Any, y: Any | None = None) -> Any:
        if not self.spec.device_resident:
            raise ValueError("TaichiLinearOperator requires device-resident data")
        return self.apply_kernel(x, y)

