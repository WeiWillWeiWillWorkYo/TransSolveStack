"""Identity/no-op preconditioner."""

from __future__ import annotations

from typing import Any

from transsolvestack.operators.base import LinearOperator
from transsolvestack.preconditioners.base import Preconditioner


class NoPreconditioner(Preconditioner):
    name = "none"

    def setup(self, operator: LinearOperator) -> None:
        return None

    def apply(self, r: Any, z: Any | None = None) -> Any:
        if z is None:
            return r
        return z

