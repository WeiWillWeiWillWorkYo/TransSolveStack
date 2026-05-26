"""Solver interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from transsolvestack.core.result import PolicyPlan, SolveResult
from transsolvestack.core.types import SolveContext
from transsolvestack.operators.base import LinearOperator


class Solver(ABC):
    name: str

    @abstractmethod
    def solve(
        self,
        operator: LinearOperator,
        rhs: Any,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        """Execute a solve."""

