"""Execution engine skeleton."""

from __future__ import annotations

from typing import Any

from transsolvestack.core.result import PolicyPlan, SolveResult
from transsolvestack.core.types import SolveContext
from transsolvestack.operators.base import LinearOperator
from transsolvestack.runtime.plan_resolver import PlanResolver


class TaichiExecutionEngine:
    """GPU-first execution engine."""

    def __init__(self, resolver: PlanResolver | None = None) -> None:
        self.resolver = resolver or PlanResolver()

    def solve(
        self,
        operator: LinearOperator,
        rhs: Any,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        if plan.backend != "taichi_gpu":
            raise ValueError(f"TaichiExecutionEngine cannot execute {plan.backend}")
        resolved = self.resolver.resolve(plan)
        return resolved.solver.solve(operator, rhs, context, plan)
