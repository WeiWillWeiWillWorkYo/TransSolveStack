"""Taichi CG/PCG solver skeleton."""

from __future__ import annotations

from typing import Any

from transsolvestack.core.result import PolicyPlan, RunTrace, SolveResult
from transsolvestack.core.types import SolveContext
from transsolvestack.operators.base import LinearOperator
from transsolvestack.solvers.base import Solver


class _NotImplementedTaichiSolver(Solver):
    """Base class for GPU solver contracts before kernels are implemented."""

    name = "taichi_solver"

    def solve(
        self,
        operator: LinearOperator,
        rhs: Any,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        method = getattr(operator, f"solve_{self.name}", None)
        if method is None:
            raise NotImplementedError(
                f"operator {operator.__class__.__name__} does not implement "
                f"solve_{self.name}"
            )
        return method(rhs=rhs, context=context, plan=plan)

    @staticmethod
    def empty_trace(run_id: str, plan: PolicyPlan) -> RunTrace:
        return RunTrace(
            run_id=run_id,
            plan_id=plan.plan_id,
            backend=plan.backend,
            status="failed",
            failure_class="not_implemented",
        )


class TaichiCGSolver(_NotImplementedTaichiSolver):
    """Phase 1 placeholder for GPU-resident CG."""

    name = "cg"


class TaichiPCGSolver(_NotImplementedTaichiSolver):
    """Phase 1 placeholder for GPU-resident PCG."""

    name = "pcg"


class TaichiBiCGSTABSolver(_NotImplementedTaichiSolver):
    """Phase 1 placeholder for GPU-resident BiCGSTAB."""

    name = "bicgstab"


class TaichiGMRESSolver(_NotImplementedTaichiSolver):
    """GPU-resident restarted GMRES for assembled CSR matrices."""

    name = "gmres"


class TaichiRichardsonSolver(_NotImplementedTaichiSolver):
    """GPU-resident weighted Richardson iteration."""

    name = "richardson"


class TaichiChebyshevSolver(_NotImplementedTaichiSolver):
    """GPU-resident Chebyshev semi-iteration."""

    name = "chebyshev"
