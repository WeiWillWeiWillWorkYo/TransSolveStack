"""Phase 1 heuristic policy planner."""

from __future__ import annotations

from transsolvestack.core.result import PolicyPlan
from transsolvestack.core.types import LinearSystem, SolveContext
from transsolvestack.policies.contracts import PolicyPlanner


class TaichiFirstHeuristicPlanner(PolicyPlanner):
    """Conservative Taichi/GPU-first planner for Phase 1."""

    def plan(self, system: LinearSystem, context: SolveContext) -> PolicyPlan:
        backend = context.required_backend or "taichi_gpu"
        if backend != "taichi_gpu" and not context.allow_cpu_reference:
            raise ValueError(f"Unsupported non-GPU backend requested: {backend}")

        solver_name = "pcg" if system.operator.symmetry in {"spd", "symmetric"} else "bicgstab"
        preconditioner = {"name": "jacobi", "device_resident": True}

        return PolicyPlan(
            plan_id=f"plan:{system.system_id}:{context.context_id}:taichi",
            backend=backend,
            solver={
                "name": solver_name,
                "max_iter": context.max_iter,
                "tolerance_abs": context.tolerance_abs,
                "tolerance_rel": context.tolerance_rel,
            },
            preconditioner=preconditioner,
            reuse={
                "device_buffers": True,
                "warm_start": bool(context.metadata.get("warm_start", False)),
            },
            fallback_chain=[],
            audit={
                "policy": "TaichiFirstHeuristicPlanner",
                "cpu_reference_allowed": context.allow_cpu_reference,
            },
        )

