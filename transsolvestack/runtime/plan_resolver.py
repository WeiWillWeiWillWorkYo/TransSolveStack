"""Resolve PolicyPlan objects into runtime components."""

from __future__ import annotations

from dataclasses import dataclass

from transsolvestack.backends.registry import (
    BackendDescriptor,
    BackendRegistry,
    default_backend_registry,
)
from transsolvestack.core.result import PolicyPlan
from transsolvestack.preconditioners.base import Preconditioner
from transsolvestack.preconditioners.registry import (
    PreconditionerRegistry,
    default_preconditioner_registry,
)
from transsolvestack.solvers.base import Solver
from transsolvestack.solvers.registry import SolverRegistry, default_solver_registry


@dataclass(frozen=True)
class ResolvedExecutionPlan:
    policy_plan: PolicyPlan
    backend: BackendDescriptor
    solver: Solver
    preconditioner: Preconditioner


class PlanResolver:
    def __init__(
        self,
        backend_registry: BackendRegistry | None = None,
        solver_registry: SolverRegistry | None = None,
        preconditioner_registry: PreconditionerRegistry | None = None,
    ) -> None:
        self.backend_registry = backend_registry or default_backend_registry()
        self.solver_registry = solver_registry or default_solver_registry()
        self.preconditioner_registry = (
            preconditioner_registry or default_preconditioner_registry()
        )

    def resolve(self, plan: PolicyPlan) -> ResolvedExecutionPlan:
        backend = self.backend_registry.get(plan.backend)
        solver_name = _required_name(plan.solver, "solver")
        preconditioner_name = plan.preconditioner.get("name", "none")

        if solver_name not in backend.supported_solvers:
            raise ValueError(
                f"solver {solver_name} is not supported by backend {backend.name}"
            )
        if preconditioner_name not in backend.supported_preconditioners:
            raise ValueError(
                f"preconditioner {preconditioner_name} is not supported by "
                f"backend {backend.name}"
            )

        return ResolvedExecutionPlan(
            policy_plan=plan,
            backend=backend,
            solver=self.solver_registry.create(solver_name),
            preconditioner=self.preconditioner_registry.create(preconditioner_name),
        )


def _required_name(config: dict, label: str) -> str:
    name = config.get("name")
    if not name:
        raise ValueError(f"{label} config must include a name")
    return str(name)

