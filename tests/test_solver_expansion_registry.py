from transsolvestack.backends.registry import default_backend_registry
from transsolvestack.core.result import PolicyPlan
from transsolvestack.runtime.plan_resolver import PlanResolver
from transsolvestack.solvers.registry import default_solver_registry


def test_solver_registry_contains_expanded_gpu_solvers():
    names = set(default_solver_registry().names())
    assert {"cg", "pcg", "bicgstab", "richardson", "chebyshev"} <= names


def test_taichi_backend_advertises_expanded_solvers():
    backend = default_backend_registry().get("taichi_gpu")
    assert "richardson" in backend.supported_solvers
    assert "chebyshev" in backend.supported_solvers
    assert "ilu0" in backend.supported_preconditioners
    assert "symmetric_equilibration" in backend.supported_preconditioners
    assert "row_column_equilibration" in backend.supported_preconditioners


def test_plan_resolver_accepts_expanded_solver_plans():
    resolver = PlanResolver()
    for solver_name in ("richardson", "chebyshev"):
        resolved = resolver.resolve(
            PolicyPlan(
                plan_id=f"test:{solver_name}",
                backend="taichi_gpu",
                solver={"name": solver_name, "max_iter": 300},
                preconditioner={"name": "jacobi"},
            )
        )
        assert resolved.solver.name == solver_name
