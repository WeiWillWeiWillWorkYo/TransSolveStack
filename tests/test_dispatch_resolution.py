import pytest

from transsolvestack.core.result import PolicyPlan
from transsolvestack.preconditioners.identity import NoPreconditioner
from transsolvestack.preconditioners.registry import default_preconditioner_registry
from transsolvestack.runtime.plan_resolver import PlanResolver
from transsolvestack.solvers.registry import default_solver_registry


def test_default_solver_registry_contains_phase1_krylov_solvers():
    registry = default_solver_registry()
    assert registry.names() == [
        "bicgstab",
        "cg",
        "chebyshev",
        "gmres",
        "pcg",
        "richardson",
    ]
    assert registry.create("pcg").name == "pcg"
    assert registry.create("gmres").name == "gmres"
    assert registry.create("richardson").name == "richardson"
    assert registry.create("chebyshev").name == "chebyshev"


def test_default_preconditioner_registry_contains_phase1_preconditioners():
    registry = default_preconditioner_registry()
    assert registry.names() == [
        "block_jacobi",
        "ilu0",
        "jacobi",
        "none",
        "row_column_equilibration",
        "symmetric_equilibration",
    ]
    assert isinstance(registry.create("none"), NoPreconditioner)


def test_plan_resolver_resolves_taichi_plan_components():
    plan = PolicyPlan(
        plan_id="plan",
        backend="taichi_gpu",
        solver={"name": "pcg"},
        preconditioner={"name": "jacobi"},
    )
    resolved = PlanResolver().resolve(plan)
    assert resolved.backend.name == "taichi_gpu"
    assert resolved.solver.name == "pcg"
    assert resolved.preconditioner.name == "taichi_jacobi"


def test_plan_resolver_accepts_symmetric_equilibration_preconditioner():
    plan = PolicyPlan(
        plan_id="plan",
        backend="taichi_gpu",
        solver={"name": "pcg"},
        preconditioner={"name": "symmetric_equilibration"},
    )
    resolved = PlanResolver().resolve(plan)
    assert resolved.preconditioner.name == "taichi_symmetric_equilibration"


def test_plan_resolver_accepts_row_column_equilibration_preconditioner():
    plan = PolicyPlan(
        plan_id="plan",
        backend="taichi_gpu",
        solver={"name": "bicgstab"},
        preconditioner={"name": "row_column_equilibration"},
    )
    resolved = PlanResolver().resolve(plan)
    assert resolved.preconditioner.name == "taichi_row_column_equilibration"


def test_plan_resolver_accepts_ilu0_preconditioner():
    plan = PolicyPlan(
        plan_id="plan",
        backend="taichi_gpu",
        solver={"name": "bicgstab"},
        preconditioner={"name": "ilu0"},
    )
    resolved = PlanResolver().resolve(plan)
    assert resolved.preconditioner.name == "taichi_ilu0"


def test_plan_resolver_defaults_to_no_preconditioner():
    plan = PolicyPlan(
        plan_id="plan",
        backend="taichi_gpu",
        solver={"name": "cg"},
    )
    resolved = PlanResolver().resolve(plan)
    assert resolved.preconditioner.name == "none"


def test_plan_resolver_rejects_solver_not_supported_by_backend():
    plan = PolicyPlan(
        plan_id="plan",
        backend="taichi_gpu",
        solver={"name": "direct_lu"},
    )
    with pytest.raises(ValueError):
        PlanResolver().resolve(plan)


def test_plan_resolver_rejects_missing_solver_name():
    plan = PolicyPlan(
        plan_id="plan",
        backend="taichi_gpu",
        solver={},
    )
    with pytest.raises(ValueError):
        PlanResolver().resolve(plan)
