from transsolvestack.backends.registry import default_backend_registry
from transsolvestack.core.types import LinearOperatorSpec, LinearSystem, SolveContext
from transsolvestack.policies.heuristic import TaichiFirstHeuristicPlanner


def test_default_backend_registry_prefers_taichi_gpu():
    registry = default_backend_registry()
    assert "taichi_gpu" in registry.names()
    assert registry.get("taichi_gpu").primary is True


def test_heuristic_planner_emits_taichi_plan():
    spec = LinearOperatorSpec(
        operator_id="op",
        kind="matrix_free",
        shape=(10, 10),
        symmetry="spd",
    )
    system = LinearSystem(system_id="sys", operator=spec)
    context = SolveContext(context_id="ctx")
    plan = TaichiFirstHeuristicPlanner().plan(system, context)
    assert plan.backend == "taichi_gpu"
    assert plan.solver["name"] == "pcg"
    assert plan.preconditioner["name"] == "jacobi"

