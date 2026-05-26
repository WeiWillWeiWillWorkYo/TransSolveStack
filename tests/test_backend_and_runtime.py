import pytest

from transsolvestack.backends.registry import default_backend_registry
from transsolvestack.core.result import PolicyPlan
from transsolvestack.core.types import LinearOperatorSpec, SolveContext
from transsolvestack.operators.taichi_operator import TaichiLinearOperator
from transsolvestack.runtime.engine import TaichiExecutionEngine


def test_backend_registry_marks_cpu_as_reference_only():
    registry = default_backend_registry()
    cpu = registry.get("cpu_reference")
    taichi = registry.get("taichi_gpu")
    assert taichi.primary is True
    assert cpu.primary is False
    assert cpu.metadata["purpose"] == "correctness_reference_only"


def test_taichi_execution_engine_rejects_non_taichi_plan():
    engine = TaichiExecutionEngine()
    operator = TaichiLinearOperator(
        spec=LinearOperatorSpec(
            operator_id="op",
            kind="matrix_free",
            shape=(2, 2),
        ),
        apply_kernel=lambda x, y=None: x,
    )
    plan = PolicyPlan(
        plan_id="plan",
        backend="cpu_reference",
        solver={"name": "reference_direct"},
    )
    with pytest.raises(ValueError):
        engine.solve(operator, rhs=None, context=SolveContext(context_id="ctx"), plan=plan)


def test_taichi_execution_engine_fails_explicitly_until_solver_dispatch_exists():
    engine = TaichiExecutionEngine()
    operator = TaichiLinearOperator(
        spec=LinearOperatorSpec(
            operator_id="op",
            kind="matrix_free",
            shape=(2, 2),
        ),
        apply_kernel=lambda x, y=None: x,
    )
    plan = PolicyPlan(
        plan_id="plan",
        backend="taichi_gpu",
        solver={"name": "pcg"},
    )
    with pytest.raises(NotImplementedError):
        engine.solve(operator, rhs=None, context=SolveContext(context_id="ctx"), plan=plan)

