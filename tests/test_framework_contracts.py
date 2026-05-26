from dataclasses import asdict

import pytest

from transsolvestack.core.result import PolicyPlan, RunTrace
from transsolvestack.core.types import (
    ImplicitSolveContext,
    LinearOperatorSpec,
    LinearSystem,
    SolveContext,
)
from transsolvestack.operators.taichi_operator import TaichiLinearOperator
from transsolvestack.profiling.trace import trace_to_record


def test_core_contracts_are_serializable():
    spec = LinearOperatorSpec(
        operator_id="op",
        kind="matrix_free",
        shape=(16, 16),
        dtype="float32",
        symmetry="spd",
        metadata={"family": "unit"},
    )
    system = LinearSystem(system_id="sys", operator=spec, family_id="family")
    context = ImplicitSolveContext(
        context_id="ctx",
        repeated_solve_group_id="seq",
        timestep_id="t0",
        newton_iter=0,
    )
    assert asdict(system)["operator"]["device_resident"] is True
    assert asdict(context)["warm_start"] is True


def test_trace_record_contains_gpu_accounting_fields():
    trace = RunTrace(
        run_id="run",
        plan_id="plan",
        backend="taichi_gpu",
        status="success",
        gpu_kernel_time_ms=1.25,
        transfer_time_ms=0.0,
        num_iterations=4,
        residual_history=[1.0, 0.25],
    )
    record = trace_to_record(trace)
    assert record["backend"] == "taichi_gpu"
    assert record["gpu_kernel_time_ms"] == 1.25
    assert record["residual_history"] == [1.0, 0.25]


def test_taichi_operator_invokes_injected_apply_kernel_without_gpu():
    spec = LinearOperatorSpec(
        operator_id="op",
        kind="matrix_free",
        shape=(2, 2),
        device_resident=True,
    )
    operator = TaichiLinearOperator(spec=spec, apply_kernel=lambda x, y=None: x + 1)
    assert operator.apply(2) == 3


def test_taichi_operator_rejects_non_device_resident_spec():
    spec = LinearOperatorSpec(
        operator_id="op",
        kind="matrix_free",
        shape=(2, 2),
        device_resident=False,
    )
    operator = TaichiLinearOperator(spec=spec, apply_kernel=lambda x, y=None: x)
    with pytest.raises(ValueError):
        operator.apply(1)


def test_policy_plan_is_not_solver_label_only():
    plan = PolicyPlan(
        plan_id="plan",
        backend="taichi_gpu",
        solver={"name": "pcg"},
        preconditioner={"name": "jacobi"},
        reuse={"device_buffers": True},
        audit={"policy": "unit"},
    )
    record = asdict(plan)
    assert record["backend"] == "taichi_gpu"
    assert record["solver"]["name"] == "pcg"
    assert record["preconditioner"]["name"] == "jacobi"
    assert record["reuse"]["device_buffers"] is True

