from transsolvestack.core.result import PolicyPlan, RunTrace, SolveResult
from transsolvestack.runtime.guarded_fallback import (
    RuntimeGuardConfig,
    execute_with_fallback_guard,
)


def test_guarded_fallback_retries_failed_primary():
    primary = PolicyPlan(
        plan_id="primary",
        backend="taichi_gpu",
        solver={"name": "cg"},
        audit={"candidate_id": "primary"},
    )
    fallback = PolicyPlan(
        plan_id="fallback",
        backend="taichi_gpu",
        solver={"name": "pcg"},
        audit={"candidate_id": "fallback"},
    )

    def execute(plan: PolicyPlan) -> SolveResult:
        status = "success" if plan.plan_id == "fallback" else "failed"
        trace = RunTrace(
            run_id=plan.plan_id,
            plan_id=plan.plan_id,
            backend=plan.backend,
            status=status,
            failure_class=None if status == "success" else "not_converged",
        )
        return SolveResult(status=status, solution=(1.0,), trace=trace)

    outcome = execute_with_fallback_guard((primary, fallback), execute)

    assert outcome.guard_status == "success"
    assert outcome.used_fallback is True
    assert outcome.result.status == "fallback_success"
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].status == "failed"
    assert outcome.result.trace.metadata["runtime_guard"]["used_fallback"] is True


def test_guarded_fallback_records_exceptions():
    primary = PolicyPlan(
        plan_id="primary",
        backend="taichi_gpu",
        solver={"name": "cg"},
        audit={"candidate_id": "primary"},
    )

    def execute(_plan: PolicyPlan) -> SolveResult:
        raise RuntimeError("boom")

    outcome = execute_with_fallback_guard(
        (primary,),
        execute,
        config=RuntimeGuardConfig(fallback_on_exception=False),
    )

    assert outcome.guard_status == "exception"
    assert outcome.result.status == "failed"
    assert outcome.attempts[0].exception_type == "RuntimeError"
