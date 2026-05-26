"""Runtime fallback and timeout guard helpers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from transsolvestack.core.result import PolicyPlan, RunTrace, SolveResult


SUCCESS_STATUSES = {"success", "fallback_success"}


@dataclass(frozen=True)
class RuntimeGuardConfig:
    fallback_on_failed_status: bool = True
    fallback_on_exception: bool = True
    fallback_on_timeout: bool = True
    max_attempt_wall_time_ms: float | None = None
    max_total_wall_time_ms: float | None = None
    supports_preemptive_kill: bool = False


@dataclass(frozen=True)
class GuardedAttemptRecord:
    attempt_index: int
    plan_id: str
    candidate_id: str | None
    status: str
    failure_class: str | None
    elapsed_wall_time_ms: float
    timed_out: bool
    exception_type: str | None = None
    exception_message: str | None = None


@dataclass(frozen=True)
class GuardedSolveOutcome:
    result: SolveResult
    attempts: tuple[GuardedAttemptRecord, ...]
    used_fallback: bool
    guard_status: str
    config: RuntimeGuardConfig


def execute_with_fallback_guard(
    plans: Iterable[PolicyPlan],
    execute_attempt: Callable[[PolicyPlan], SolveResult],
    *,
    config: RuntimeGuardConfig | None = None,
) -> GuardedSolveOutcome:
    """Execute plans in order, retrying fallbacks on failure or timeout.

    Timeout handling is post-attempt. It prevents accepting or continuing from a
    too-slow attempt and can stop further fallback attempts after a total budget,
    but it does not preempt an in-flight GPU kernel.
    """

    active_config = config or RuntimeGuardConfig()
    attempts = tuple(plans)
    if not attempts:
        raise ValueError("execute_with_fallback_guard requires at least one plan")

    records: list[GuardedAttemptRecord] = []
    total_start = time.monotonic()
    last_result: SolveResult | None = None
    last_exception: Exception | None = None

    for index, plan in enumerate(attempts):
        if _total_budget_exhausted(active_config, total_start):
            break
        attempt_start = time.monotonic()
        try:
            result = execute_attempt(plan)
        except Exception as exc:  # pragma: no cover - type depends on caller
            elapsed = _elapsed_ms(attempt_start)
            last_exception = exc
            records.append(
                GuardedAttemptRecord(
                    attempt_index=index,
                    plan_id=plan.plan_id,
                    candidate_id=_candidate_id(plan),
                    status="exception",
                    failure_class="exception",
                    elapsed_wall_time_ms=elapsed,
                    timed_out=_attempt_timed_out(active_config, elapsed),
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            )
            if not active_config.fallback_on_exception:
                result = _exception_result(plan, exc, records)
                return _finalize(
                    result,
                    records,
                    used_fallback=index > 0,
                    guard_status="exception",
                    config=active_config,
                )
            continue

        elapsed = _elapsed_ms(attempt_start)
        timed_out = _attempt_timed_out(active_config, elapsed)
        status = str(result.status)
        failure_class = result.trace.failure_class
        records.append(
            GuardedAttemptRecord(
                attempt_index=index,
                plan_id=plan.plan_id,
                candidate_id=_candidate_id(plan),
                status=status,
                failure_class=failure_class,
                elapsed_wall_time_ms=elapsed,
                timed_out=timed_out,
            )
        )
        last_result = result

        if timed_out and active_config.fallback_on_timeout:
            _mark_result_failed(result, "timeout", used_fallback=index > 0)
            continue
        if status in SUCCESS_STATUSES and not timed_out:
            if index > 0:
                result.status = "fallback_success"
                result.trace.status = "fallback_success"
            return _finalize(
                result,
                records,
                used_fallback=index > 0,
                guard_status="success",
                config=active_config,
            )
        if not active_config.fallback_on_failed_status:
            return _finalize(
                result,
                records,
                used_fallback=index > 0,
                guard_status="failed",
                config=active_config,
            )

    if last_result is not None:
        _mark_result_failed(last_result, last_result.trace.failure_class, used_fallback=len(records) > 1)
        return _finalize(
            last_result,
            records,
            used_fallback=len(records) > 1,
            guard_status="fallback_exhausted",
            config=active_config,
        )

    assert last_exception is not None
    result = _exception_result(attempts[min(len(records), len(attempts) - 1)], last_exception, records)
    return _finalize(
        result,
        records,
        used_fallback=len(records) > 1,
        guard_status="fallback_exhausted",
        config=active_config,
    )


def _finalize(
    result: SolveResult,
    records: list[GuardedAttemptRecord],
    *,
    used_fallback: bool,
    guard_status: str,
    config: RuntimeGuardConfig,
) -> GuardedSolveOutcome:
    payload = {
        "guard_status": guard_status,
        "used_fallback": used_fallback,
        "attempts": [asdict(record) for record in records],
        "config": asdict(config),
    }
    result.trace.fallback_events = [asdict(record) for record in records[1:]]
    result.trace.metadata["runtime_guard"] = payload
    result.metadata["runtime_guard"] = payload
    return GuardedSolveOutcome(
        result=result,
        attempts=tuple(records),
        used_fallback=used_fallback,
        guard_status=guard_status,
        config=config,
    )


def _exception_result(
    plan: PolicyPlan,
    exc: Exception,
    records: list[GuardedAttemptRecord],
) -> SolveResult:
    status = "fallback_failed" if len(records) > 1 else "failed"
    trace = RunTrace(
        run_id=f"guarded_exception:{plan.plan_id}",
        plan_id=plan.plan_id,
        backend=plan.backend,
        status=status,
        failure_class="exception",
        metadata={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
    )
    return SolveResult(status=status, solution=None, trace=trace)


def _mark_result_failed(
    result: SolveResult,
    failure_class: str | None,
    *,
    used_fallback: bool,
) -> None:
    result.status = "fallback_failed" if used_fallback else "failed"
    result.trace.status = result.status
    result.trace.failure_class = failure_class or "not_converged"


def _attempt_timed_out(config: RuntimeGuardConfig, elapsed_ms: float) -> bool:
    return (
        config.max_attempt_wall_time_ms is not None
        and elapsed_ms > config.max_attempt_wall_time_ms
    )


def _total_budget_exhausted(config: RuntimeGuardConfig, total_start: float) -> bool:
    return (
        config.max_total_wall_time_ms is not None
        and _elapsed_ms(total_start) > config.max_total_wall_time_ms
    )


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000.0


def _candidate_id(plan: PolicyPlan) -> str | None:
    value = plan.audit.get("candidate_id")
    return None if value is None else str(value)
