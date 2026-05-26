"""Policy, trace, and result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SolveStatus = Literal[
    "success",
    "failed",
    "fallback_success",
    "fallback_failed",
    "dry_run",
]


@dataclass(frozen=True)
class PolicyPlan:
    """Auditable execution plan for a solve."""

    plan_id: str
    backend: str
    solver: dict[str, Any]
    preconditioner: dict[str, Any] = field(default_factory=dict)
    reuse: dict[str, Any] = field(default_factory=dict)
    coarsening: dict[str, Any] = field(default_factory=dict)
    correction: dict[str, Any] = field(default_factory=dict)
    fallback_chain: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunTrace:
    """Runtime trace with GPU-aware accounting."""

    run_id: str
    plan_id: str
    backend: str
    status: SolveStatus
    failure_class: str | None = None
    setup_time_ms: float | None = None
    solve_time_ms: float | None = None
    total_time_ms: float | None = None
    gpu_kernel_time_ms: float | None = None
    transfer_time_ms: float | None = None
    peak_device_memory_mb: float | None = None
    num_iterations: int | None = None
    final_residual_norm: float | None = None
    residual_history: list[float] = field(default_factory=list)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolveResult:
    """Result returned by the runtime."""

    status: SolveStatus
    solution: Any | None
    trace: RunTrace
    metadata: dict[str, Any] = field(default_factory=dict)
