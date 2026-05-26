"""Core data contracts for GPU-first solve planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


OperatorKind = Literal[
    "matrix_free",
    "structured_stencil",
    "block_sparse",
    "assembled_sparse",
]


@dataclass(frozen=True)
class LinearOperatorSpec:
    """Metadata for an operator without forcing CPU matrix materialization."""

    operator_id: str
    kind: OperatorKind
    shape: tuple[int, int]
    dtype: str = "float32"
    symmetry: str = "unknown"
    device_resident: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LinearSystem:
    """A linear system described by an operator spec and optional RHS metadata."""

    system_id: str
    operator: LinearOperatorSpec
    rhs_id: str | None = None
    family_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveContext:
    """Numerical and runtime constraints for one solve."""

    context_id: str
    tolerance_abs: float = 1.0e-8
    tolerance_rel: float = 1.0e-6
    max_iter: int = 1000
    precision: str = "float32"
    required_backend: str = "taichi_gpu"
    allow_cpu_reference: bool = False
    budget_time_ms: float | None = None
    budget_memory_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImplicitSolveContext(SolveContext):
    """Solve context for repeated implicit-loop subproblems."""

    timestep_id: str | None = None
    newton_iter: int | None = None
    repeated_solve_group_id: str | None = None
    warm_start: bool = True
    reuse_operator_buffers: bool = True

