"""Backend registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    device_type: str
    primary: bool = False
    supports_device_residency: bool = True
    supported_operator_kinds: tuple[str, ...] = ()
    supported_solvers: tuple[str, ...] = ()
    supported_preconditioners: tuple[str, ...] = ()
    timing_capabilities: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, BackendDescriptor] = {}

    def register(self, descriptor: BackendDescriptor) -> None:
        self._backends[descriptor.name] = descriptor

    def get(self, name: str) -> BackendDescriptor:
        return self._backends[name]

    def names(self) -> list[str]:
        return sorted(self._backends)


def default_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        BackendDescriptor(
            name="taichi_gpu",
            device_type="gpu",
            primary=True,
            supported_operator_kinds=(
                "matrix_free",
                "structured_stencil",
                "block_sparse",
                "assembled_sparse",
            ),
            supported_solvers=(
                "cg",
                "pcg",
                "bicgstab",
                "gmres",
                "richardson",
                "chebyshev",
            ),
            supported_preconditioners=(
                "none",
                "jacobi",
                "block_jacobi",
                "ilu0",
                "symmetric_equilibration",
                "row_column_equilibration",
            ),
            timing_capabilities=("wall_time", "kernel_time", "sync_time"),
        )
    )
    registry.register(
        BackendDescriptor(
            name="cpu_reference",
            device_type="cpu",
            primary=False,
            supports_device_residency=False,
            supported_operator_kinds=("assembled_sparse",),
            supported_solvers=("reference_direct",),
            supported_preconditioners=("none",),
            metadata={"purpose": "correctness_reference_only"},
        )
    )
    return registry
