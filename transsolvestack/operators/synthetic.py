"""Synthetic operator family metadata.

This module does not materialize matrices and does not initialize Taichi. It
only builds auditable LinearSystem metadata for dry-run planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

from transsolvestack.core.types import LinearOperatorSpec, LinearSystem, OperatorKind


@dataclass(frozen=True)
class SyntheticOperatorFamily:
    family_id: str
    operator_kind: OperatorKind
    dimensions: int
    symmetry: str
    entries_per_row: int
    dofs_per_node: int = 1
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def unknowns(self, size: tuple[int, ...]) -> int:
        self.validate_size(size)
        return int(prod(size)) * self.dofs_per_node

    def effective_nnz(self, size: tuple[int, ...]) -> int:
        return self.unknowns(size) * self.entries_per_row

    def validate_size(self, size: tuple[int, ...]) -> None:
        if len(size) != self.dimensions:
            raise ValueError(
                f"{self.family_id} expects {self.dimensions} dimensions, got {size}"
            )
        if any(int(dim) <= 0 for dim in size):
            raise ValueError(f"{self.family_id} size must be positive, got {size}")


@dataclass(frozen=True)
class SyntheticSystemSpec:
    system: LinearSystem
    family: SyntheticOperatorFamily
    grid_size: tuple[int, ...]
    estimated_unknowns: int
    estimated_effective_nnz: int


_FAMILIES: dict[str, SyntheticOperatorFamily] = {
    "poisson_2d_stencil": SyntheticOperatorFamily(
        family_id="poisson_2d_stencil",
        operator_kind="structured_stencil",
        dimensions=2,
        symmetry="spd",
        entries_per_row=5,
        description="2D five-point Poisson stencil.",
    ),
    "anisotropic_diffusion_2d": SyntheticOperatorFamily(
        family_id="anisotropic_diffusion_2d",
        operator_kind="structured_stencil",
        dimensions=2,
        symmetry="spd",
        entries_per_row=5,
        description="2D anisotropic diffusion stencil with metadata-only coefficients.",
        metadata={"anisotropy": "configurable"},
    ),
    "strong_anisotropic_diffusion_2d": SyntheticOperatorFamily(
        family_id="strong_anisotropic_diffusion_2d",
        operator_kind="structured_stencil",
        dimensions=2,
        symmetry="spd",
        entries_per_row=5,
        description="2D strong anisotropic diffusion stencil.",
        metadata={"anisotropy": "strong"},
    ),
    "poisson_3d_stencil": SyntheticOperatorFamily(
        family_id="poisson_3d_stencil",
        operator_kind="structured_stencil",
        dimensions=3,
        symmetry="spd",
        entries_per_row=7,
        description="3D seven-point Poisson stencil.",
    ),
    "linear_elasticity_block_2d": SyntheticOperatorFamily(
        family_id="linear_elasticity_block_2d",
        operator_kind="block_sparse",
        dimensions=2,
        symmetry="spd",
        entries_per_row=18,
        dofs_per_node=2,
        description="2D vector-valued block operator metadata for elasticity-like tests.",
    ),
}


def known_synthetic_families() -> dict[str, SyntheticOperatorFamily]:
    return dict(_FAMILIES)


def get_synthetic_family(family_id: str) -> SyntheticOperatorFamily:
    try:
        return _FAMILIES[family_id]
    except KeyError as exc:
        raise ValueError(f"unknown synthetic family: {family_id}") from exc


def build_synthetic_system(
    family_id: str,
    size: tuple[int, ...],
    dtype: str = "float32",
    expected_operator_kind: str | None = None,
) -> SyntheticSystemSpec:
    """Build LinearSystem metadata for a synthetic operator instance."""

    family = get_synthetic_family(family_id)
    if expected_operator_kind is not None and expected_operator_kind != family.operator_kind:
        raise ValueError(
            f"workload operator_kind {expected_operator_kind} does not match "
            f"registered kind {family.operator_kind} for {family_id}"
        )
    unknowns = family.unknowns(size)
    effective_nnz = family.effective_nnz(size)
    size_label = "x".join(str(dim) for dim in size)
    operator_id = f"{family_id}:{size_label}:{dtype}"
    operator = LinearOperatorSpec(
        operator_id=operator_id,
        kind=family.operator_kind,
        shape=(unknowns, unknowns),
        dtype=dtype,
        symmetry=family.symmetry,
        device_resident=True,
        metadata={
            "synthetic": True,
            "family_id": family_id,
            "grid_size": size,
            "dofs_per_node": family.dofs_per_node,
            "estimated_effective_nnz": effective_nnz,
            "description": family.description,
            **family.metadata,
        },
    )
    system = LinearSystem(
        system_id=f"synthetic:{operator_id}",
        operator=operator,
        family_id=family_id,
        metadata={
            "estimated_unknowns": unknowns,
            "estimated_effective_nnz": effective_nnz,
            "materialized": False,
        },
    )
    return SyntheticSystemSpec(
        system=system,
        family=family,
        grid_size=size,
        estimated_unknowns=unknowns,
        estimated_effective_nnz=effective_nnz,
    )
