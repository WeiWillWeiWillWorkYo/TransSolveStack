"""Preconditioner registry and Taichi defaults."""

from __future__ import annotations

from collections.abc import Callable

from transsolvestack.preconditioners.base import Preconditioner
from transsolvestack.preconditioners.identity import NoPreconditioner
from transsolvestack.preconditioners.taichi_jacobi import (
    TaichiBlockJacobiPreconditioner,
    TaichiIlu0Preconditioner,
    TaichiJacobiPreconditioner,
    TaichiRowColumnEquilibrationPreconditioner,
    TaichiSymmetricEquilibrationPreconditioner,
)


PreconditionerFactory = Callable[[], Preconditioner]


class PreconditionerRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, PreconditionerFactory] = {}

    def register(self, name: str, factory: PreconditionerFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> Preconditioner:
        try:
            return self._factories[name]()
        except KeyError as exc:
            raise ValueError(f"unknown preconditioner: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._factories)


def default_preconditioner_registry() -> PreconditionerRegistry:
    registry = PreconditionerRegistry()
    registry.register("none", NoPreconditioner)
    registry.register("jacobi", TaichiJacobiPreconditioner)
    registry.register("block_jacobi", TaichiBlockJacobiPreconditioner)
    registry.register("ilu0", TaichiIlu0Preconditioner)
    registry.register("symmetric_equilibration", TaichiSymmetricEquilibrationPreconditioner)
    registry.register("row_column_equilibration", TaichiRowColumnEquilibrationPreconditioner)
    return registry
