"""Solver registry and Taichi solver defaults."""

from __future__ import annotations

from collections.abc import Callable

from transsolvestack.solvers.base import Solver
from transsolvestack.solvers.taichi_cg import (
    TaichiBiCGSTABSolver,
    TaichiChebyshevSolver,
    TaichiCGSolver,
    TaichiGMRESSolver,
    TaichiPCGSolver,
    TaichiRichardsonSolver,
)


SolverFactory = Callable[[], Solver]


class SolverRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, SolverFactory] = {}

    def register(self, name: str, factory: SolverFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> Solver:
        try:
            return self._factories[name]()
        except KeyError as exc:
            raise ValueError(f"unknown solver: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._factories)


def default_solver_registry() -> SolverRegistry:
    registry = SolverRegistry()
    registry.register("cg", TaichiCGSolver)
    registry.register("pcg", TaichiPCGSolver)
    registry.register("bicgstab", TaichiBiCGSTABSolver)
    registry.register("gmres", TaichiGMRESSolver)
    registry.register("richardson", TaichiRichardsonSolver)
    registry.register("chebyshev", TaichiChebyshevSolver)
    return registry
