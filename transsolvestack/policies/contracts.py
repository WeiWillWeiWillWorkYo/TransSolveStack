"""Policy interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from transsolvestack.core.result import PolicyPlan
from transsolvestack.core.types import LinearSystem, SolveContext


class Policy(ABC):
    """Base class for all auditable policies."""

    name: str

    @abstractmethod
    def explain(self) -> dict:
        """Return serializable policy metadata."""


class SolverPolicy(Policy):
    @abstractmethod
    def select_solver(self, system: LinearSystem, context: SolveContext) -> dict:
        """Select solver configuration."""


class PreconditionerPolicy(Policy):
    @abstractmethod
    def select_preconditioner(
        self, system: LinearSystem, context: SolveContext
    ) -> dict:
        """Select preconditioner configuration."""


class BackendDispatchPolicy(Policy):
    @abstractmethod
    def select_backend(self, system: LinearSystem, context: SolveContext) -> str:
        """Select backend name."""


class FallbackPolicy(Policy):
    @abstractmethod
    def fallback_chain(self, system: LinearSystem, context: SolveContext) -> list[dict]:
        """Return fallback configurations."""


class PolicyPlanner(ABC):
    @abstractmethod
    def plan(self, system: LinearSystem, context: SolveContext) -> PolicyPlan:
        """Build an executable PolicyPlan."""

