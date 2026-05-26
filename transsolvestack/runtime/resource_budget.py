"""Resource budget guards for shared CPU/GPU machines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBudget:
    """Conservative limits used before allocating device-heavy workloads."""

    mode: str = "coexistence"
    max_gpu_memory_fraction: float = 0.20
    max_host_threads: int = 2
    max_wall_time_s: float = 30.0
    max_problem_n: int = 65_536
    max_effective_nnz: int = 1_000_000
    max_default_iterations: int = 300
    max_dataset_download_gb: float = 2.0
    require_confirmation_above_smoke: bool = True
    allow_background_benchmark: bool = False

    def validate_problem_size(self, n: int, effective_nnz: int | None = None) -> None:
        """Fail closed if a planned operator exceeds this budget."""

        if n > self.max_problem_n:
            raise ResourceBudgetExceeded(
                f"problem size {n} exceeds budget max_problem_n={self.max_problem_n}"
            )
        if effective_nnz is not None and effective_nnz > self.max_effective_nnz:
            raise ResourceBudgetExceeded(
                "effective nnz "
                f"{effective_nnz} exceeds budget max_effective_nnz="
                f"{self.max_effective_nnz}"
            )

    def validate_iterations(self, max_iter: int) -> None:
        if max_iter > self.max_default_iterations:
            raise ResourceBudgetExceeded(
                f"max_iter {max_iter} exceeds budget max_default_iterations="
                f"{self.max_default_iterations}"
            )

    def validate_download_size(self, download_gb: float) -> None:
        if download_gb > self.max_dataset_download_gb:
            raise ResourceBudgetExceeded(
                f"download size {download_gb} GB exceeds budget "
                f"max_dataset_download_gb={self.max_dataset_download_gb}"
            )


class ResourceBudgetExceeded(RuntimeError):
    """Raised when a planned workload would exceed the active resource budget."""

