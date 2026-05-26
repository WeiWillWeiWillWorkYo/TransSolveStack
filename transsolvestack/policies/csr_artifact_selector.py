"""CSR policy selector backed by real CSR selector rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.core.result import PolicyPlan
from transsolvestack.datasets.csr import CsrMatrix, csr_matrix_from_record
from transsolvestack.profiling.artifacts import read_jsonl


SUCCESS_STATUSES = {"success", "fallback_success"}


@dataclass(frozen=True)
class CsrProfiledCandidate:
    matrix_id: str
    context_id: str
    candidate_id: str
    solver: str
    preconditioner: str
    precision: str
    solver_parameters: dict[str, Any]
    status: str
    is_oracle: bool
    success_rate: float
    measurement_repeats: int
    solve_time_ms: float | None
    median_solve_time_ms: float | None
    solve_time_iqr_ms: float | None
    wall_time_ms: float | None
    regret_vs_oracle_ms: float | None
    num_iterations: int | None
    final_relative_residual: float | None
    cpu_recomputed_relative_residual: float | None
    solution_relative_error: float | None
    features: dict[str, Any]
    source_path: str

    @property
    def is_success(self) -> bool:
        return self.status in SUCCESS_STATUSES and self.success_rate >= 1.0


@dataclass(frozen=True)
class CsrCandidateSelection:
    plan: PolicyPlan
    candidate_id: str
    reason: str
    profile: CsrProfiledCandidate | None = None
    fallback_candidate_ids: tuple[str, ...] = ()


class CsrArtifactPolicySelector:
    """Select CSR solver candidates from measured selector-readiness rows.

    The selector is intentionally data-driven: it does not hard-code a finite
    solver list. Any candidate present in the selector rows can be selected or
    used as fallback if it passed the recorded numeric checks.
    """

    def __init__(
        self,
        profiles: Iterable[CsrProfiledCandidate],
        fallback_candidate_id: str | None = None,
    ) -> None:
        self._profiles = tuple(profiles)
        self._fallback_candidate_id = fallback_candidate_id
        if fallback_candidate_id and fallback_candidate_id not in {
            profile.candidate_id for profile in self._profiles
        }:
            raise ValueError(f"unknown CSR fallback candidate: {fallback_candidate_id}")

    @classmethod
    def from_selector_rows(
        cls,
        selector_rows_path: str | Path,
        fallback_candidate_id: str | None = None,
    ) -> "CsrArtifactPolicySelector":
        source_path = str(selector_rows_path)
        return cls(
            (
                _profile_from_row(row, source_path)
                for row in read_jsonl(selector_rows_path)
            ),
            fallback_candidate_id=fallback_candidate_id,
        )

    def select(
        self,
        csr: CsrMatrix | dict[str, Any],
        *,
        context_id: str = "phase1_csr_selector",
        objective: str = "min_solve_time_success",
    ) -> CsrCandidateSelection:
        matrix = _coerce_csr(csr)
        matching = tuple(
            profile
            for profile in self._profiles
            if profile.matrix_id == matrix.matrix_id and profile.context_id == context_id
        )
        feasible = tuple(profile for profile in matching if profile.is_success)
        if feasible:
            selected = _best_profile(feasible, objective)
            fallback_ids = tuple(
                profile.candidate_id
                for profile in sorted(feasible, key=_profile_sort_key)
                if profile.candidate_id != selected.candidate_id
            )
            return CsrCandidateSelection(
                plan=_build_plan(
                    profile=selected,
                    matrix_id=matrix.matrix_id,
                    context_id=context_id,
                    reason="profiled_success",
                    fallback_profiles=tuple(
                        profile
                        for profile in sorted(feasible, key=_profile_sort_key)
                        if profile.candidate_id != selected.candidate_id
                    ),
                ),
                candidate_id=selected.candidate_id,
                reason="profiled_success",
                profile=selected,
                fallback_candidate_ids=fallback_ids,
            )

        fallback = self._fallback_profile()
        return CsrCandidateSelection(
            plan=_build_plan(
                profile=fallback,
                matrix_id=matrix.matrix_id,
                context_id=context_id,
                reason="fallback_no_profile",
                fallback_profiles=tuple(
                    profile
                    for profile in self._profiles
                    if profile.candidate_id != fallback.candidate_id and profile.is_success
                ),
            ),
            candidate_id=fallback.candidate_id,
            reason="fallback_no_profile",
            profile=None,
            fallback_candidate_ids=tuple(
                profile.candidate_id
                for profile in self._profiles
                if profile.candidate_id != fallback.candidate_id and profile.is_success
            ),
        )

    def select_candidate(
        self,
        csr: CsrMatrix | dict[str, Any],
        candidate_id: str,
        *,
        context_id: str = "phase1_csr_selector",
        reason: str = "guarded_candidate",
    ) -> CsrCandidateSelection:
        """Build a plan for a specific profiled-success candidate.

        This is used by learned-policy guards after they have decided a learned
        recommendation is eligible for runtime. The method deliberately rejects
        screened-out, not-profiled, not-applicable, or unknown candidates so
        learned policy cannot bypass artifact-backed safety.
        """

        matrix = _coerce_csr(csr)
        matching = tuple(
            profile
            for profile in self._profiles
            if profile.matrix_id == matrix.matrix_id and profile.context_id == context_id
        )
        feasible = tuple(profile for profile in matching if profile.is_success)
        selected = next(
            (profile for profile in feasible if profile.candidate_id == candidate_id),
            None,
        )
        if selected is None:
            raise ValueError(
                "CSR learned guard candidate is not a profiled success: "
                f"{matrix.matrix_id}:{context_id}:{candidate_id}"
            )
        fallback_profiles = tuple(
            profile
            for profile in sorted(feasible, key=_profile_sort_key)
            if profile.candidate_id != selected.candidate_id
        )
        fallback_ids = tuple(profile.candidate_id for profile in fallback_profiles)
        return CsrCandidateSelection(
            plan=_build_plan(
                profile=selected,
                matrix_id=matrix.matrix_id,
                context_id=context_id,
                reason=reason,
                fallback_profiles=fallback_profiles,
            ),
            candidate_id=selected.candidate_id,
            reason=reason,
            profile=selected,
            fallback_candidate_ids=fallback_ids,
        )

    def _fallback_profile(self) -> CsrProfiledCandidate:
        feasible = tuple(profile for profile in self._profiles if profile.is_success)
        if not feasible:
            raise ValueError("CSR selector has no successful fallback candidates")
        if self._fallback_candidate_id:
            for profile in feasible:
                if profile.candidate_id == self._fallback_candidate_id:
                    return profile
        return sorted(feasible, key=_profile_sort_key)[0]


def _profile_from_row(row: dict[str, Any], source_path: str) -> CsrProfiledCandidate:
    return CsrProfiledCandidate(
        matrix_id=str(row["matrix_id"]),
        context_id=str(row["context_id"]),
        candidate_id=str(row["candidate_id"]),
        solver=str(row["solver"]),
        preconditioner=str(row["preconditioner"]),
        precision=str(row["precision"]),
        solver_parameters=dict(row.get("solver_parameters", {})),
        status=str(row["target_status"]),
        is_oracle=bool(row["label_is_oracle"]),
        success_rate=float(row.get("target_success_rate", 1.0)),
        measurement_repeats=int(row.get("target_measurement_repeats", 1)),
        solve_time_ms=_optional_float(row.get("target_solve_time_ms")),
        median_solve_time_ms=_optional_float(row.get("target_median_solve_time_ms")),
        solve_time_iqr_ms=_optional_float(row.get("target_solve_time_iqr_ms")),
        wall_time_ms=_optional_float(row.get("target_wall_time_ms")),
        regret_vs_oracle_ms=_optional_float(row.get("target_regret_vs_oracle_ms")),
        num_iterations=_optional_int(row.get("target_num_iterations")),
        final_relative_residual=_optional_float(
            row.get("target_final_relative_residual")
        ),
        cpu_recomputed_relative_residual=_optional_float(
            row.get("target_cpu_recomputed_relative_residual")
        ),
        solution_relative_error=_optional_float(
            row.get("target_solution_relative_error")
        ),
        features=dict(row.get("features", {})),
        source_path=source_path,
    )


def _build_plan(
    *,
    profile: CsrProfiledCandidate,
    matrix_id: str,
    context_id: str,
    reason: str,
    fallback_profiles: tuple[CsrProfiledCandidate, ...],
) -> PolicyPlan:
    return PolicyPlan(
        plan_id=(
            f"csr_artifact_policy:{matrix_id}:{context_id}:{profile.candidate_id}"
        ),
        backend="taichi_gpu",
        solver=_solver_payload(profile),
        preconditioner={"name": profile.preconditioner},
        reuse={
            "device_buffers": True,
            "warm_start": False,
        },
        fallback_chain=[_fallback_entry(item) for item in fallback_profiles],
        budget={},
        audit={
            "selector": "csr_artifact",
            "matrix_id": matrix_id,
            "context_id": context_id,
            "candidate_id": profile.candidate_id,
            "selection_reason": reason,
            "profile_source": profile.source_path,
            "profile_status": profile.status,
            "is_oracle": profile.is_oracle,
            "success_rate": profile.success_rate,
            "measurement_repeats": profile.measurement_repeats,
            "solve_time_ms": profile.solve_time_ms,
            "median_solve_time_ms": profile.median_solve_time_ms,
            "solve_time_iqr_ms": profile.solve_time_iqr_ms,
            "wall_time_ms": profile.wall_time_ms,
            "regret_vs_oracle_ms": profile.regret_vs_oracle_ms,
            "num_iterations": profile.num_iterations,
            "final_relative_residual": profile.final_relative_residual,
            "cpu_recomputed_relative_residual": (
                profile.cpu_recomputed_relative_residual
            ),
            "solution_relative_error": profile.solution_relative_error,
            "solver_parameters": dict(profile.solver_parameters),
        },
    )


def _fallback_entry(profile: CsrProfiledCandidate) -> dict[str, Any]:
    return {
        "candidate_id": profile.candidate_id,
        "solver": _solver_payload(profile),
        "preconditioner": {"name": profile.preconditioner},
        "reuse": {
            "device_buffers": True,
            "warm_start": False,
        },
    }


def _solver_payload(profile: CsrProfiledCandidate) -> dict[str, Any]:
    payload = {
        "name": profile.solver,
        "precision": profile.precision,
    }
    payload.update(profile.solver_parameters)
    return payload


def _best_profile(
    profiles: tuple[CsrProfiledCandidate, ...],
    objective: str,
) -> CsrProfiledCandidate:
    if objective != "min_solve_time_success":
        raise ValueError(f"unsupported CSR artifact policy objective: {objective}")
    return min(profiles, key=_profile_sort_key)


def _profile_sort_key(profile: CsrProfiledCandidate) -> tuple[float, float, str]:
    solve_time = (
        profile.median_solve_time_ms
        if profile.median_solve_time_ms is not None
        else profile.solve_time_ms
        if profile.solve_time_ms is not None
        else float("inf")
    )
    iqr = profile.solve_time_iqr_ms if profile.solve_time_iqr_ms is not None else 0.0
    return (solve_time, iqr, profile.candidate_id)


def _coerce_csr(csr: CsrMatrix | dict[str, Any]) -> CsrMatrix:
    if isinstance(csr, CsrMatrix):
        return csr
    return csr_matrix_from_record(csr)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
