"""Policy selector backed by benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.benchmarks.candidates import CandidateConfig, CandidateSet
from transsolvestack.core.result import PolicyPlan
from transsolvestack.profiling.artifacts import read_jsonl


SUCCESS_STATUSES = {"success", "fallback_success"}


@dataclass(frozen=True)
class ProfiledCandidate:
    system_id: str
    context_id: str
    candidate_id: str
    status: str
    is_oracle: bool
    total_time_ms: float | None
    regret_vs_oracle: float | None
    num_iterations: int | None
    final_residual_norm: float | None
    relative_error_to_true: float | None
    source_path: str

    @property
    def is_success(self) -> bool:
        return self.status in SUCCESS_STATUSES


@dataclass(frozen=True)
class CandidateSelection:
    plan: PolicyPlan
    candidate_id: str
    reason: str
    profile: ProfiledCandidate | None = None
    fallback_candidate_ids: tuple[str, ...] = ()


class BenchmarkArtifactPolicySelector:
    """Create PolicyPlan objects from measured candidate evaluation artifacts."""

    def __init__(
        self,
        candidate_set: CandidateSet,
        profiles: Iterable[ProfiledCandidate],
        fallback_candidate_id: str | None = None,
    ) -> None:
        self.candidate_set = candidate_set
        self._candidates = {
            candidate.candidate_id: candidate for candidate in candidate_set.candidates
        }
        self._profiles = tuple(profiles)
        self._fallback_candidate_id = fallback_candidate_id
        if fallback_candidate_id and fallback_candidate_id not in self._candidates:
            raise ValueError(f"unknown fallback candidate: {fallback_candidate_id}")

    @classmethod
    def from_evaluation_artifact(
        cls,
        candidate_set: CandidateSet,
        evaluation_path: str | Path,
        fallback_candidate_id: str | None = None,
    ) -> "BenchmarkArtifactPolicySelector":
        source_path = str(evaluation_path)
        profiles = tuple(
            _profiled_candidate_from_row(row, source_path)
            for row in read_jsonl(evaluation_path)
        )
        return cls(
            candidate_set=candidate_set,
            profiles=profiles,
            fallback_candidate_id=fallback_candidate_id,
        )

    def select(
        self,
        system_id: str,
        context_id: str,
        objective: str = "min_total_time_success",
    ) -> CandidateSelection:
        matching_profiles = tuple(
            profile
            for profile in self._profiles
            if profile.system_id == system_id and profile.context_id == context_id
        )
        feasible_profiles = tuple(
            profile
            for profile in matching_profiles
            if profile.is_success and profile.candidate_id in self._candidates
        )
        if feasible_profiles:
            selected = _best_profile(feasible_profiles, objective)
            fallback_ids = tuple(
                profile.candidate_id
                for profile in sorted(feasible_profiles, key=_profile_sort_key)
                if profile.candidate_id != selected.candidate_id
            )
            return CandidateSelection(
                plan=self._build_plan(
                    candidate=self._candidates[selected.candidate_id],
                    system_id=system_id,
                    context_id=context_id,
                    reason="profiled_success",
                    profile=selected,
                    fallback_candidate_ids=fallback_ids,
                ),
                candidate_id=selected.candidate_id,
                reason="profiled_success",
                profile=selected,
                fallback_candidate_ids=fallback_ids,
            )

        fallback = self._fallback_candidate()
        fallback_ids = tuple(
            candidate.candidate_id
            for candidate in self.candidate_set.candidates
            if candidate.candidate_id != fallback.candidate_id
        )
        return CandidateSelection(
            plan=self._build_plan(
                candidate=fallback,
                system_id=system_id,
                context_id=context_id,
                reason="fallback_no_profile",
                profile=None,
                fallback_candidate_ids=fallback_ids,
            ),
            candidate_id=fallback.candidate_id,
            reason="fallback_no_profile",
            profile=None,
            fallback_candidate_ids=fallback_ids,
        )

    def _fallback_candidate(self) -> CandidateConfig:
        if self._fallback_candidate_id:
            return self._candidates[self._fallback_candidate_id]
        return self.candidate_set.candidates[0]

    def _build_plan(
        self,
        candidate: CandidateConfig,
        system_id: str,
        context_id: str,
        reason: str,
        profile: ProfiledCandidate | None,
        fallback_candidate_ids: tuple[str, ...],
    ) -> PolicyPlan:
        return PolicyPlan(
            plan_id=(
                f"artifact_policy:{self.candidate_set.candidate_set_id}:"
                f"{system_id}:{context_id}:{candidate.candidate_id}"
            ),
            backend=self.candidate_set.backend,
            solver=dict(candidate.solver),
            preconditioner=dict(candidate.preconditioner) or {"name": "none"},
            reuse=dict(candidate.reuse),
            fallback_chain=[
                _fallback_entry(self._candidates[candidate_id])
                for candidate_id in fallback_candidate_ids
                if candidate_id in self._candidates
            ],
            budget=_budget_from_candidate(candidate),
            audit={
                "selector": "benchmark_artifact",
                "candidate_set_id": self.candidate_set.candidate_set_id,
                "system_id": system_id,
                "context_id": context_id,
                "candidate_id": candidate.candidate_id,
                "selection_reason": reason,
                "profile_source": profile.source_path if profile else None,
                "profile_status": profile.status if profile else None,
                "is_oracle": profile.is_oracle if profile else False,
                "total_time_ms": profile.total_time_ms if profile else None,
                "regret_vs_oracle": profile.regret_vs_oracle if profile else None,
                "num_iterations": profile.num_iterations if profile else None,
                "final_residual_norm": profile.final_residual_norm if profile else None,
                "relative_error_to_true": (
                    profile.relative_error_to_true if profile else None
                ),
            },
        )


def _profiled_candidate_from_row(
    row: dict[str, Any],
    source_path: str,
) -> ProfiledCandidate:
    return ProfiledCandidate(
        system_id=str(row["system_id"]),
        context_id=str(row["context_id"]),
        candidate_id=str(row["candidate_id"]),
        status=str(row["status"]),
        is_oracle=bool(row.get("is_oracle", False)),
        total_time_ms=_optional_float(row.get("total_time_ms")),
        regret_vs_oracle=_optional_float(row.get("regret_vs_oracle")),
        num_iterations=_optional_int(row.get("num_iterations")),
        final_residual_norm=_optional_float(row.get("final_residual_norm")),
        relative_error_to_true=_optional_float(row.get("relative_error_to_true")),
        source_path=source_path,
    )


def _best_profile(
    profiles: tuple[ProfiledCandidate, ...],
    objective: str,
) -> ProfiledCandidate:
    if objective != "min_total_time_success":
        raise ValueError(f"unsupported artifact policy objective: {objective}")
    return min(profiles, key=_profile_sort_key)


def _profile_sort_key(profile: ProfiledCandidate) -> tuple[float, float, str]:
    time = profile.total_time_ms if profile.total_time_ms is not None else float("inf")
    regret = (
        profile.regret_vs_oracle
        if profile.regret_vs_oracle is not None
        else float("inf")
    )
    return (time, regret, profile.candidate_id)


def _fallback_entry(candidate: CandidateConfig) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "solver": dict(candidate.solver),
        "preconditioner": dict(candidate.preconditioner) or {"name": "none"},
        "reuse": dict(candidate.reuse),
    }


def _budget_from_candidate(candidate: CandidateConfig) -> dict[str, Any]:
    if candidate.max_iter is None:
        return {}
    return {"max_iter": candidate.max_iter}


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
