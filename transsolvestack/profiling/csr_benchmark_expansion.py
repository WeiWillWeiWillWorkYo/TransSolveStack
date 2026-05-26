"""Resource-safe planning for expanding real CSR benchmark coverage."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_BENCHMARK_EXPANSION_SCHEMA_VERSION = "phase1_csr_benchmark_expansion_plan_v1"


@dataclass(frozen=True)
class CsrBenchmarkMatrixPlan:
    schema_version: str
    plan_id: str
    matrix_id: str
    selection_rank: int
    group: str
    name: str
    n_rows: int
    n_cols: int
    nnz: int
    size_bucket: str
    kind: str
    symmetry_class: str
    likely_symmetric: bool
    likely_spd: bool
    candidate_profile: str
    local_path: str
    archive_size_bytes: int
    planned_import: bool
    planned_gpu_benchmark: bool
    skipped_reason: str | None


@dataclass(frozen=True)
class CsrBenchmarkCandidatePlan:
    schema_version: str
    plan_id: str
    matrix_id: str
    candidate_id: str
    solver: str
    preconditioner: str
    precision: str
    solver_parameters: dict[str, Any]
    measurement_repeats: int
    max_iter: int
    tolerance_rel: float
    requires_cpu_screen: bool
    estimated_matvecs: int
    estimated_nnz_visits: int
    planned_status: str


@dataclass(frozen=True)
class CsrBenchmarkExpansionSummary:
    status: str
    schema_version: str
    plan_id: str
    source_selection_path: str
    source_selector_rows_path: str
    resource_limits_path: str
    runtime_selector_changed: bool
    executes_gpu: bool
    selected_source_matrices: int
    already_profiled_matrices: int
    eligible_unprofiled_matrices: int
    planned_matrices: int
    planned_candidate_jobs: int
    planned_gpu_solve_attempts: int
    measurement_repeats: int
    max_new_matrices: int
    max_planned_gpu_solves: int
    max_rows: int
    max_cols: int
    max_nnz: int
    max_archive_size_bytes: int
    max_iter: int
    tolerance_rel: float
    total_planned_nnz: int
    estimated_total_nnz_visits: int
    by_candidate_profile: dict[str, int]
    by_solver: dict[str, int]
    skipped_existing_matrices: int
    skipped_resource_limit_matrices: int
    skipped_budget_matrices: int
    next_step: str


@dataclass(frozen=True)
class CsrBenchmarkExpansionPlan:
    matrix_rows: tuple[CsrBenchmarkMatrixPlan, ...]
    candidate_rows: tuple[CsrBenchmarkCandidatePlan, ...]
    summary: CsrBenchmarkExpansionSummary
    schema: dict[str, Any]


def build_csr_benchmark_expansion_plan_from_files(
    selection_path: str | Path,
    selector_rows_path: str | Path,
    resource_limits_path: str | Path,
    *,
    plan_id: str = "phase1_csr_benchmark_expansion_m49",
    max_new_matrices: int = 8,
    max_planned_gpu_solves: int = 24,
    measurement_repeats: int = 1,
    max_rows: int = 10_000,
    max_cols: int = 10_000,
    max_nnz: int = 100_000,
    max_archive_size_bytes: int = 8_000_000,
    max_iter: int = 256,
    tolerance_rel: float = 1.0e-5,
) -> CsrBenchmarkExpansionPlan:
    selection_rows = tuple(read_jsonl(selection_path))
    selector_rows = tuple(read_jsonl(selector_rows_path))
    limits = _read_resource_limits(resource_limits_path)
    return build_csr_benchmark_expansion_plan(
        selection_rows,
        selector_rows,
        limits,
        source_selection_path=str(selection_path),
        source_selector_rows_path=str(selector_rows_path),
        resource_limits_path=str(resource_limits_path),
        plan_id=plan_id,
        max_new_matrices=max_new_matrices,
        max_planned_gpu_solves=max_planned_gpu_solves,
        measurement_repeats=measurement_repeats,
        max_rows=max_rows,
        max_cols=max_cols,
        max_nnz=max_nnz,
        max_archive_size_bytes=max_archive_size_bytes,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )


def build_csr_benchmark_expansion_plan(
    selection_rows: Iterable[dict[str, Any]],
    selector_rows: Iterable[dict[str, Any]],
    resource_limits: dict[str, Any],
    *,
    source_selection_path: str,
    source_selector_rows_path: str,
    resource_limits_path: str,
    plan_id: str = "phase1_csr_benchmark_expansion_m49",
    max_new_matrices: int = 8,
    max_planned_gpu_solves: int = 24,
    measurement_repeats: int = 1,
    max_rows: int = 10_000,
    max_cols: int = 10_000,
    max_nnz: int = 100_000,
    max_archive_size_bytes: int = 8_000_000,
    max_iter: int = 256,
    tolerance_rel: float = 1.0e-5,
) -> CsrBenchmarkExpansionPlan:
    limits = dict(resource_limits.get("limits", resource_limits))
    _validate_resource_budget(
        limits,
        measurement_repeats=measurement_repeats,
        max_rows=max_rows,
        max_cols=max_cols,
        max_nnz=max_nnz,
        max_iter=max_iter,
    )
    selected = tuple(sorted(selection_rows, key=lambda row: int(row["selection_rank"])))
    profiled_ids = {str(row["matrix_id"]) for row in selector_rows}
    eligible = tuple(
        row
        for row in selected
        if str(row["matrix_id"]) not in profiled_ids
        and _within_resource_limits(
            row,
            max_rows=max_rows,
            max_cols=max_cols,
            max_nnz=max_nnz,
            max_archive_size_bytes=max_archive_size_bytes,
        )
    )
    planned_source = _choose_balanced_matrices(eligible, max_new_matrices)
    matrix_rows = tuple(
        _matrix_plan(row, plan_id=plan_id, skipped_reason=None)
        for row in planned_source
    )
    candidate_rows = _candidate_plans(
        matrix_rows,
        plan_id=plan_id,
        measurement_repeats=measurement_repeats,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
        max_planned_gpu_solves=max_planned_gpu_solves,
    )
    planned_matrix_ids = {row.matrix_id for row in matrix_rows}
    skipped_existing = sum(1 for row in selected if str(row["matrix_id"]) in profiled_ids)
    skipped_resource = sum(
        1
        for row in selected
        if str(row["matrix_id"]) not in profiled_ids
        and not _within_resource_limits(
            row,
            max_rows=max_rows,
            max_cols=max_cols,
            max_nnz=max_nnz,
            max_archive_size_bytes=max_archive_size_bytes,
        )
    )
    skipped_budget = sum(
        1
        for row in eligible
        if str(row["matrix_id"]) not in planned_matrix_ids
    )
    total_nnz = sum(row.nnz for row in matrix_rows)
    summary = CsrBenchmarkExpansionSummary(
        status="passed" if matrix_rows and candidate_rows else "failed",
        schema_version=CSR_BENCHMARK_EXPANSION_SCHEMA_VERSION,
        plan_id=plan_id,
        source_selection_path=source_selection_path,
        source_selector_rows_path=source_selector_rows_path,
        resource_limits_path=resource_limits_path,
        runtime_selector_changed=False,
        executes_gpu=False,
        selected_source_matrices=len(selected),
        already_profiled_matrices=len(profiled_ids),
        eligible_unprofiled_matrices=len(eligible),
        planned_matrices=len(matrix_rows),
        planned_candidate_jobs=len(candidate_rows),
        planned_gpu_solve_attempts=len(candidate_rows) * measurement_repeats,
        measurement_repeats=measurement_repeats,
        max_new_matrices=max_new_matrices,
        max_planned_gpu_solves=max_planned_gpu_solves,
        max_rows=max_rows,
        max_cols=max_cols,
        max_nnz=max_nnz,
        max_archive_size_bytes=max_archive_size_bytes,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
        total_planned_nnz=total_nnz,
        estimated_total_nnz_visits=sum(row.estimated_nnz_visits for row in candidate_rows),
        by_candidate_profile=dict(Counter(row.candidate_profile for row in matrix_rows)),
        by_solver=dict(Counter(row.solver for row in candidate_rows)),
        skipped_existing_matrices=skipped_existing,
        skipped_resource_limit_matrices=skipped_resource,
        skipped_budget_matrices=skipped_budget,
        next_step="import_planned_matrices_then_run_cpu_screened_taichi_csr_micro_campaign",
    )
    return CsrBenchmarkExpansionPlan(
        matrix_rows=matrix_rows,
        candidate_rows=candidate_rows,
        summary=summary,
        schema=_schema(),
    )


def write_csr_benchmark_matrix_plan(
    rows: Iterable[CsrBenchmarkMatrixPlan],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_benchmark_candidate_plan(
    rows: Iterable[CsrBenchmarkCandidatePlan],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_benchmark_expansion_summary(
    summary: CsrBenchmarkExpansionSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_benchmark_expansion_schema(schema: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_benchmark_expansion_report(
    plan: CsrBenchmarkExpansionPlan,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = plan.summary
    lines = [
        "# CSR Benchmark Expansion Plan",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- plan_id: `{summary.plan_id}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        f"- executes_gpu: `{summary.executes_gpu}`",
        f"- selected_source_matrices: `{summary.selected_source_matrices}`",
        f"- already_profiled_matrices: `{summary.already_profiled_matrices}`",
        f"- eligible_unprofiled_matrices: `{summary.eligible_unprofiled_matrices}`",
        f"- planned_matrices: `{summary.planned_matrices}`",
        f"- planned_candidate_jobs: `{summary.planned_candidate_jobs}`",
        f"- planned_gpu_solve_attempts: `{summary.planned_gpu_solve_attempts}`",
        f"- total_planned_nnz: `{summary.total_planned_nnz}`",
        f"- estimated_total_nnz_visits: `{summary.estimated_total_nnz_visits}`",
        f"- by_candidate_profile: `{summary.by_candidate_profile}`",
        f"- by_solver: `{summary.by_solver}`",
        f"- skipped_existing_matrices: `{summary.skipped_existing_matrices}`",
        f"- skipped_resource_limit_matrices: `{summary.skipped_resource_limit_matrices}`",
        f"- skipped_budget_matrices: `{summary.skipped_budget_matrices}`",
        f"- next_step: `{summary.next_step}`",
        "",
        "## Matrix Queue",
        "",
        "| rank | matrix | shape | nnz | profile | symmetry | archive_mb |",
        "|---:|---|---:|---:|---|---|---:|",
    ]
    for row in plan.matrix_rows:
        lines.append(
            "| "
            f"{row.selection_rank} | "
            f"{row.matrix_id} | "
            f"{row.n_rows}x{row.n_cols} | "
            f"{row.nnz} | "
            f"{row.candidate_profile} | "
            f"{row.symmetry_class} | "
            f"{row.archive_size_bytes / 1_000_000:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Queue",
            "",
            "| matrix | candidate | solver | preconditioner | repeats | max_iter | estimated_nnz_visits |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in plan.candidate_rows:
        lines.append(
            "| "
            f"{row.matrix_id} | "
            f"{row.candidate_id} | "
            f"{row.solver} | "
            f"{row.preconditioner} | "
            f"{row.measurement_repeats} | "
            f"{row.max_iter} | "
            f"{row.estimated_nnz_visits} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _validate_resource_budget(
    limits: dict[str, Any],
    *,
    measurement_repeats: int,
    max_rows: int,
    max_cols: int,
    max_nnz: int,
    max_iter: int,
) -> None:
    if measurement_repeats < 1:
        raise ValueError("measurement_repeats must be >= 1")
    if int(limits.get("allow_background_benchmark", False)) != 0:
        raise ValueError("M49 expects background benchmark execution to be disabled")
    if max_rows > int(limits.get("max_problem_n", max_rows)):
        raise ValueError("max_rows exceeds resource limit")
    if max_cols > int(limits.get("max_problem_n", max_cols)):
        raise ValueError("max_cols exceeds resource limit")
    if max_nnz > int(limits.get("max_effective_nnz", max_nnz)):
        raise ValueError("max_nnz exceeds resource limit")
    if max_iter > int(limits.get("max_default_iterations", max_iter)):
        raise ValueError("max_iter exceeds resource limit")


def _within_resource_limits(
    row: dict[str, Any],
    *,
    max_rows: int,
    max_cols: int,
    max_nnz: int,
    max_archive_size_bytes: int,
) -> bool:
    return (
        int(row["n_rows"]) <= max_rows
        and int(row["n_cols"]) <= max_cols
        and int(row["nnz"]) <= max_nnz
        and int(row.get("archive_size_bytes") or 0) <= max_archive_size_bytes
        and Path(str(row["local_path"])).exists()
    )


def _choose_balanced_matrices(
    rows: tuple[dict[str, Any], ...],
    max_new_matrices: int,
) -> tuple[dict[str, Any], ...]:
    symmetric = tuple(row for row in rows if _likely_symmetric(row))
    general = tuple(row for row in rows if not _likely_symmetric(row))
    symmetric_quota = min(len(symmetric), max_new_matrices // 2)
    general_quota = min(len(general), max_new_matrices - symmetric_quota)
    chosen = list(symmetric[:symmetric_quota])
    chosen.extend(general[:general_quota])
    if len(chosen) < max_new_matrices:
        chosen_ids = {str(row["matrix_id"]) for row in chosen}
        for row in rows:
            if str(row["matrix_id"]) not in chosen_ids:
                chosen.append(row)
                chosen_ids.add(str(row["matrix_id"]))
                if len(chosen) >= max_new_matrices:
                    break
    return tuple(sorted(chosen, key=lambda row: int(row["selection_rank"])))


def _matrix_plan(
    row: dict[str, Any],
    *,
    plan_id: str,
    skipped_reason: str | None,
) -> CsrBenchmarkMatrixPlan:
    likely_symmetric = _likely_symmetric(row)
    likely_spd = bool(row.get("is_pos_def"))
    return CsrBenchmarkMatrixPlan(
        schema_version=CSR_BENCHMARK_EXPANSION_SCHEMA_VERSION,
        plan_id=plan_id,
        matrix_id=str(row["matrix_id"]),
        selection_rank=int(row["selection_rank"]),
        group=str(row["group"]),
        name=str(row["name"]),
        n_rows=int(row["n_rows"]),
        n_cols=int(row["n_cols"]),
        nnz=int(row["nnz"]),
        size_bucket=str(row["size_bucket"]),
        kind=str(row["kind"]),
        symmetry_class=str(row["symmetry_class"]),
        likely_symmetric=likely_symmetric,
        likely_spd=likely_spd,
        candidate_profile=("symmetric" if likely_symmetric else "general"),
        local_path=str(row["local_path"]),
        archive_size_bytes=int(row.get("archive_size_bytes") or 0),
        planned_import=True,
        planned_gpu_benchmark=True,
        skipped_reason=skipped_reason,
    )


def _candidate_plans(
    matrix_rows: tuple[CsrBenchmarkMatrixPlan, ...],
    *,
    plan_id: str,
    measurement_repeats: int,
    max_iter: int,
    tolerance_rel: float,
    max_planned_gpu_solves: int,
) -> tuple[CsrBenchmarkCandidatePlan, ...]:
    rows: list[CsrBenchmarkCandidatePlan] = []
    for matrix in matrix_rows:
        for candidate in _candidate_templates(matrix):
            if len(rows) >= max_planned_gpu_solves:
                return tuple(rows)
            rows.append(
                CsrBenchmarkCandidatePlan(
                    schema_version=CSR_BENCHMARK_EXPANSION_SCHEMA_VERSION,
                    plan_id=plan_id,
                    matrix_id=matrix.matrix_id,
                    candidate_id=str(candidate["candidate_id"]),
                    solver=str(candidate["solver"]),
                    preconditioner=str(candidate["preconditioner"]),
                    precision="float64",
                    solver_parameters=dict(candidate.get("solver_parameters", {})),
                    measurement_repeats=measurement_repeats,
                    max_iter=max_iter,
                    tolerance_rel=tolerance_rel,
                    requires_cpu_screen=True,
                    estimated_matvecs=_estimated_matvecs(str(candidate["solver"]), max_iter),
                    estimated_nnz_visits=(
                        matrix.nnz
                        * _estimated_matvecs(str(candidate["solver"]), max_iter)
                        * measurement_repeats
                    ),
                    planned_status="queued_cpu_screen_required",
                )
            )
    return tuple(rows)


def _candidate_templates(matrix: CsrBenchmarkMatrixPlan) -> tuple[dict[str, Any], ...]:
    if matrix.likely_symmetric:
        return (
            {
                "candidate_id": "taichi_csr_cg_none_float64",
                "solver": "cg",
                "preconditioner": "none",
            },
            {
                "candidate_id": "taichi_csr_pcg_jacobi_float64",
                "solver": "pcg",
                "preconditioner": "jacobi",
            },
            {
                "candidate_id": "taichi_csr_richardson_jacobi_float64",
                "solver": "richardson",
                "preconditioner": "jacobi",
                "solver_parameters": {"omega_source": "cpu_screen_required"},
            },
        )
    return (
        {
            "candidate_id": "taichi_csr_bicgstab_none_float64",
            "solver": "bicgstab",
            "preconditioner": "none",
        },
        {
            "candidate_id": "taichi_csr_bicgstab_jacobi_float64",
            "solver": "bicgstab",
            "preconditioner": "jacobi",
        },
        {
            "candidate_id": "taichi_csr_gmres_jacobi_restart16_float64",
            "solver": "gmres",
            "preconditioner": "jacobi",
            "solver_parameters": {"restart": 16},
        },
    )


def _estimated_matvecs(solver: str, max_iter: int) -> int:
    if solver == "bicgstab":
        return max_iter * 2
    if solver == "gmres":
        return max_iter + max_iter // 16
    return max_iter


def _likely_symmetric(row: dict[str, Any]) -> bool:
    if str(row["symmetry_class"]) in {"spd", "symmetric"}:
        return True
    return (
        float(row.get("pattern_symmetry", 0.0)) >= 0.999
        and float(row.get("numerical_symmetry", 0.0)) >= 0.999
    )


def _read_resource_limits(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        return _read_resource_limits_without_yaml(text)
    return dict(yaml.safe_load(text))


def _read_resource_limits_without_yaml(text: str) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    active = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "limits:":
            active = True
            continue
        if active and ":" in stripped:
            key, raw_value = stripped.split(":", 1)
            limits[key] = _parse_scalar(raw_value.strip())
    return {"limits": limits}


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"')


def _schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_BENCHMARK_EXPANSION_SCHEMA_VERSION,
        "task": "resource_safe_real_csr_benchmark_scaleout_planning",
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "matrix_queue": {
            "planned_import": "matrix is selected for a future bounded CSR import",
            "planned_gpu_benchmark": "matrix is selected for future CPU-screened Taichi CSR solves",
        },
        "candidate_queue": {
            "planned_status": "queued_cpu_screen_required",
            "requires_cpu_screen": True,
            "measurement_repeats_default": 1,
        },
        "integration_boundary": {
            "status": "plan_only_no_benchmark_execution",
            "next_step": "execute_this_queue_with_existing_csr_import_and_solver_smoke_boundaries",
        },
    }
