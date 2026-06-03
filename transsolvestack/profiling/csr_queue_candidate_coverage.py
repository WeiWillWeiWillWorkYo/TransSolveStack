"""Diagnose candidate and solver coverage for rolling CSR queue batches."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.backends.registry import default_backend_registry
from transsolvestack.preconditioners.registry import default_preconditioner_registry
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.solvers.registry import default_solver_registry


CSR_QUEUE_CANDIDATE_COVERAGE_SCHEMA_VERSION = (
    "phase1_csr_queue_candidate_coverage_v1"
)


@dataclass(frozen=True)
class CsrQueueBatchCoverageRow:
    schema_version: str
    batch_id: str
    batch_rank: int
    batch_outcome: str
    planned_total_nnz: int
    planned_max_matrix_nnz: int
    candidate_jobs: int
    gpu_success_rows: int
    cpu_screened_out_rows: int
    gpu_failed_rows: int
    selector_oracle_rows: int
    success_rate: float
    screen_rate: float
    low_success: bool
    screen_only_no_oracle: bool


@dataclass(frozen=True)
class CsrQueueCandidateCoverageRow:
    schema_version: str
    candidate_key: str
    candidate_id: str
    solver: str
    preconditioner: str
    solver_parameters_key: str
    attempted_rows: int
    success_rows: int
    screened_out_rows: int
    gpu_failed_rows: int
    success_rate: float
    screen_rate: float
    source_batch_count: int
    source_batches: tuple[str, ...]
    current_queue_candidate: bool


@dataclass(frozen=True)
class CsrQueueCandidateGapRow:
    schema_version: str
    gap_id: str
    priority: str
    candidate_profile: str
    proposed_candidate_id: str
    solver: str
    preconditioner: str
    solver_parameters: dict[str, Any]
    reason: str
    queue_ready: bool
    blocker: str | None
    next_action: str
    evidence: str


@dataclass(frozen=True)
class CsrQueueCandidateCoverageSummary:
    status: str
    schema_version: str
    runtime_selector_changed: bool
    executes_gpu: bool
    imports_matrices: bool
    source_batch_count: int
    completed_queue_batches: int
    next_pending_batch_id: str | None
    latest_batch_id: str
    latest_batch_outcome: str
    recent_window_batches: tuple[str, ...]
    recent_candidate_jobs: int
    recent_gpu_success_rows: int
    recent_cpu_screened_out_rows: int
    recent_selector_oracle_rows: int
    recent_success_rate: float
    recent_screen_only_batches: int
    trigger_candidate_coverage: bool
    trigger_reason: str | None
    current_queue_solvers: tuple[str, ...]
    current_queue_preconditioners: tuple[str, ...]
    current_queue_candidate_keys: tuple[str, ...]
    implemented_backend_solvers: tuple[str, ...]
    implemented_backend_preconditioners: tuple[str, ...]
    queue_missing_supported_solvers: tuple[str, ...]
    queue_missing_supported_preconditioners: tuple[str, ...]
    candidate_coverage_rows: int
    planned_gap_count: int
    queue_ready_gap_count: int
    cpu_screen_blocked_gap_count: int
    next_step: str


@dataclass(frozen=True)
class CsrQueueCandidateCoverage:
    batch_rows: tuple[CsrQueueBatchCoverageRow, ...]
    candidate_rows: tuple[CsrQueueCandidateCoverageRow, ...]
    gap_rows: tuple[CsrQueueCandidateGapRow, ...]
    summary: CsrQueueCandidateCoverageSummary
    schema: dict[str, Any]


def build_csr_queue_candidate_coverage_from_files(
    *,
    queue_batches_root: str | Path = "runs",
    full_queue_jobs_path: str | Path = (
        "runs/phase1_csr_full_dataset_queue/csr_full_dataset_queue_jobs.jsonl"
    ),
    queue_training_pool_summary_path: str | Path = (
        "runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json"
    ),
    recent_window: int = 2,
    low_success_threshold: float = 0.5,
) -> CsrQueueCandidateCoverage:
    batch_dirs = _completed_batch_dirs(Path(queue_batches_root))
    batch_summaries = tuple(
        json.loads(
            (path / "csr_queue_batch_execution_summary.json").read_text(
                encoding="utf-8"
            )
        )
        for path in batch_dirs
    )
    result_rows = []
    for path in batch_dirs:
        batch_id = str(
            json.loads(
                (path / "csr_queue_batch_execution_summary.json").read_text(
                    encoding="utf-8"
                )
            )["batch_id"]
        )
        for row in read_jsonl(path / "csr_micro_campaign_results.jsonl"):
            item = dict(row)
            item["source_batch_id"] = batch_id
            result_rows.append(item)
    full_queue_jobs = tuple(read_jsonl(full_queue_jobs_path))
    pool_summary = json.loads(
        Path(queue_training_pool_summary_path).read_text(encoding="utf-8")
    )
    return build_csr_queue_candidate_coverage(
        batch_summaries,
        result_rows,
        full_queue_jobs,
        pool_summary=pool_summary,
        recent_window=recent_window,
        low_success_threshold=low_success_threshold,
    )


def build_csr_queue_candidate_coverage(
    batch_summaries: Iterable[dict[str, Any]],
    result_rows: Iterable[dict[str, Any]],
    full_queue_jobs: Iterable[dict[str, Any]],
    *,
    pool_summary: dict[str, Any],
    recent_window: int = 2,
    low_success_threshold: float = 0.5,
) -> CsrQueueCandidateCoverage:
    summaries = tuple(sorted(batch_summaries, key=lambda row: int(row["batch_rank"])))
    results = tuple(result_rows)
    jobs = tuple(full_queue_jobs)
    if not summaries:
        raise ValueError("no completed queue batch summaries")
    if recent_window < 1:
        raise ValueError("recent_window must be >= 1")

    batch_rows = tuple(
        _batch_row(summary, low_success_threshold=low_success_threshold)
        for summary in summaries
    )
    current_queue_keys = tuple(sorted({_candidate_key(row) for row in jobs}))
    candidate_rows = _candidate_rows(results, current_queue_keys=current_queue_keys)
    gap_rows = _gap_rows(
        current_queue_keys=current_queue_keys,
        batch_rows=batch_rows,
        candidate_rows=candidate_rows,
    )
    summary = _summary(
        batch_rows=batch_rows,
        candidate_rows=candidate_rows,
        gap_rows=gap_rows,
        jobs=jobs,
        pool_summary=pool_summary,
        recent_window=recent_window,
    )
    return CsrQueueCandidateCoverage(
        batch_rows=batch_rows,
        candidate_rows=candidate_rows,
        gap_rows=gap_rows,
        summary=summary,
        schema=_schema(),
    )


def write_csr_queue_candidate_coverage_artifacts(
    coverage: CsrQueueCandidateCoverage,
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "batch_rows": output / "csr_queue_candidate_coverage_batch_rows.jsonl",
        "candidate_rows": output / "csr_queue_candidate_coverage_candidate_rows.jsonl",
        "gap_rows": output / "csr_queue_candidate_coverage_gap_rows.jsonl",
        "summary": output / "csr_queue_candidate_coverage_summary.json",
        "schema": output / "csr_queue_candidate_coverage_schema.json",
        "report": output / "csr_queue_candidate_coverage_report.md",
    }
    write_jsonl((asdict(row) for row in coverage.batch_rows), paths["batch_rows"])
    write_jsonl((asdict(row) for row in coverage.candidate_rows), paths["candidate_rows"])
    write_jsonl((asdict(row) for row in coverage.gap_rows), paths["gap_rows"])
    paths["summary"].write_text(
        json.dumps(asdict(coverage.summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["schema"].write_text(
        json.dumps(coverage.schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(coverage, paths["report"])
    return paths


def _completed_batch_dirs(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in sorted(root.glob("phase1_csr_queue_batch_*"), key=lambda item: item.name):
        summary_path = path / "csr_queue_batch_execution_summary.json"
        results_path = path / "csr_micro_campaign_results.jsonl"
        if not summary_path.exists() or not results_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("status") == "passed":
            paths.append(path)
    return tuple(paths)


def _batch_row(
    summary: dict[str, Any],
    *,
    low_success_threshold: float,
) -> CsrQueueBatchCoverageRow:
    candidate_jobs = int(summary["candidate_jobs"])
    success_rows = int(summary["gpu_success_rows"])
    screen_rows = int(summary["cpu_screened_out_rows"])
    success_rate = success_rows / candidate_jobs if candidate_jobs else 0.0
    screen_rate = screen_rows / candidate_jobs if candidate_jobs else 0.0
    return CsrQueueBatchCoverageRow(
        schema_version=CSR_QUEUE_CANDIDATE_COVERAGE_SCHEMA_VERSION,
        batch_id=str(summary["batch_id"]),
        batch_rank=int(summary["batch_rank"]),
        batch_outcome=str(summary["batch_outcome"]),
        planned_total_nnz=int(summary["planned_total_nnz"]),
        planned_max_matrix_nnz=int(summary["planned_max_matrix_nnz"]),
        candidate_jobs=candidate_jobs,
        gpu_success_rows=success_rows,
        cpu_screened_out_rows=screen_rows,
        gpu_failed_rows=int(summary["gpu_failed_rows"]),
        selector_oracle_rows=int(summary["selector_oracle_rows"]),
        success_rate=success_rate,
        screen_rate=screen_rate,
        low_success=success_rate < low_success_threshold,
        screen_only_no_oracle=bool(summary["completed_without_oracle"]),
    )


def _candidate_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    current_queue_keys: tuple[str, ...],
) -> tuple[CsrQueueCandidateCoverageRow, ...]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[_candidate_key(row)].append(row)
    output: list[CsrQueueCandidateCoverageRow] = []
    current_key_set = set(current_queue_keys)
    for key, items in sorted(by_key.items()):
        first = items[0]
        statuses = Counter(str(item["status"]) for item in items)
        source_batches = tuple(
            sorted({str(item.get("source_batch_id", "")) for item in items if item.get("source_batch_id")})
        )
        attempted = len(items)
        output.append(
            CsrQueueCandidateCoverageRow(
                schema_version=CSR_QUEUE_CANDIDATE_COVERAGE_SCHEMA_VERSION,
                candidate_key=key,
                candidate_id=str(first["candidate_id"]),
                solver=str(first["solver"]),
                preconditioner=str(first["preconditioner"]),
                solver_parameters_key=_solver_parameters_key(first),
                attempted_rows=attempted,
                success_rows=int(statuses.get("success", 0)),
                screened_out_rows=int(statuses.get("screened_out", 0)),
                gpu_failed_rows=int(statuses.get("failed", 0)),
                success_rate=(float(statuses.get("success", 0)) / attempted if attempted else 0.0),
                screen_rate=(
                    float(statuses.get("screened_out", 0)) / attempted if attempted else 0.0
                ),
                source_batch_count=len(source_batches),
                source_batches=source_batches,
                current_queue_candidate=key in current_key_set,
            )
        )
    return tuple(output)


def _gap_rows(
    *,
    current_queue_keys: tuple[str, ...],
    batch_rows: tuple[CsrQueueBatchCoverageRow, ...],
    candidate_rows: tuple[CsrQueueCandidateCoverageRow, ...],
) -> tuple[CsrQueueCandidateGapRow, ...]:
    current = set(current_queue_keys)
    recent_low = bool(batch_rows and batch_rows[-1].screen_only_no_oracle)
    low_general_success = _success_rate_for_solver_group(
        candidate_rows,
        solvers={"bicgstab", "gmres"},
    ) < 0.25
    gaps = [
        _gap(
            gap_id="general_gmres_jacobi_restart32",
            priority="high" if recent_low else "medium",
            candidate_profile="general",
            proposed_candidate_id="taichi_csr_gmres_jacobi_restart32_float64",
            solver="gmres",
            preconditioner="jacobi",
            solver_parameters={"restart": 32},
            reason="gmres_restart_parameter_coverage_gap",
            queue_ready=True,
            blocker=None,
            next_action="add_to_candidate_coverage_probe_queue",
            evidence="Current queue only includes GMRES restart=16.",
        ),
        _gap(
            gap_id="general_gmres_jacobi_restart64",
            priority="medium",
            candidate_profile="general",
            proposed_candidate_id="taichi_csr_gmres_jacobi_restart64_float64",
            solver="gmres",
            preconditioner="jacobi",
            solver_parameters={"restart": 64},
            reason="gmres_restart_parameter_coverage_gap",
            queue_ready=True,
            blocker=None,
            next_action="add_to_candidate_coverage_probe_queue_after_restart32",
            evidence="CPU screen already accepts GMRES restart parameters.",
        ),
        _gap(
            gap_id="general_bicgstab_ilu0",
            priority="high" if low_general_success else "medium",
            candidate_profile="general",
            proposed_candidate_id="taichi_csr_bicgstab_ilu0_float64",
            solver="bicgstab",
            preconditioner="ilu0",
            solver_parameters={},
            reason="implemented_preconditioner_not_in_queue",
            queue_ready=False,
            blocker="generic_queue_cpu_screen_for_ilu0_not_integrated",
            next_action="probe_with_existing_ilu0_coverage_path_before_queue_merge",
            evidence="Existing ILU0 coverage artifacts show merge-ready successes.",
        ),
        _gap(
            gap_id="general_bicgstab_row_column_equilibration",
            priority="medium",
            candidate_profile="general",
            proposed_candidate_id="taichi_csr_bicgstab_row_column_equilibration_float64",
            solver="bicgstab",
            preconditioner="row_column_equilibration",
            solver_parameters={},
            reason="implemented_preconditioner_not_in_queue",
            queue_ready=False,
            blocker="generic_queue_cpu_screen_for_row_column_equilibration_not_integrated",
            next_action="add_scaled_cpu_screen_before_queue_merge",
            evidence="Row/column equilibration has dedicated Taichi CSR smoke evidence.",
        ),
        _gap(
            gap_id="symmetric_chebyshev_jacobi",
            priority="medium",
            candidate_profile="symmetric",
            proposed_candidate_id="taichi_csr_chebyshev_jacobi_float64",
            solver="chebyshev",
            preconditioner="jacobi",
            solver_parameters={},
            reason="implemented_solver_not_in_queue",
            queue_ready=False,
            blocker="generic_queue_cpu_screen_for_chebyshev_not_integrated",
            next_action="add_chebyshev_cpu_screen_before_queue_merge",
            evidence="Chebyshev has dedicated Taichi CSR smoke evidence.",
        ),
        _gap(
            gap_id="symmetric_pcg_symmetric_equilibration",
            priority="medium",
            candidate_profile="symmetric",
            proposed_candidate_id="taichi_csr_pcg_symmetric_equilibration_float64",
            solver="pcg",
            preconditioner="symmetric_equilibration",
            solver_parameters={},
            reason="implemented_preconditioner_not_in_queue",
            queue_ready=False,
            blocker="generic_queue_cpu_screen_for_symmetric_equilibration_not_integrated",
            next_action="add_scaled_cpu_screen_before_queue_merge",
            evidence="Symmetric equilibration has dedicated Taichi CSR smoke evidence.",
        ),
    ]
    return tuple(row for row in gaps if _candidate_key(asdict(row), proposed=True) not in current)


def _gap(**kwargs: Any) -> CsrQueueCandidateGapRow:
    return CsrQueueCandidateGapRow(
        schema_version=CSR_QUEUE_CANDIDATE_COVERAGE_SCHEMA_VERSION,
        **kwargs,
    )


def _summary(
    *,
    batch_rows: tuple[CsrQueueBatchCoverageRow, ...],
    candidate_rows: tuple[CsrQueueCandidateCoverageRow, ...],
    gap_rows: tuple[CsrQueueCandidateGapRow, ...],
    jobs: tuple[dict[str, Any], ...],
    pool_summary: dict[str, Any],
    recent_window: int,
) -> CsrQueueCandidateCoverageSummary:
    recent = batch_rows[-recent_window:]
    recent_jobs = sum(row.candidate_jobs for row in recent)
    recent_success = sum(row.gpu_success_rows for row in recent)
    recent_screen = sum(row.cpu_screened_out_rows for row in recent)
    recent_oracle = sum(row.selector_oracle_rows for row in recent)
    trigger = (
        len(recent) >= 2
        and recent[-1].screen_only_no_oracle
        and (recent[-2].low_success or recent[-2].screen_only_no_oracle)
    )
    current_solvers = tuple(sorted({str(row["solver"]) for row in jobs}))
    current_preconditioners = tuple(sorted({str(row["preconditioner"]) for row in jobs}))
    current_keys = tuple(sorted({_candidate_key(row) for row in jobs}))
    backend = default_backend_registry().get("taichi_gpu")
    implemented_solvers = tuple(default_solver_registry().names())
    implemented_preconditioners = tuple(default_preconditioner_registry().names())
    missing_solvers = tuple(
        solver
        for solver in implemented_solvers
        if solver in backend.supported_solvers and solver not in current_solvers
    )
    missing_preconditioners = tuple(
        preconditioner
        for preconditioner in implemented_preconditioners
        if preconditioner in backend.supported_preconditioners
        and preconditioner not in current_preconditioners
    )
    return CsrQueueCandidateCoverageSummary(
        status="passed" if batch_rows and candidate_rows and gap_rows else "failed",
        schema_version=CSR_QUEUE_CANDIDATE_COVERAGE_SCHEMA_VERSION,
        runtime_selector_changed=False,
        executes_gpu=False,
        imports_matrices=False,
        source_batch_count=len(batch_rows),
        completed_queue_batches=int(pool_summary["completed_queue_batches"]),
        next_pending_batch_id=pool_summary.get("next_pending_batch_id"),
        latest_batch_id=batch_rows[-1].batch_id,
        latest_batch_outcome=batch_rows[-1].batch_outcome,
        recent_window_batches=tuple(row.batch_id for row in recent),
        recent_candidate_jobs=recent_jobs,
        recent_gpu_success_rows=recent_success,
        recent_cpu_screened_out_rows=recent_screen,
        recent_selector_oracle_rows=recent_oracle,
        recent_success_rate=recent_success / recent_jobs if recent_jobs else 0.0,
        recent_screen_only_batches=sum(1 for row in recent if row.screen_only_no_oracle),
        trigger_candidate_coverage=trigger,
        trigger_reason=(
            "screen_only_after_low_success_batch" if trigger else None
        ),
        current_queue_solvers=current_solvers,
        current_queue_preconditioners=current_preconditioners,
        current_queue_candidate_keys=current_keys,
        implemented_backend_solvers=implemented_solvers,
        implemented_backend_preconditioners=implemented_preconditioners,
        queue_missing_supported_solvers=missing_solvers,
        queue_missing_supported_preconditioners=missing_preconditioners,
        candidate_coverage_rows=len(candidate_rows),
        planned_gap_count=len(gap_rows),
        queue_ready_gap_count=sum(1 for row in gap_rows if row.queue_ready),
        cpu_screen_blocked_gap_count=sum(1 for row in gap_rows if not row.queue_ready),
        next_step=(
            "run_queue_ready_gmres_restart_probe_then_integrate_cpu_screens_for_ilu0_equilibration_chebyshev"
            if trigger
            else "continue_queue_ingestion_or_run_low_cost_queue_ready_gap_probe"
        ),
    )


def _candidate_key(row: dict[str, Any], *, proposed: bool = False) -> str:
    solver = str(row["solver"])
    preconditioner = str(row["preconditioner"])
    params = row.get("solver_parameters", {})
    if not isinstance(params, dict):
        params = {}
    params = dict(params)
    for key in ("restart", "omega", "omega_source"):
        if key in row:
            params[key] = row[key]
    if proposed:
        params = dict(params)
    elif solver == "richardson":
        params = {"omega_source": "cpu_screen_required"}
    return f"{solver}:{preconditioner}:{json.dumps(params, sort_keys=True)}"


def _solver_parameters_key(row: dict[str, Any]) -> str:
    params = row.get("solver_parameters", {})
    if not isinstance(params, dict):
        params = {}
    keys = ("restart", "omega", "omega_source")
    normalized = {key: row[key] for key in keys if key in row}
    normalized.update(params)
    if str(row.get("solver")) == "richardson" and "omega" in normalized:
        normalized["omega_source"] = "cpu_screen_selected"
        normalized.pop("omega", None)
    return json.dumps(normalized, sort_keys=True)


def _success_rate_for_solver_group(
    rows: tuple[CsrQueueCandidateCoverageRow, ...],
    *,
    solvers: set[str],
) -> float:
    attempted = sum(row.attempted_rows for row in rows if row.solver in solvers)
    success = sum(row.success_rows for row in rows if row.solver in solvers)
    return success / attempted if attempted else 0.0


def _schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_QUEUE_CANDIDATE_COVERAGE_SCHEMA_VERSION,
        "task": "diagnose_solver_candidate_coverage_for_completed_csr_queue_batches",
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "imports_matrices": False,
        "integration_boundary": {
            "status": "diagnostic_plan_only",
            "does_not_modify_full_dataset_queue": True,
            "does_not_promote_learned_runtime_selector": True,
        },
        "gap_policy": {
            "queue_ready": "can be probed with existing generic CPU screen path",
            "cpu_screen_blocked": "requires CPU-screen integration or dedicated probe before queue merge",
        },
    }


def _write_report(coverage: CsrQueueCandidateCoverage, path: Path) -> Path:
    summary = coverage.summary
    lines = [
        "# CSR Queue Candidate Coverage",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        f"- executes_gpu: `{summary.executes_gpu}`",
        f"- completed_queue_batches: `{summary.completed_queue_batches}`",
        f"- next_pending_batch_id: `{summary.next_pending_batch_id}`",
        f"- latest_batch_id: `{summary.latest_batch_id}`",
        f"- latest_batch_outcome: `{summary.latest_batch_outcome}`",
        f"- recent_window_batches: `{list(summary.recent_window_batches)}`",
        f"- recent_success_rate: `{summary.recent_success_rate}`",
        f"- trigger_candidate_coverage: `{summary.trigger_candidate_coverage}`",
        f"- trigger_reason: `{summary.trigger_reason}`",
        f"- current_queue_solvers: `{list(summary.current_queue_solvers)}`",
        f"- current_queue_preconditioners: `{list(summary.current_queue_preconditioners)}`",
        f"- queue_missing_supported_solvers: `{list(summary.queue_missing_supported_solvers)}`",
        f"- queue_missing_supported_preconditioners: `{list(summary.queue_missing_supported_preconditioners)}`",
        f"- planned_gap_count: `{summary.planned_gap_count}`",
        f"- queue_ready_gap_count: `{summary.queue_ready_gap_count}`",
        f"- cpu_screen_blocked_gap_count: `{summary.cpu_screen_blocked_gap_count}`",
        f"- next_step: `{summary.next_step}`",
        "",
        "## Recent Batches",
        "",
        "| batch | outcome | jobs | success | screened | oracle | success_rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in coverage.batch_rows[-8:]:
        lines.append(
            "| "
            f"{row.batch_id} | "
            f"{row.batch_outcome} | "
            f"{row.candidate_jobs} | "
            f"{row.gpu_success_rows} | "
            f"{row.cpu_screened_out_rows} | "
            f"{row.selector_oracle_rows} | "
            f"{row.success_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Gaps",
            "",
            "| priority | gap | candidate | queue_ready | blocker | next_action |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in coverage.gap_rows:
        lines.append(
            "| "
            f"{row.priority} | "
            f"{row.gap_id} | "
            f"{row.proposed_candidate_id} | "
            f"{row.queue_ready} | "
            f"{row.blocker or ''} | "
            f"{row.next_action} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
