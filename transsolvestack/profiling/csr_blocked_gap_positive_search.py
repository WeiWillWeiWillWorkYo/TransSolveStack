"""Find positive GPU evidence for previously blocked CSR candidate gaps."""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.datasets.csr import csr_matrix_from_record
from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
from transsolvestack.policies.csr_selector_data import (
    build_csr_selector_export,
    write_csr_diagnostic_rows,
    write_csr_selector_report,
    write_csr_selector_rows,
    write_csr_selector_schema,
    write_csr_selector_summary,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.csr_blocked_gap_runtime import (
    _gpu_failed_row,
    _run_gpu_once,
)
from transsolvestack.profiling.csr_blocked_gap_screens import (
    screen_blocked_gap_candidate,
)
from transsolvestack.runtime.engine import TaichiExecutionEngine


CSR_BLOCKED_GAP_POSITIVE_SEARCH_SCHEMA_VERSION = (
    "phase1_csr_blocked_gap_positive_search_v1"
)

DEFAULT_CSR_RECORD_PATHS = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "runs/phase1_csr_queue_batch_00001/csr_matrices.jsonl",
    "runs/phase1_csr_queue_batch_00003/csr_matrices.jsonl",
    "runs/phase1_csr_queue_batch_00004/csr_matrices.jsonl",
    "runs/phase1_csr_queue_batch_00005/csr_matrices.jsonl",
    "runs/phase1_csr_queue_batch_00006/csr_matrices.jsonl",
    "runs/phase1_csr_queue_batch_00007/csr_matrices.jsonl",
    "runs/phase1_csr_queue_batch_00008/csr_matrices.jsonl",
    "runs/phase1_csr_queue_batch_00009/csr_matrices.jsonl",
    "runs/phase1_csr_queue_batch_00010/csr_matrices.jsonl",
)


def run_csr_blocked_gap_positive_search_from_files(
    *,
    csr_record_paths: Iterable[str | Path] = DEFAULT_CSR_RECORD_PATHS,
    output_dir: str | Path = "runs/phase1_csr_blocked_gap_positive_search",
    max_rows: int = 2_048,
    max_cols: int = 2_048,
    max_nnz: int = 20_000,
    max_iter: int = 300,
    tolerance_rel: float = 1.0e-5,
    max_positive_per_gap: int = 2,
    device_memory_gb: float = 0.20,
) -> dict[str, Any]:
    paths = tuple(str(path) for path in csr_record_paths)
    records = _load_unique_records(paths)
    selected, search_rows = _select_positive_candidates(
        records,
        max_rows=max_rows,
        max_cols=max_cols,
        max_nnz=max_nnz,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
        max_positive_per_gap=max_positive_per_gap,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ensure_taichi_cuda(device_memory_gb=device_memory_gb)
    engine = TaichiExecutionEngine()
    result_rows = tuple(
        _evaluate_selected_candidate(item, engine=engine)
        for item in selected
    )
    selected_records = tuple(item["record"] for item in selected)
    selector_export = build_csr_selector_export(
        selected_records,
        result_rows,
        context_id="phase1_csr_blocked_gap_positive_search",
    )
    summary = _summary(
        csr_record_paths=paths,
        records=records,
        selected=selected,
        search_rows=search_rows,
        result_rows=result_rows,
        selector_oracle_rows=int(selector_export.summary.num_oracle_rows),
        max_rows=max_rows,
        max_cols=max_cols,
        max_nnz=max_nnz,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
        max_positive_per_gap=max_positive_per_gap,
        device_memory_gb=device_memory_gb,
    )
    schema = _schema()
    paths_out = _write_artifacts(
        output,
        selected=selected,
        search_rows=search_rows,
        result_rows=result_rows,
        selector_export=selector_export,
        summary=summary,
        schema=schema,
    )
    return {
        "paths": {key: str(value) for key, value in paths_out.items()},
        "summary": summary,
        "schema": schema,
        "selected": selected,
        "search_rows": search_rows,
        "result_rows": result_rows,
    }


def _load_unique_records(paths: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for path in paths:
        source = Path(path)
        if not source.exists():
            continue
        for row in read_jsonl(source):
            if row.get("status") != "success":
                continue
            matrix_id = str(row["matrix_id"])
            if matrix_id in records:
                continue
            record = dict(row)
            record["positive_search_source_path"] = str(source)
            records[matrix_id] = record
    return tuple(records.values())


def _select_positive_candidates(
    records: tuple[dict[str, Any], ...],
    *,
    max_rows: int,
    max_cols: int,
    max_nnz: int,
    max_iter: int,
    tolerance_rel: float,
    max_positive_per_gap: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    selected: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    positive_by_gap: Counter[str] = Counter()
    gap_ids = tuple(row["coverage_gap_id"] for row in _candidate_templates())
    for record in records:
        if all(positive_by_gap[gap] >= max_positive_per_gap for gap in gap_ids):
            break
        matrix_id = str(record["matrix_id"])
        if (
            int(record["n_rows"]) > max_rows
            or int(record["n_cols"]) > max_cols
            or int(record["csr_nnz"]) > max_nnz
        ):
            search_rows.append(
                _search_row(
                    record,
                    coverage_gap_id="all",
                    status="skipped",
                    reason="above_positive_search_size_limit",
                )
            )
            continue
        csr = csr_matrix_from_record(record)
        for template in _candidate_templates():
            gap_id = str(template["coverage_gap_id"])
            if positive_by_gap[gap_id] >= max_positive_per_gap:
                continue
            if template["solver"] in {"pcg", "chebyshev"} and (
                csr.symmetry != "symmetric" or csr.n_rows != csr.n_cols
            ):
                search_rows.append(
                    _search_row(
                        record,
                        coverage_gap_id=gap_id,
                        status="skipped",
                        reason="requires_symmetric_square",
                    )
                )
                continue
            candidate = _candidate_from_template(
                template,
                matrix_id=matrix_id,
                max_iter=max_iter,
                tolerance_rel=tolerance_rel,
            )
            screen = screen_blocked_gap_candidate(csr, candidate)
            if screen is None:
                search_rows.append(
                    _search_row(
                        record,
                        coverage_gap_id=gap_id,
                        status="screened_out",
                        reason="unsupported_blocked_gap_screen",
                    )
                )
                continue
            if not screen["success"]:
                search_rows.append(
                    _search_row(
                        record,
                        coverage_gap_id=gap_id,
                        status="screened_out",
                        reason=str(screen.get("breakdown") or "screen_failed"),
                        screen=screen,
                    )
                )
                continue
            positive_by_gap[gap_id] += 1
            selected_item = {
                "schema_version": CSR_BLOCKED_GAP_POSITIVE_SEARCH_SCHEMA_VERSION,
                "record": record,
                "candidate": candidate,
                "cpu_screen": screen,
                "positive_rank_in_gap": positive_by_gap[gap_id],
            }
            selected.append(selected_item)
            search_rows.append(
                _search_row(
                    record,
                    coverage_gap_id=gap_id,
                    status="selected_positive",
                    reason="cpu_screen_passed",
                    screen=screen,
                    candidate=candidate,
                )
            )
    return tuple(selected), tuple(search_rows)


def _evaluate_selected_candidate(
    item: dict[str, Any],
    *,
    engine: TaichiExecutionEngine,
) -> dict[str, Any]:
    csr = csr_matrix_from_record(item["record"])
    candidate = item["candidate"]
    screen = item["cpu_screen"]
    try:
        row = _run_gpu_once(
            csr,
            candidate,
            screen=screen,
            engine=engine,
            context_id="phase1_csr_blocked_gap_positive_search",
        )
    except Exception as exc:
        row = _gpu_failed_row(csr, candidate, screen=screen, exc=exc)
    row["positive_rank_in_gap"] = int(item["positive_rank_in_gap"])
    row["positive_search_source_path"] = str(item["record"]["positive_search_source_path"])
    return row


def _candidate_templates() -> tuple[dict[str, Any], ...]:
    return (
        {
            "coverage_gap_id": "general_bicgstab_ilu0",
            "candidate_id": "taichi_csr_bicgstab_ilu0_float64",
            "solver": "bicgstab",
            "preconditioner": "ilu0",
            "solver_parameters": {},
            "preconditioner_parameters": {"pivot_tolerance": 1.0e-12},
        },
        {
            "coverage_gap_id": "general_bicgstab_row_column_equilibration",
            "candidate_id": "taichi_csr_bicgstab_row_column_equilibration_float64",
            "solver": "bicgstab",
            "preconditioner": "row_column_equilibration",
            "solver_parameters": {},
            "preconditioner_parameters": {"passes": 4},
        },
        {
            "coverage_gap_id": "symmetric_chebyshev_jacobi",
            "candidate_id": "taichi_csr_chebyshev_jacobi_float64",
            "solver": "chebyshev",
            "preconditioner": "jacobi",
            "solver_parameters": {},
            "preconditioner_parameters": {},
        },
        {
            "coverage_gap_id": "symmetric_pcg_symmetric_equilibration",
            "candidate_id": "taichi_csr_pcg_symmetric_equilibration_float64",
            "solver": "pcg",
            "preconditioner": "symmetric_equilibration",
            "solver_parameters": {},
            "preconditioner_parameters": {},
        },
    )


def _candidate_from_template(
    template: dict[str, Any],
    *,
    matrix_id: str,
    max_iter: int,
    tolerance_rel: float,
) -> dict[str, Any]:
    return {
        "schema_version": CSR_BLOCKED_GAP_POSITIVE_SEARCH_SCHEMA_VERSION,
        "queue_id": "phase1_csr_blocked_gap_positive_search",
        "job_id": f"blocked_gap_positive:{matrix_id}:{template['coverage_gap_id']}",
        "batch_id": "blocked_gap_positive_search",
        "matrix_id": matrix_id,
        "candidate_id": str(template["candidate_id"]),
        "coverage_gap_id": str(template["coverage_gap_id"]),
        "solver": str(template["solver"]),
        "preconditioner": str(template["preconditioner"]),
        "precision": "float64",
        "solver_parameters": dict(template["solver_parameters"]),
        "preconditioner_parameters": dict(template["preconditioner_parameters"]),
        "measurement_repeats": 1,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "requires_cpu_screen": True,
        "planned_gpu_solve_attempts": 1,
    }


def _search_row(
    record: dict[str, Any],
    *,
    coverage_gap_id: str,
    status: str,
    reason: str,
    screen: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "schema_version": CSR_BLOCKED_GAP_POSITIVE_SEARCH_SCHEMA_VERSION,
        "matrix_id": str(record["matrix_id"]),
        "coverage_gap_id": coverage_gap_id,
        "status": status,
        "reason": reason,
        "n_rows": int(record["n_rows"]),
        "n_cols": int(record["n_cols"]),
        "csr_nnz": int(record["csr_nnz"]),
        "field": str(record["field"]),
        "symmetry": str(record["symmetry"]),
        "source_path": str(record["positive_search_source_path"]),
    }
    if candidate is not None:
        row["candidate_id"] = str(candidate["candidate_id"])
        row["solver"] = str(candidate["solver"])
        row["preconditioner"] = str(candidate["preconditioner"])
    if screen is not None:
        row["cpu_screen"] = screen
        row["screen_relative_residual"] = screen.get("relative_residual")
        row["screen_solution_relative_error"] = screen.get("solution_relative_error")
        row["screen_iterations"] = screen.get("iterations")
    return row


def _summary(
    *,
    csr_record_paths: tuple[str, ...],
    records: tuple[dict[str, Any], ...],
    selected: tuple[dict[str, Any], ...],
    search_rows: tuple[dict[str, Any], ...],
    result_rows: tuple[dict[str, Any], ...],
    selector_oracle_rows: int,
    max_rows: int,
    max_cols: int,
    max_nnz: int,
    max_iter: int,
    tolerance_rel: float,
    max_positive_per_gap: int,
    device_memory_gb: float,
) -> dict[str, Any]:
    by_status = _counts(str(row["status"]) for row in result_rows)
    by_gap_status = _by_gap_status(result_rows)
    success_gap_ids = tuple(
        sorted(
            gap_id
            for gap_id, counts in by_gap_status.items()
            if int(counts.get("success", 0)) > 0
        )
    )
    required_gap_ids = tuple(row["coverage_gap_id"] for row in _candidate_templates())
    gpu_success_rows = int(by_status.get("success", 0))
    gpu_failed_rows = int(by_status.get("failed", 0))
    status = (
        "passed"
        if len(selected) >= len(required_gap_ids)
        and set(success_gap_ids) == set(required_gap_ids)
        and gpu_success_rows == len(result_rows)
        and gpu_failed_rows == 0
        and selector_oracle_rows > 0
        else "failed"
    )
    return {
        "status": status,
        "schema_version": CSR_BLOCKED_GAP_POSITIVE_SEARCH_SCHEMA_VERSION,
        "runtime_selector_changed": False,
        "executes_gpu": True,
        "queue_merge_ready": False,
        "positive_evidence_found": status == "passed",
        "csr_record_paths": csr_record_paths,
        "source_records": len(records),
        "search_rows": len(search_rows),
        "selected_candidates": len(selected),
        "candidate_jobs": len(result_rows),
        "required_gap_ids": required_gap_ids,
        "positive_gpu_gap_ids": success_gap_ids,
        "max_positive_per_gap": max_positive_per_gap,
        "max_rows": max_rows,
        "max_cols": max_cols,
        "max_nnz": max_nnz,
        "max_iter": max_iter,
        "tolerance_rel": tolerance_rel,
        "device_memory_gb": device_memory_gb,
        "gpu_success_rows": gpu_success_rows,
        "gpu_failed_rows": gpu_failed_rows,
        "selector_rows": len(result_rows),
        "selector_oracle_rows": selector_oracle_rows,
        "by_status": by_status,
        "by_gap_status": by_gap_status,
        "by_search_status": _counts(str(row["status"]) for row in search_rows),
        "selected_matrix_ids": tuple(sorted({row["matrix_id"] for row in result_rows})),
        "next_step": (
            "integrate positive blocked-gap rows into guarded selector evidence "
            "without generic queue merge"
        ),
    }


def _schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_BLOCKED_GAP_POSITIVE_SEARCH_SCHEMA_VERSION,
        "task": "find_positive_gpu_evidence_for_blocked_csr_candidate_gaps",
        "execution_boundary": {
            "executes_gpu": True,
            "cpu_screen_before_gpu": True,
            "runtime_selector_changed": False,
            "generic_queue_merge": False,
        },
        "positive_gate": [
            "CPU admission screen must pass",
            "Taichi GPU solve must pass numeric gates",
            "all four blocked gap families must have at least one success",
        ],
    }


def _write_artifacts(
    output: Path,
    *,
    selected: tuple[dict[str, Any], ...],
    search_rows: tuple[dict[str, Any], ...],
    result_rows: tuple[dict[str, Any], ...],
    selector_export,
    summary: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "selected_candidates": output / "csr_blocked_gap_positive_candidates.jsonl",
        "search_rows": output / "csr_blocked_gap_positive_search_rows.jsonl",
        "results": output / "csr_blocked_gap_positive_results.jsonl",
        "summary": output / "csr_blocked_gap_positive_search_summary.json",
        "schema": output / "csr_blocked_gap_positive_search_schema.json",
        "report": output / "csr_blocked_gap_positive_search_report.md",
        "diagnostics": output / "csr_blocked_gap_positive_diagnostic_rows.jsonl",
        "selector_rows": output / "csr_blocked_gap_positive_selector_rows.jsonl",
        "selector_summary": output / "csr_blocked_gap_positive_selector_summary.json",
        "selector_schema": output / "csr_blocked_gap_positive_selector_schema.json",
        "selector_report": output / "csr_blocked_gap_positive_selector_report.md",
    }
    write_jsonl((_selected_row(item) for item in selected), paths["selected_candidates"])
    write_jsonl(search_rows, paths["search_rows"])
    write_jsonl(result_rows, paths["results"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["schema"].write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(result_rows, summary, paths["report"])
    write_csr_diagnostic_rows(selector_export.diagnostics, paths["diagnostics"])
    write_csr_selector_rows(selector_export.selector_rows, paths["selector_rows"])
    write_csr_selector_summary(selector_export.summary, paths["selector_summary"])
    write_csr_selector_schema(paths["selector_schema"])
    write_csr_selector_report(selector_export, paths["selector_report"])
    return paths


def _selected_row(item: dict[str, Any]) -> dict[str, Any]:
    record = item["record"]
    candidate = item["candidate"]
    return {
        "schema_version": CSR_BLOCKED_GAP_POSITIVE_SEARCH_SCHEMA_VERSION,
        "matrix_id": str(record["matrix_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "coverage_gap_id": str(candidate["coverage_gap_id"]),
        "solver": str(candidate["solver"]),
        "preconditioner": str(candidate["preconditioner"]),
        "positive_rank_in_gap": int(item["positive_rank_in_gap"]),
        "n_rows": int(record["n_rows"]),
        "n_cols": int(record["n_cols"]),
        "csr_nnz": int(record["csr_nnz"]),
        "field": str(record["field"]),
        "symmetry": str(record["symmetry"]),
        "source_path": str(record["positive_search_source_path"]),
        "cpu_screen": item["cpu_screen"],
    }


def _write_report(
    result_rows: tuple[dict[str, Any], ...],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# CSR Blocked Gap Positive Search",
        "",
        f"- status: `{summary['status']}`",
        f"- positive_evidence_found: `{summary['positive_evidence_found']}`",
        f"- selected_candidates: `{summary['selected_candidates']}`",
        f"- gpu_success_rows: `{summary['gpu_success_rows']}`",
        f"- gpu_failed_rows: `{summary['gpu_failed_rows']}`",
        f"- positive_gpu_gap_ids: `{list(summary['positive_gpu_gap_ids'])}`",
        f"- queue_merge_ready: `{summary['queue_merge_ready']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
        "| gap | matrix | candidate | status | iters | rel_res | sol_err |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            "| "
            f"{row['coverage_gap_id']} | "
            f"{row['matrix_id']} | "
            f"{row['candidate_id']} | "
            f"{row['status']} | "
            f"{row.get('num_iterations')} | "
            f"{_fmt(row.get('final_relative_residual'))} | "
            f"{_fmt(row.get('solution_relative_error'))} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _by_gap_status(rows: tuple[dict[str, Any], ...]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        counts.setdefault(str(row["coverage_gap_id"]), Counter())[str(row["status"])] += 1
    return {gap_id: dict(counter) for gap_id, counter in sorted(counts.items())}


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not number == number:
        return "nan"
    if number in {float("inf"), float("-inf")}:
        return str(number)
    return f"{number:.6g}"
