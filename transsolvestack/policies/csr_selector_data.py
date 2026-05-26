"""CSR selector feature and benchmark-row export."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.datasets.csr import csr_matrix_from_record
from transsolvestack.datasets.diagnostics import diagnose_csr_matrix
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_SELECTOR_SCHEMA_VERSION = "phase1_csr_selector_features_v8"


STRUCTURAL_SKIP_STATUS_BY_REASON = {
    "not_symmetric": "not_applicable",
    "after_selection_limit": "not_profiled",
    "above_smoke_size_limit": "not_profiled",
}


@dataclass(frozen=True)
class CsrDiagnosticFeatureRow:
    schema_version: str
    matrix_id: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CsrSelectorRow:
    schema_version: str
    matrix_id: str
    context_id: str
    candidate_id: str
    solver: str
    preconditioner: str
    precision: str
    solver_parameters: dict[str, Any]
    label_is_oracle: bool
    target_status: str
    target_success_rate: float
    target_measurement_repeats: int
    target_solve_time_ms: float | None
    target_median_solve_time_ms: float | None
    target_solve_time_iqr_ms: float | None
    target_wall_time_ms: float | None
    target_regret_vs_oracle_ms: float | None
    target_num_iterations: int | None
    target_final_relative_residual: float | None
    target_cpu_recomputed_relative_residual: float | None
    target_solution_relative_error: float | None
    target_failure_reason: str | None
    target_applicability_status: str
    target_applicability_reason: str | None
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CsrSelectorSummary:
    status: str
    schema_version: str
    num_diagnostic_rows: int
    num_selector_rows: int
    num_success_rows: int
    num_failed_rows: int
    num_applicability_rows: int
    num_oracle_rows: int
    num_matrices_with_selector_rows: int
    max_final_relative_residual: float
    max_cpu_recomputed_relative_residual: float
    max_solution_relative_error: float
    max_success_final_relative_residual: float
    max_success_cpu_recomputed_relative_residual: float
    max_success_solution_relative_error: float
    max_failed_final_relative_residual: float
    min_success_rate: float
    max_solve_time_iqr_ms: float
    failure_reason_counts: dict[str, int] = field(default_factory=dict)
    applicability_status_counts: dict[str, int] = field(default_factory=dict)
    applicability_reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CsrSelectorExport:
    diagnostics: tuple[CsrDiagnosticFeatureRow, ...]
    selector_rows: tuple[CsrSelectorRow, ...]
    summary: CsrSelectorSummary


def build_csr_selector_export(
    csr_records: Iterable[dict[str, Any]],
    solve_rows: Iterable[dict[str, Any]],
    *,
    context_id: str = "phase1_csr_selector",
) -> CsrSelectorExport:
    diagnostics = tuple(_diagnostic_row(row) for row in csr_records)
    diagnostic_by_id = {row.matrix_id: row for row in diagnostics}
    solve_rows_tuple = tuple(solve_rows)
    oracle_candidates = _oracle_candidates(solve_rows_tuple)
    selector_rows = tuple(
        _selector_row(
            row,
            diagnostic_by_id=diagnostic_by_id,
            oracle_candidates=oracle_candidates,
            context_id=context_id,
        )
        for row in solve_rows_tuple
    )
    summary = _summary(diagnostics, selector_rows)
    return CsrSelectorExport(
        diagnostics=diagnostics,
        selector_rows=selector_rows,
        summary=summary,
    )


def build_csr_selector_export_from_files(
    csr_path: str | Path,
    solve_results_path: str | Path | Iterable[str | Path],
    *,
    screened_results_path: str | Path | Iterable[str | Path] | None = None,
    context_id: str = "phase1_csr_selector",
) -> CsrSelectorExport:
    csr_rows = tuple(read_jsonl(csr_path))
    solve_rows = list(_read_solve_rows(solve_results_path))
    if screened_results_path is not None:
        solve_rows.extend(_read_screened_rows(screened_results_path, csr_rows))
    return build_csr_selector_export(
        csr_rows,
        solve_rows,
        context_id=context_id,
    )


def write_csr_diagnostic_rows(
    rows: Iterable[CsrDiagnosticFeatureRow],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_selector_rows(
    rows: Iterable[CsrSelectorRow],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_selector_summary(summary: CsrSelectorSummary, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_csr_selector_schema(path: str | Path) -> Path:
    schema = {
        "schema_version": CSR_SELECTOR_SCHEMA_VERSION,
        "task": "rank_real_csr_solver_preconditioner_precision_candidates",
        "input_features": {
            "matrix": [
                "n_rows",
                "n_cols",
                "csr_nnz",
                "field",
                "declared_symmetry",
                "actual_symmetric",
                "symmetry_missing_pairs",
                "symmetry_relative_error",
                "diagonal_present_count",
                "zero_diagonal_count",
                "min_abs_diagonal",
                "max_abs_diagonal",
                "has_nonpositive_diagonal",
                "recommended_precision",
            ],
            "candidate": [
                "solver",
                "preconditioner",
                "precision",
                "solver_parameters",
                "measurement_repeats",
                "success_rate",
                "median_solve_time_ms",
                "solve_time_iqr_ms",
                "screened_out_source",
                "failure_reason",
                "applicability_status",
                "applicability_reason",
            ],
        },
        "targets": [
            "label_is_oracle",
            "target_status",
            "target_success_rate",
            "target_measurement_repeats",
            "target_solve_time_ms",
            "target_median_solve_time_ms",
            "target_solve_time_iqr_ms",
            "target_regret_vs_oracle_ms",
            "target_num_iterations",
            "target_final_relative_residual",
            "target_cpu_recomputed_relative_residual",
            "target_solution_relative_error",
            "target_failure_reason",
            "target_applicability_status",
            "target_applicability_reason",
        ],
        "integration_boundary": {
            "status": "ready_for_real_csr_scaleout",
            "model_required": False,
            "next_step": "scale_real_csr_solver_rows_and_train_selector",
        },
        "candidate_id_policy": {
            "parameterized_candidates": ["gmres.restart"],
            "selection_filter": (
                "runtime selectors must require target_status=success and "
                "target_success_rate=1.0"
            ),
        },
        "negative_row_policy": {
            "status_values": ["screened_out"],
            "source": "CPU reference screens recorded before GPU execution",
            "accepted_reasons": [
                "cpu_reference_cg_screen_failed",
                "cpu_reference_bicgstab_screen_failed",
                "cpu_reference_gmres_screen_failed",
                "cpu_reference_richardson_screen_failed",
                "cpu_reference_chebyshev_screen_failed",
            ],
        },
        "applicability_label_policy": {
            "status_values": ["not_applicable", "not_profiled"],
            "not_applicable_reasons": ["not_symmetric"],
            "not_profiled_reasons": [
                "after_selection_limit",
                "above_smoke_size_limit",
            ],
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_csr_selector_report(export: CsrSelectorExport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = export.summary
    lines = [
        "# CSR Selector Readiness",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- diagnostic_rows: `{summary.num_diagnostic_rows}`",
        f"- selector_rows: `{summary.num_selector_rows}`",
        f"- success_rows: `{summary.num_success_rows}`",
        f"- failed_rows: `{summary.num_failed_rows}`",
        f"- applicability_rows: `{summary.num_applicability_rows}`",
        f"- oracle_rows: `{summary.num_oracle_rows}`",
        f"- matrices_with_selector_rows: `{summary.num_matrices_with_selector_rows}`",
        f"- max_final_relative_residual: `{summary.max_final_relative_residual:.6g}`",
        f"- max_cpu_recomputed_relative_residual: "
        f"`{summary.max_cpu_recomputed_relative_residual:.6g}`",
        f"- max_solution_relative_error: `{summary.max_solution_relative_error:.6g}`",
        f"- min_success_rate: `{summary.min_success_rate:.6g}`",
        f"- max_solve_time_iqr_ms: `{summary.max_solve_time_iqr_ms:.6g}`",
        f"- failure_reason_counts: `{dict(summary.failure_reason_counts)}`",
        f"- applicability_status_counts: `{dict(summary.applicability_status_counts)}`",
        f"- applicability_reason_counts: `{dict(summary.applicability_reason_counts)}`",
        "",
        "| matrix | candidate | oracle | status | failure | applicability | reason | success_rate | repeats | median_ms | regret_ms | rel_res | sol_err |",
        "|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in export.selector_rows:
        lines.append(
            "| "
            f"{row.matrix_id} | "
            f"{row.candidate_id} | "
            f"{'yes' if row.label_is_oracle else 'no'} | "
            f"{row.target_status} | "
            f"{row.target_failure_reason or ''} | "
            f"{row.target_applicability_status} | "
            f"{row.target_applicability_reason or ''} | "
            f"{_fmt(row.target_success_rate)} | "
            f"{row.target_measurement_repeats} | "
            f"{_fmt(row.target_median_solve_time_ms)} | "
            f"{_fmt(row.target_regret_vs_oracle_ms)} | "
            f"{_fmt(row.target_final_relative_residual)} | "
            f"{_fmt(row.target_solution_relative_error)} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _diagnostic_row(row: dict[str, Any]) -> CsrDiagnosticFeatureRow:
    diagnostic = diagnose_csr_matrix(csr_matrix_from_record(row))
    return CsrDiagnosticFeatureRow(
        schema_version=CSR_SELECTOR_SCHEMA_VERSION,
        matrix_id=diagnostic.matrix_id,
        features=asdict(diagnostic),
    )


def _read_solve_rows(paths: str | Path | Iterable[str | Path]) -> tuple[dict[str, Any], ...]:
    if isinstance(paths, str | Path):
        path_tuple = (paths,)
    else:
        path_tuple = tuple(paths)
    rows: list[dict[str, Any]] = []
    for path in path_tuple:
        rows.extend(read_jsonl(path))
    return tuple(rows)


def _read_screened_rows(
    paths: str | Path | Iterable[str | Path],
    csr_records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(paths, str | Path):
        path_tuple = (paths,)
    else:
        path_tuple = tuple(paths)
    csr_by_id = {str(row["matrix_id"]): row for row in csr_records}
    rows: list[dict[str, Any]] = []
    for path in path_tuple:
        source_path = Path(path)
        summary = json.loads(source_path.read_text(encoding="utf-8"))
        precision = str(summary.get("precision", "unknown"))
        max_iter = _optional_int(summary.get("max_iter"))
        rows.extend(
            _screened_rows_from_summary(
                summary,
                csr_by_id=csr_by_id,
                precision=precision,
                max_iter=max_iter,
                source_path=source_path,
            )
        )
    return tuple(rows)


def _screened_rows_from_summary(
    summary: dict[str, Any],
    *,
    csr_by_id: dict[str, dict[str, Any]],
    precision: str,
    max_iter: int | None,
    source_path: Path,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for config in summary.get("screened_out_restart_configs", ()):
        rows.append(
            _screened_out_row(
                matrix_id=str(config["matrix_id"]),
                solver="gmres",
                preconditioner=str(config["preconditioner"]),
                precision=precision,
                solver_parameters={"restart": int(config["restart"])},
                screen=dict(config["screen"]),
                failure_reason="cpu_reference_gmres_screen_failed",
                csr_by_id=csr_by_id,
                max_iter=max_iter,
                source_path=source_path,
            )
        )
    for skipped in summary.get("skipped_candidates", ()):
        reason = str(skipped.get("reason", ""))
        if reason == "cpu_reference_cg_screen_failed":
            if "cg_screen" in skipped:
                rows.append(
                    _screened_out_row(
                        matrix_id=str(skipped["matrix_id"]),
                        solver="cg",
                        preconditioner="none",
                        precision=precision,
                        solver_parameters={},
                        screen=dict(skipped["cg_screen"]),
                        failure_reason=reason,
                        csr_by_id=csr_by_id,
                        max_iter=max_iter,
                        source_path=source_path,
                    )
                )
            if "pcg_screen" in skipped:
                rows.append(
                    _screened_out_row(
                        matrix_id=str(skipped["matrix_id"]),
                        solver="pcg",
                        preconditioner="jacobi",
                        precision=precision,
                        solver_parameters={},
                        screen=dict(skipped["pcg_screen"]),
                        failure_reason=reason,
                        csr_by_id=csr_by_id,
                        max_iter=max_iter,
                        source_path=source_path,
                    )
                )
        elif reason == "cpu_reference_bicgstab_screen_failed":
            for preconditioner, screen in skipped.get("screens", {}).items():
                rows.append(
                    _screened_out_row(
                        matrix_id=str(skipped["matrix_id"]),
                        solver="bicgstab",
                        preconditioner=str(preconditioner),
                        precision=precision,
                        solver_parameters={},
                        screen=dict(screen),
                        failure_reason=reason,
                        csr_by_id=csr_by_id,
                        max_iter=max_iter,
                        source_path=source_path,
                    )
                )
        elif reason == "cpu_reference_gmres_screen_failed":
            for preconditioner, restart_screens in skipped.get("screens", {}).items():
                for restart, screen in restart_screens.items():
                    rows.append(
                        _screened_out_row(
                            matrix_id=str(skipped["matrix_id"]),
                            solver="gmres",
                            preconditioner=str(preconditioner),
                            precision=precision,
                            solver_parameters={"restart": int(restart)},
                            screen=dict(screen),
                            failure_reason=reason,
                            csr_by_id=csr_by_id,
                            max_iter=max_iter,
                            source_path=source_path,
                        )
                    )
        elif reason == "cpu_reference_richardson_screen_failed":
            screen = dict(skipped["screen"])
            rows.append(
                _screened_out_row(
                    matrix_id=str(skipped["matrix_id"]),
                    solver="richardson",
                    preconditioner="jacobi",
                    precision=precision,
                    solver_parameters={
                        "omega": screen.get("omega", summary.get("omega")),
                    },
                    screen=screen,
                    failure_reason=reason,
                    csr_by_id=csr_by_id,
                    max_iter=max_iter,
                    source_path=source_path,
                )
            )
        elif reason == "cpu_reference_chebyshev_screen_failed":
            screen = dict(skipped["screen"])
            bounds = dict(skipped.get("bounds", {}))
            rows.append(
                _screened_out_row(
                    matrix_id=str(skipped["matrix_id"]),
                    solver="chebyshev",
                    preconditioner="jacobi",
                    precision=precision,
                    solver_parameters={
                        "lambda_min": screen.get("lambda_min", bounds.get("lambda_min")),
                        "lambda_max": screen.get("lambda_max", bounds.get("lambda_max")),
                        "spectral_bounds_source": bounds.get("source"),
                    },
                    screen=screen,
                    failure_reason=reason,
                    csr_by_id=csr_by_id,
                    max_iter=max_iter,
                    source_path=source_path,
                )
            )
        elif reason in STRUCTURAL_SKIP_STATUS_BY_REASON:
            for solver, preconditioner, solver_parameters in _structural_candidates(
                source_path=source_path,
                summary=summary,
            ):
                rows.append(
                    _applicability_row(
                        matrix_id=str(skipped["matrix_id"]),
                        solver=solver,
                        preconditioner=preconditioner,
                        precision=precision,
                        solver_parameters=solver_parameters,
                        applicability_status=STRUCTURAL_SKIP_STATUS_BY_REASON[reason],
                        applicability_reason=reason,
                        csr_by_id=csr_by_id,
                        source_path=source_path,
                    )
                )
    return tuple(rows)


def _structural_candidates(
    *,
    source_path: Path,
    summary: dict[str, Any],
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    name = source_path.name
    if name == "csr_solve_summary.json":
        return (
            ("cg", "none", {}),
            ("pcg", "jacobi", {}),
        )
    if name == "csr_bicgstab_summary.json":
        return (
            ("bicgstab", "none", {}),
            ("bicgstab", "jacobi", {}),
        )
    if name == "csr_gmres_summary.json":
        return tuple(
            ("gmres", "jacobi", {"restart": int(restart)})
            for restart in summary.get("restarts", ())
        )
    if name == "csr_richardson_summary.json":
        return (
            ("richardson", "jacobi", {"omega": summary.get("omega")}),
        )
    if name == "csr_chebyshev_summary.json":
        return (("chebyshev", "jacobi", {}),)
    return ()


def _screened_out_row(
    *,
    matrix_id: str,
    solver: str,
    preconditioner: str,
    precision: str,
    solver_parameters: dict[str, Any],
    screen: dict[str, Any],
    failure_reason: str,
    csr_by_id: dict[str, dict[str, Any]],
    max_iter: int | None,
    source_path: Path,
) -> dict[str, Any]:
    csr = csr_by_id[matrix_id]
    row = {
        "backend": "cpu_reference_screen",
        "csr_nnz": int(csr["csr_nnz"]),
        "cpu_recomputed_relative_residual": screen.get("relative_residual"),
        "cpu_screen": screen,
        "failure_reason": failure_reason,
        "failure_reasons": (failure_reason,),
        "field": str(csr["field"]),
        "final_relative_residual": screen.get("relative_residual"),
        "input": "rhs=A@ones",
        "matrix_id": matrix_id,
        "measurement_repeats": 1,
        "n_cols": int(csr["n_cols"]),
        "n_rows": int(csr["n_rows"]),
        "num_iterations": screen.get("iterations"),
        "precision": precision,
        "preconditioner": preconditioner,
        "screened_out_source": str(source_path),
        "solver": solver,
        "solution_relative_error": screen.get("solution_relative_error"),
        "status": "screened_out",
        "success_count": 0,
        "success_rate": 0.0,
        "symmetry": str(csr["symmetry"]),
        "target_screen_max_iter": max_iter,
    }
    row.update(
        {
            key: value
            for key, value in solver_parameters.items()
            if value is not None
        }
    )
    return row


def _applicability_row(
    *,
    matrix_id: str,
    solver: str,
    preconditioner: str,
    precision: str,
    solver_parameters: dict[str, Any],
    applicability_status: str,
    applicability_reason: str,
    csr_by_id: dict[str, dict[str, Any]],
    source_path: Path,
) -> dict[str, Any]:
    csr = csr_by_id[matrix_id]
    row = {
        "backend": "not_run_applicability_label",
        "csr_nnz": int(csr["csr_nnz"]),
        "field": str(csr["field"]),
        "input": "rhs=A@ones",
        "matrix_id": matrix_id,
        "measurement_repeats": 0,
        "n_cols": int(csr["n_cols"]),
        "n_rows": int(csr["n_rows"]),
        "precision": precision,
        "preconditioner": preconditioner,
        "screened_out_source": str(source_path),
        "solver": solver,
        "status": applicability_status,
        "success_count": 0,
        "success_rate": 0.0,
        "symmetry": str(csr["symmetry"]),
        "applicability_status": applicability_status,
        "applicability_reason": applicability_reason,
    }
    row.update(
        {
            key: value
            for key, value in solver_parameters.items()
            if value is not None
        }
    )
    return row


def _selector_row(
    row: dict[str, Any],
    *,
    diagnostic_by_id: dict[str, CsrDiagnosticFeatureRow],
    oracle_candidates: dict[str, tuple[float, float, str]],
    context_id: str,
) -> CsrSelectorRow:
    matrix_id = str(row["matrix_id"])
    solver = str(row["solver"])
    preconditioner = str(row["preconditioner"])
    precision = str(row.get("precision", "unknown"))
    solve_time = _optional_float(row.get("solve_time_ms"))
    median_solve_time = _optional_float(row.get("median_solve_time_ms", solve_time))
    wall_time = _optional_float(row.get("wall_time_ms"))
    solve_time_iqr = _optional_float(row.get("solve_time_iqr_ms", 0.0))
    success_rate = _optional_float(
        row.get("success_rate", 1.0 if row.get("status") == "success" else 0.0)
    )
    measurement_repeats = _optional_int(row.get("measurement_repeats", 1))
    if measurement_repeats is None:
        measurement_repeats = 1
    solver_parameters = _solver_parameters_from_row(row)
    status = str(row["status"])
    applicability_status = str(
        row.get(
            "applicability_status",
            "applicable" if status in {"success", "screened_out"} else status,
        )
    )
    applicability_reason = (
        None
        if row.get("applicability_reason") is None
        else str(row.get("applicability_reason"))
    )
    candidate_id = _candidate_id(
        solver=solver,
        preconditioner=preconditioner,
        precision=precision,
        solver_parameters=solver_parameters,
    )
    oracle = oracle_candidates.get(matrix_id)
    oracle_time = oracle[0] if oracle is not None else None
    oracle_candidate_id = oracle[2] if oracle is not None else None
    features = dict(diagnostic_by_id[matrix_id].features)
    features.update(
        {
            "candidate_id": candidate_id,
            "solver": solver,
            "preconditioner": preconditioner,
            "precision": precision,
            "solver_parameters": solver_parameters,
            "measurement_repeats": measurement_repeats,
            "success_rate": success_rate,
            "median_solve_time_ms": median_solve_time,
            "solve_time_iqr_ms": solve_time_iqr,
            "screened_out_source": row.get("screened_out_source"),
            "failure_reason": row.get("failure_reason"),
            "applicability_status": applicability_status,
            "applicability_reason": applicability_reason,
        }
    )
    return CsrSelectorRow(
        schema_version=CSR_SELECTOR_SCHEMA_VERSION,
        matrix_id=matrix_id,
        context_id=context_id,
        candidate_id=candidate_id,
        solver=solver,
        preconditioner=preconditioner,
        precision=precision,
        solver_parameters=solver_parameters,
        label_is_oracle=status == "success" and candidate_id == oracle_candidate_id,
        target_status=status,
        target_success_rate=success_rate or 0.0,
        target_measurement_repeats=measurement_repeats,
        target_solve_time_ms=solve_time,
        target_median_solve_time_ms=median_solve_time,
        target_solve_time_iqr_ms=solve_time_iqr,
        target_wall_time_ms=wall_time,
        target_regret_vs_oracle_ms=(
            None
            if median_solve_time is None or oracle_time is None
            else median_solve_time - oracle_time
        ),
        target_num_iterations=_optional_int(row.get("num_iterations")),
        target_final_relative_residual=_optional_float(row.get("final_relative_residual")),
        target_cpu_recomputed_relative_residual=_optional_float(
            row.get("cpu_recomputed_relative_residual")
        ),
        target_solution_relative_error=_optional_float(row.get("solution_relative_error")),
        target_failure_reason=(
            str(row.get("failure_reason"))
            if status == "screened_out" and row.get("failure_reason") is not None
            else None
        ),
        target_applicability_status=applicability_status,
        target_applicability_reason=applicability_reason,
        features=features,
    )


def _solver_parameters_from_row(row: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in (
        "restart",
        "omega",
        "lambda_min",
        "lambda_max",
        "spectral_bounds_source",
    ):
        if key in row and row[key] is not None:
            params[key] = row[key]
    return params


def _oracle_candidates(rows: tuple[dict[str, Any], ...]) -> dict[str, tuple[float, float, str]]:
    oracle: dict[str, tuple[float, float, str]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        success_rate = _optional_float(row.get("success_rate", 1.0))
        if success_rate is None or success_rate < 1.0:
            continue
        solve_time = _optional_float(row.get("median_solve_time_ms", row.get("solve_time_ms")))
        if solve_time is None:
            continue
        solve_iqr = _optional_float(row.get("solve_time_iqr_ms", 0.0)) or 0.0
        matrix_id = str(row["matrix_id"])
        candidate_id = _candidate_id(
            solver=str(row["solver"]),
            preconditioner=str(row["preconditioner"]),
            precision=str(row.get("precision", "unknown")),
            solver_parameters=_solver_parameters_from_row(row),
        )
        current = oracle.get(matrix_id)
        candidate = (solve_time, solve_iqr, candidate_id)
        if current is None or candidate < current:
            oracle[matrix_id] = candidate
    return oracle


def _summary(
    diagnostics: tuple[CsrDiagnosticFeatureRow, ...],
    selector_rows: tuple[CsrSelectorRow, ...],
) -> CsrSelectorSummary:
    success_rows = tuple(row for row in selector_rows if row.target_status == "success")
    failed_rows = tuple(row for row in selector_rows if row.target_status == "screened_out")
    applicability_rows = tuple(
        row
        for row in selector_rows
        if row.target_status in {"not_applicable", "not_profiled"}
    )
    num_oracle = sum(1 for row in selector_rows if row.label_is_oracle)
    max_final = max(
        (row.target_final_relative_residual or 0.0 for row in success_rows),
        default=0.0,
    )
    max_cpu = max(
        (row.target_cpu_recomputed_relative_residual or 0.0 for row in success_rows),
        default=0.0,
    )
    max_solution = max(
        (row.target_solution_relative_error or 0.0 for row in success_rows),
        default=0.0,
    )
    max_failed_final = max(
        (row.target_final_relative_residual or 0.0 for row in failed_rows),
        default=0.0,
    )
    min_success_rate = min(
        (row.target_success_rate for row in selector_rows),
        default=0.0,
    )
    max_iqr = max(
        (row.target_solve_time_iqr_ms or 0.0 for row in selector_rows),
        default=0.0,
    )
    failure_reason_counts = dict(
        sorted(
            Counter(
                row.target_failure_reason
                for row in failed_rows
                if row.target_failure_reason
            ).items()
        )
    )
    applicability_status_counts = dict(
        sorted(Counter(row.target_status for row in applicability_rows).items())
    )
    applicability_reason_counts = dict(
        sorted(
            Counter(
                row.target_applicability_reason
                for row in applicability_rows
                if row.target_applicability_reason
            ).items()
        )
    )
    status = (
        "passed"
        if diagnostics
        and selector_rows
        and success_rows
        and num_oracle == len({row.matrix_id for row in success_rows})
        and all(row.target_status == "screened_out" for row in failed_rows)
        and all(
            row.target_status in {"not_applicable", "not_profiled"}
            for row in applicability_rows
        )
        and len(success_rows) + len(failed_rows) + len(applicability_rows)
        == len(selector_rows)
        and max_final <= 1.0e-5
        and max_cpu <= 1.0e-4
        and max_solution <= 5.0e-3
        and all(row.target_success_rate >= 1.0 for row in success_rows)
        and all(row.target_success_rate < 1.0 for row in failed_rows)
        and all(row.target_success_rate < 1.0 for row in applicability_rows)
        and all(row.target_applicability_status == "applicable" for row in success_rows)
        and all(row.target_applicability_status == "applicable" for row in failed_rows)
        and all(row.target_applicability_reason for row in applicability_rows)
        else "failed"
    )
    return CsrSelectorSummary(
        status=status,
        schema_version=CSR_SELECTOR_SCHEMA_VERSION,
        num_diagnostic_rows=len(diagnostics),
        num_selector_rows=len(selector_rows),
        num_success_rows=len(success_rows),
        num_failed_rows=len(failed_rows),
        num_applicability_rows=len(applicability_rows),
        num_oracle_rows=num_oracle,
        num_matrices_with_selector_rows=len({row.matrix_id for row in selector_rows}),
        max_final_relative_residual=max_final,
        max_cpu_recomputed_relative_residual=max_cpu,
        max_solution_relative_error=max_solution,
        max_success_final_relative_residual=max_final,
        max_success_cpu_recomputed_relative_residual=max_cpu,
        max_success_solution_relative_error=max_solution,
        max_failed_final_relative_residual=max_failed_final,
        min_success_rate=min_success_rate,
        max_solve_time_iqr_ms=max_iqr,
        failure_reason_counts=failure_reason_counts,
        applicability_status_counts=applicability_status_counts,
        applicability_reason_counts=applicability_reason_counts,
    )


def _candidate_id(
    *,
    solver: str,
    preconditioner: str,
    precision: str,
    solver_parameters: dict[str, Any],
) -> str:
    parts = ["taichi", "csr", solver, preconditioner]
    if solver == "gmres" and "restart" in solver_parameters:
        parts.append(f"restart{int(solver_parameters['restart'])}")
    parts.append(precision)
    return "_".join(parts)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"
