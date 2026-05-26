"""Run Taichi GPU CSR vector-primitive smoke tests on imported CSR fixtures."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.csr import CsrMatrix
from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl
from transsolvestack.profiling.provenance import (
    CORE_TAICHI_CSR_PRIMITIVES_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csr",
        default="runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_taichi_csr_primitives")
    parser.add_argument("--max-matrices", type=int, default=4)
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    args = parser.parse_args()

    rows = read_jsonl(args.csr)[: args.max_matrices]
    if not rows:
        raise SystemExit("no CSR rows available for Taichi CSR primitive smoke")
    ensure_taichi_cuda(device_memory_gb=args.device_memory_gb)
    results = tuple(_run_one(row) for row in rows)
    summary = _build_summary(
        results,
        source_csr_path=args.csr,
        device_memory_gb=args.device_memory_gb,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output / "csr_primitives_results.jsonl",
        "summary": output / "csr_primitives_summary.json",
        "report": output / "csr_primitives_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_jsonl(results, paths["results"])
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(results, summary, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="taichi_csr_primitives_smoke",
            command="scripts/tss_taichi_csr_primitives_smoke.py",
            tracked_files=CORE_TAICHI_CSR_PRIMITIVES_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "num_matrices": summary["num_matrices"],
                "num_success": summary["num_success"],
                "num_failed": summary["num_failed"],
                "max_matvec_relative_l2_error": summary[
                    "max_matvec_relative_l2_error"
                ],
                "max_residual_relative_norm": summary[
                    "max_residual_relative_norm"
                ],
                "max_dot_relative_error": summary["max_dot_relative_error"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


def _run_one(row: dict) -> dict:
    csr = _csr_from_record(row)
    x = tuple(1.0 for _ in range(csr.n_cols))
    expected_y = csr.matvec(x)
    expected_dot_y_y = _dot(expected_y, expected_y)
    expected_norm_y = math.sqrt(max(expected_dot_y_y, 0.0))

    setup_start = time.perf_counter()
    operator = TaichiCsrMatrixOperator.from_csr_matrix(csr)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    matvec_start = time.perf_counter()
    actual_y = operator.apply(x)
    matvec_ms = (time.perf_counter() - matvec_start) * 1000.0

    residual_start = time.perf_counter()
    residual = operator.residual(b=expected_y)
    residual_ms = (time.perf_counter() - residual_start) * 1000.0

    dot_start = time.perf_counter()
    dot_y_y = operator.dot("y", "y")
    dot_ms = (time.perf_counter() - dot_start) * 1000.0

    norm_start = time.perf_counter()
    norm_y = operator.norm("y")
    residual_norm = operator.norm("r")
    norm_ms = (time.perf_counter() - norm_start) * 1000.0

    matvec_max_abs = _max_abs_error(expected_y, actual_y)
    matvec_rel_l2 = _relative_l2_error(expected_y, actual_y)
    scale = max(1.0, max((abs(value) for value in expected_y), default=0.0))
    matvec_abs_tolerance = max(1.0e-4, 1.0e-5 * scale)
    residual_abs = max((abs(value) for value in residual), default=0.0)
    residual_relative_norm = residual_norm / max(expected_norm_y, 1.0e-30)
    dot_relative_error = _relative_scalar_error(expected_dot_y_y, dot_y_y)
    norm_relative_error = _relative_scalar_error(expected_norm_y, norm_y)

    status = "success"
    failure_reasons: list[str] = []
    if matvec_max_abs > matvec_abs_tolerance or matvec_rel_l2 > 1.0e-5:
        status = "failed"
        failure_reasons.append("matvec_error")
    if residual_relative_norm > 1.0e-5 or residual_abs > matvec_abs_tolerance:
        status = "failed"
        failure_reasons.append("residual_error")
    if dot_relative_error > 1.0e-4:
        status = "failed"
        failure_reasons.append("dot_error")
    if norm_relative_error > 1.0e-4:
        status = "failed"
        failure_reasons.append("norm_error")

    return {
        "matrix_id": csr.matrix_id,
        "status": status,
        "failure_reasons": tuple(failure_reasons),
        "n_rows": csr.n_rows,
        "n_cols": csr.n_cols,
        "csr_nnz": csr.nnz,
        "field": csr.field,
        "symmetry": csr.symmetry,
        "setup_time_ms": setup_ms,
        "matvec_time_ms": matvec_ms,
        "residual_time_ms": residual_ms,
        "dot_time_ms": dot_ms,
        "norm_time_ms": norm_ms,
        "matvec_max_abs_error": matvec_max_abs,
        "matvec_relative_l2_error": matvec_rel_l2,
        "matvec_tolerance_abs": matvec_abs_tolerance,
        "residual_norm": residual_norm,
        "residual_relative_norm": residual_relative_norm,
        "residual_max_abs": residual_abs,
        "dot_y_y": dot_y_y,
        "expected_dot_y_y": expected_dot_y_y,
        "dot_relative_error": dot_relative_error,
        "norm_y": norm_y,
        "expected_norm_y": expected_norm_y,
        "norm_relative_error": norm_relative_error,
        "input": "ones",
        "backend": "taichi_gpu",
    }


def _csr_from_record(row: dict) -> CsrMatrix:
    csr = CsrMatrix(
        matrix_id=str(row["matrix_id"]),
        n_rows=int(row["n_rows"]),
        n_cols=int(row["n_cols"]),
        row_ptr=tuple(int(value) for value in row["row_ptr"]),
        col_ind=tuple(int(value) for value in row["col_ind"]),
        values=tuple(float(value) for value in row["values"]),
        field=str(row["field"]),
        symmetry=str(row["symmetry"]),
        source_path=str(row["source_archive_path"]),
        metadata=dict(row.get("metadata", {})),
    )
    csr.validate()
    return csr


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _max_abs_error(expected: tuple[float, ...], actual: tuple[float, ...]) -> float:
    return max((abs(a - b) for a, b in zip(expected, actual)), default=0.0)


def _relative_l2_error(expected: tuple[float, ...], actual: tuple[float, ...]) -> float:
    diff_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(expected, actual)))
    expected_norm = max(math.sqrt(sum(value * value for value in expected)), 1.0e-30)
    return diff_norm / expected_norm


def _relative_scalar_error(expected: float, actual: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-30)


def _build_summary(
    results: tuple[dict, ...],
    *,
    source_csr_path: str,
    device_memory_gb: float,
) -> dict:
    num_success = sum(1 for row in results if row["status"] == "success")
    num_failed = len(results) - num_success
    return {
        "status": "passed" if results and num_failed == 0 else "failed",
        "source_csr_path": source_csr_path,
        "device_memory_gb": device_memory_gb,
        "num_matrices": len(results),
        "num_success": num_success,
        "num_failed": num_failed,
        "total_csr_nnz": sum(int(row["csr_nnz"]) for row in results),
        "max_matvec_abs_error": max(
            (float(row["matvec_max_abs_error"]) for row in results),
            default=0.0,
        ),
        "max_matvec_relative_l2_error": max(
            (float(row["matvec_relative_l2_error"]) for row in results),
            default=0.0,
        ),
        "max_residual_norm": max(
            (float(row["residual_norm"]) for row in results),
            default=0.0,
        ),
        "max_residual_relative_norm": max(
            (float(row["residual_relative_norm"]) for row in results),
            default=0.0,
        ),
        "max_residual_abs": max(
            (float(row["residual_max_abs"]) for row in results),
            default=0.0,
        ),
        "max_dot_relative_error": max(
            (float(row["dot_relative_error"]) for row in results),
            default=0.0,
        ),
        "max_norm_relative_error": max(
            (float(row["norm_relative_error"]) for row in results),
            default=0.0,
        ),
    }


def _write_report(results: tuple[dict, ...], summary: dict, path: Path) -> Path:
    lines = [
        "# Taichi CSR Primitive Smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- matrices: `{summary['num_matrices']}`",
        f"- success: `{summary['num_success']}`",
        f"- failed: `{summary['num_failed']}`",
        f"- total_csr_nnz: `{summary['total_csr_nnz']}`",
        f"- max_matvec_relative_l2_error: "
        f"`{summary['max_matvec_relative_l2_error']:.6g}`",
        f"- max_residual_relative_norm: "
        f"`{summary['max_residual_relative_norm']:.6g}`",
        f"- max_dot_relative_error: `{summary['max_dot_relative_error']:.6g}`",
        f"- max_norm_relative_error: `{summary['max_norm_relative_error']:.6g}`",
        "",
        "| matrix | status | shape | nnz | rel_l2 | residual_rel | dot_rel | norm_rel |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['status']} | "
            f"{row['n_rows']}x{row['n_cols']} | "
            f"{row['csr_nnz']} | "
            f"{row['matvec_relative_l2_error']:.6g} | "
            f"{row['residual_relative_norm']:.6g} | "
            f"{row['dot_relative_error']:.6g} | "
            f"{row['norm_relative_error']:.6g} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
