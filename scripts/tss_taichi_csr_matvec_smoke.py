"""Run a small Taichi GPU CSR matvec smoke test against imported CSR fixtures."""

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
    CORE_TAICHI_CSR_MATVEC_PROVENANCE_FILES,
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
    parser.add_argument("--out", default="runs/phase1_taichi_csr_matvec")
    parser.add_argument("--max-matrices", type=int, default=4)
    parser.add_argument("--device-memory-gb", type=float, default=0.25)
    args = parser.parse_args()

    rows = read_jsonl(args.csr)[: args.max_matrices]
    if not rows:
        raise SystemExit("no CSR rows available for Taichi CSR matvec smoke")
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
        "results": output / "csr_matvec_results.jsonl",
        "summary": output / "csr_matvec_summary.json",
        "report": output / "csr_matvec_report.md",
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
            artifact_kind="taichi_csr_matvec_smoke",
            command="scripts/tss_taichi_csr_matvec_smoke.py",
            tracked_files=CORE_TAICHI_CSR_MATVEC_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "num_matrices": summary["num_matrices"],
                "num_success": summary["num_success"],
                "num_failed": summary["num_failed"],
                "max_abs_error": summary["max_abs_error"],
                "max_relative_l2_error": summary["max_relative_l2_error"],
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
    expected = csr.matvec(x)
    setup_start = time.perf_counter()
    operator = TaichiCsrMatrixOperator.from_csr_matrix(csr)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    matvec_start = time.perf_counter()
    actual = operator.apply(x)
    matvec_ms = (time.perf_counter() - matvec_start) * 1000.0
    max_abs = _max_abs_error(expected, actual)
    rel_l2 = _relative_l2_error(expected, actual)
    scale = max(1.0, max((abs(value) for value in expected), default=0.0))
    tolerance = max(1.0e-4, 1.0e-5 * scale)
    status = "success" if max_abs <= tolerance and rel_l2 <= 1.0e-5 else "failed"
    return {
        "matrix_id": csr.matrix_id,
        "status": status,
        "n_rows": csr.n_rows,
        "n_cols": csr.n_cols,
        "csr_nnz": csr.nnz,
        "field": csr.field,
        "symmetry": csr.symmetry,
        "setup_time_ms": setup_ms,
        "matvec_time_ms": matvec_ms,
        "max_abs_error": max_abs,
        "relative_l2_error": rel_l2,
        "tolerance_abs": tolerance,
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


def _max_abs_error(expected: tuple[float, ...], actual: tuple[float, ...]) -> float:
    return max((abs(a - b) for a, b in zip(expected, actual)), default=0.0)


def _relative_l2_error(expected: tuple[float, ...], actual: tuple[float, ...]) -> float:
    diff_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(expected, actual)))
    expected_norm = max(math.sqrt(sum(value * value for value in expected)), 1.0e-30)
    return diff_norm / expected_norm


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
        "max_abs_error": max((float(row["max_abs_error"]) for row in results), default=0.0),
        "max_relative_l2_error": max(
            (float(row["relative_l2_error"]) for row in results),
            default=0.0,
        ),
    }


def _write_report(results: tuple[dict, ...], summary: dict, path: Path) -> Path:
    lines = [
        "# Taichi CSR Matvec Smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- matrices: `{summary['num_matrices']}`",
        f"- success: `{summary['num_success']}`",
        f"- failed: `{summary['num_failed']}`",
        f"- total_csr_nnz: `{summary['total_csr_nnz']}`",
        f"- max_abs_error: `{summary['max_abs_error']:.6g}`",
        f"- max_relative_l2_error: `{summary['max_relative_l2_error']:.6g}`",
        "",
        "| matrix | status | shape | nnz | field | symmetry | max_abs | rel_l2 | matvec_ms |",
        "|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| "
            f"{row['matrix_id']} | "
            f"{row['status']} | "
            f"{row['n_rows']}x{row['n_cols']} | "
            f"{row['csr_nnz']} | "
            f"{row['field']} | "
            f"{row['symmetry']} | "
            f"{row['max_abs_error']:.6g} | "
            f"{row['relative_l2_error']:.6g} | "
            f"{row['matvec_time_ms']:.6g} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
