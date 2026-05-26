"""Plan a resource-safe expansion of real CSR benchmark coverage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_benchmark_expansion import (
    build_csr_benchmark_expansion_plan_from_files,
    write_csr_benchmark_candidate_plan,
    write_csr_benchmark_expansion_report,
    write_csr_benchmark_expansion_schema,
    write_csr_benchmark_expansion_summary,
    write_csr_benchmark_matrix_plan,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_BENCHMARK_EXPANSION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default="runs/phase1_suitesparse_selection/selected_matrices.jsonl",
    )
    parser.add_argument(
        "--selector-rows",
        default="runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    )
    parser.add_argument("--resource-limits", default="configs/runtime/resource_limits.yaml")
    parser.add_argument("--max-new-matrices", type=int, default=8)
    parser.add_argument("--max-planned-gpu-solves", type=int, default=24)
    parser.add_argument("--measurement-repeats", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--max-cols", type=int, default=10_000)
    parser.add_argument("--max-nnz", type=int, default=100_000)
    parser.add_argument("--max-archive-mb", type=float, default=8.0)
    parser.add_argument("--max-iter", type=int, default=256)
    parser.add_argument("--tolerance-rel", type=float, default=1.0e-5)
    parser.add_argument("--out", default="runs/phase1_csr_benchmark_expansion_plan")
    args = parser.parse_args()

    plan = build_csr_benchmark_expansion_plan_from_files(
        args.selection,
        args.selector_rows,
        args.resource_limits,
        max_new_matrices=args.max_new_matrices,
        max_planned_gpu_solves=args.max_planned_gpu_solves,
        measurement_repeats=args.measurement_repeats,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        max_nnz=args.max_nnz,
        max_archive_size_bytes=int(args.max_archive_mb * 1_000_000),
        max_iter=args.max_iter,
        tolerance_rel=args.tolerance_rel,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "matrix_queue": output / "csr_benchmark_matrix_queue.jsonl",
        "candidate_queue": output / "csr_benchmark_candidate_queue.jsonl",
        "summary": output / "csr_benchmark_expansion_summary.json",
        "schema": output / "csr_benchmark_expansion_schema.json",
        "report": output / "csr_benchmark_expansion_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_benchmark_matrix_plan(plan.matrix_rows, paths["matrix_queue"])
    write_csr_benchmark_candidate_plan(plan.candidate_rows, paths["candidate_queue"])
    write_csr_benchmark_expansion_summary(plan.summary, paths["summary"])
    write_csr_benchmark_expansion_schema(plan.schema, paths["schema"])
    write_csr_benchmark_expansion_report(plan, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_benchmark_expansion_plan",
            command="scripts/tss_csr_benchmark_expansion_plan.py",
            tracked_files=CORE_CSR_BENCHMARK_EXPANSION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": plan.summary.status,
                "schema_version": plan.summary.schema_version,
                "plan_id": plan.summary.plan_id,
                "planned_matrices": plan.summary.planned_matrices,
                "planned_candidate_jobs": plan.summary.planned_candidate_jobs,
                "planned_gpu_solve_attempts": plan.summary.planned_gpu_solve_attempts,
                "runtime_selector_changed": plan.summary.runtime_selector_changed,
                "executes_gpu": plan.summary.executes_gpu,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {plan.summary.status}")
    if plan.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
