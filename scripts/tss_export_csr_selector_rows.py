"""Export real CSR diagnostics and selector training rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_selector_data import (
    build_csr_selector_export_from_files,
    write_csr_diagnostic_rows,
    write_csr_selector_report,
    write_csr_selector_rows,
    write_csr_selector_schema,
    write_csr_selector_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_SELECTOR_READINESS_PROVENANCE_FILES,
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
    parser.add_argument(
        "--solve-results",
        default="runs/phase1_taichi_csr_solve/csr_solve_results.jsonl",
    )
    parser.add_argument(
        "--solve-summary",
        default="runs/phase1_taichi_csr_solve/csr_solve_summary.json",
    )
    parser.add_argument(
        "--bicgstab-results",
        default="runs/phase1_taichi_csr_bicgstab/csr_bicgstab_results.jsonl",
    )
    parser.add_argument(
        "--bicgstab-summary",
        default="runs/phase1_taichi_csr_bicgstab/csr_bicgstab_summary.json",
    )
    parser.add_argument(
        "--gmres-results",
        default="runs/phase1_taichi_csr_gmres/csr_gmres_results.jsonl",
    )
    parser.add_argument(
        "--gmres-summary",
        default="runs/phase1_taichi_csr_gmres/csr_gmres_summary.json",
    )
    parser.add_argument(
        "--richardson-results",
        default="runs/phase1_taichi_csr_richardson/csr_richardson_results.jsonl",
    )
    parser.add_argument(
        "--richardson-summary",
        default="runs/phase1_taichi_csr_richardson/csr_richardson_summary.json",
    )
    parser.add_argument(
        "--chebyshev-results",
        default="runs/phase1_taichi_csr_chebyshev/csr_chebyshev_results.jsonl",
    )
    parser.add_argument(
        "--chebyshev-summary",
        default="runs/phase1_taichi_csr_chebyshev/csr_chebyshev_summary.json",
    )
    parser.add_argument("--out", default="runs/phase1_csr_selector_readiness")
    args = parser.parse_args()

    export = build_csr_selector_export_from_files(
        args.csr,
        (
            args.solve_results,
            args.bicgstab_results,
            args.gmres_results,
            args.richardson_results,
            args.chebyshev_results,
        ),
        screened_results_path=(
            args.solve_summary,
            args.bicgstab_summary,
            args.gmres_summary,
            args.richardson_summary,
            args.chebyshev_summary,
        ),
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "diagnostics": output / "csr_diagnostic_rows.jsonl",
        "selector_rows": output / "csr_selector_rows.jsonl",
        "summary": output / "csr_selector_summary.json",
        "schema": output / "csr_selector_schema.json",
        "report": output / "csr_selector_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_diagnostic_rows(export.diagnostics, paths["diagnostics"])
    write_csr_selector_rows(export.selector_rows, paths["selector_rows"])
    write_csr_selector_summary(export.summary, paths["summary"])
    write_csr_selector_schema(paths["schema"])
    write_csr_selector_report(export, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_selector_readiness_export",
            command="scripts/tss_export_csr_selector_rows.py",
            tracked_files=CORE_CSR_SELECTOR_READINESS_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": export.summary.status,
                "solve_result_sources": (
                    args.solve_results,
                    args.bicgstab_results,
                    args.gmres_results,
                    args.richardson_results,
                    args.chebyshev_results,
                ),
                "screened_result_sources": (
                    args.solve_summary,
                    args.bicgstab_summary,
                    args.gmres_summary,
                    args.richardson_summary,
                    args.chebyshev_summary,
                ),
                "num_diagnostic_rows": export.summary.num_diagnostic_rows,
                "num_selector_rows": export.summary.num_selector_rows,
                "num_success_rows": export.summary.num_success_rows,
                "num_failed_rows": export.summary.num_failed_rows,
                "num_applicability_rows": export.summary.num_applicability_rows,
                "num_oracle_rows": export.summary.num_oracle_rows,
                "schema_version": export.summary.schema_version,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {export.summary.status}")
    if export.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
