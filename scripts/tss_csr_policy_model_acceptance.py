"""Run the CSR policy model artifact acceptance gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_policy_model_acceptance import (
    build_csr_policy_model_acceptance_from_files,
    write_csr_policy_model_acceptance_report,
    write_csr_policy_model_acceptance_rows,
    write_csr_policy_model_acceptance_schema,
    write_csr_policy_model_acceptance_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_POLICY_MODEL_ACCEPTANCE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-artifact",
        default="runs/phase1_csr_policy_model_artifact/csr_policy_model_artifact.json",
    )
    parser.add_argument(
        "--csr",
        default="runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    )
    parser.add_argument(
        "--selector-rows",
        default="runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    )
    parser.add_argument(
        "--learned-guard-summary",
        default="runs/phase1_csr_learned_guard/csr_learned_guard_summary.json",
    )
    parser.add_argument(
        "--guarded-auto-solve-summary",
        default="runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_summary.json",
    )
    parser.add_argument(
        "--guarded-auto-solve-results",
        default="runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_results.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_csr_policy_model_acceptance")
    args = parser.parse_args()

    data = build_csr_policy_model_acceptance_from_files(
        args.model_artifact,
        args.csr,
        args.selector_rows,
        args.learned_guard_summary,
        args.guarded_auto_solve_summary,
        args.guarded_auto_solve_results,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": output / "csr_policy_model_acceptance_rows.jsonl",
        "summary": output / "csr_policy_model_acceptance_summary.json",
        "schema": output / "csr_policy_model_acceptance_schema.json",
        "report": output / "csr_policy_model_acceptance_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_policy_model_acceptance_rows(data["rows"], paths["rows"])
    write_csr_policy_model_acceptance_summary(data, paths["summary"])
    write_csr_policy_model_acceptance_schema(data, paths["schema"])
    write_csr_policy_model_acceptance_report(data, paths["report"])
    summary = data["summary"]
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_policy_model_acceptance",
            command="scripts/tss_csr_policy_model_acceptance.py",
            tracked_files=CORE_CSR_POLICY_MODEL_ACCEPTANCE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "model_id": summary["model_id"],
                "accepted_for_shadow": summary["accepted_for_shadow"],
                "accepted_for_runtime_promotion": summary[
                    "accepted_for_runtime_promotion"
                ],
                "guarded_gpu_shadow_smoke_checked": summary[
                    "guarded_gpu_shadow_smoke_checked"
                ],
                "runtime_selector_changed": summary["runtime_selector_changed"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"accepted_for_shadow: {summary['accepted_for_shadow']}")
    print(
        "accepted_for_runtime_promotion: "
        f"{summary['accepted_for_runtime_promotion']}"
    )
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
