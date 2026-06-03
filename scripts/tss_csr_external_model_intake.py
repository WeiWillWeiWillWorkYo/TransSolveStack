"""Run end-to-end intake for an external CSR policy model checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_external_model_intake import (
    run_csr_external_model_intake_from_files,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_EXTERNAL_MODEL_INTAKE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="runs/phase1_csr_external_model_intake")
    parser.add_argument(
        "--source-model",
        default="runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json",
    )
    parser.add_argument(
        "--tensors",
        default="runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json",
    )
    parser.add_argument(
        "--request-index",
        default="runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl",
    )
    parser.add_argument(
        "--baseline-summary",
        default="runs/phase1_csr_transformer_ready/combined_csr_learning_summary.json",
    )
    parser.add_argument(
        "--baseline-predictions",
        default="runs/phase1_csr_transformer_ready/combined_csr_baseline_predictions.jsonl",
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
        "--guarded-auto-solve-summary",
        default=(
            "runs/phase1_csr_guarded_auto_solve/"
            "csr_guarded_auto_solve_summary.json"
        ),
    )
    parser.add_argument(
        "--guarded-auto-solve-results",
        default=(
            "runs/phase1_csr_guarded_auto_solve/"
            "csr_guarded_auto_solve_results.jsonl"
        ),
    )
    parser.add_argument("--contributor-id", default="wei_cui_reference")
    parser.add_argument("--contributor-name", default="Wei CUI")
    parser.add_argument(
        "--contribution-name",
        default="csr_external_ranker_reference_intake",
    )
    parser.add_argument("--contribution-version", default="phase1-reference")
    args = parser.parse_args()

    data = run_csr_external_model_intake_from_files(
        checkpoint_path=args.checkpoint,
        output_dir=args.out,
        source_model_path=args.source_model,
        tensor_path=args.tensors,
        request_index_path=args.request_index,
        baseline_summary_path=args.baseline_summary,
        baseline_predictions_path=args.baseline_predictions,
        csr_path=args.csr,
        selector_path=args.selector_rows,
        guarded_auto_solve_summary_path=args.guarded_auto_solve_summary,
        guarded_auto_solve_results_path=args.guarded_auto_solve_results,
        contributor_id=args.contributor_id,
        contributor_name=args.contributor_name,
        contribution_name=args.contribution_name,
        contribution_version=args.contribution_version,
    )
    summary = data["summary"]
    output = Path(args.out)
    manifest_path = output / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_external_model_intake",
            command="scripts/tss_csr_external_model_intake.py",
            tracked_files=CORE_CSR_EXTERNAL_MODEL_INTAKE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "intake_ready": summary["intake_ready"],
                "shadow_submission_ready": summary["shadow_submission_ready"],
                "runtime_promotion_ready": summary["runtime_promotion_ready"],
                "adapter_ready": summary["adapter_ready"],
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
        manifest_path,
    )
    for key, value in data["paths"].items():
        print(f"{key}: {value}")
    print(f"manifest: {manifest_path}")
    print(f"status: {summary['status']}")
    print(f"shadow_submission_ready: {summary['shadow_submission_ready']}")
    print(f"runtime_promotion_ready: {summary['runtime_promotion_ready']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
