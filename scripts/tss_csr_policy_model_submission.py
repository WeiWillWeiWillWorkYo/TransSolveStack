"""Build a submission package for a CSR policy model artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_policy_model_submission import (
    build_csr_policy_model_submission_from_files,
    write_csr_policy_model_submission_manifest,
    write_csr_policy_model_submission_model_card,
    write_csr_policy_model_submission_report,
    write_csr_policy_model_submission_rows,
    write_csr_policy_model_submission_schema,
    write_csr_policy_model_submission_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_POLICY_MODEL_SUBMISSION_PROVENANCE_FILES,
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
        "--acceptance-summary",
        default=(
            "runs/phase1_csr_policy_model_acceptance/"
            "csr_policy_model_acceptance_summary.json"
        ),
    )
    parser.add_argument(
        "--acceptance-rows",
        default=(
            "runs/phase1_csr_policy_model_acceptance/"
            "csr_policy_model_acceptance_rows.jsonl"
        ),
    )
    parser.add_argument("--out", default="runs/phase1_csr_policy_model_submission")
    parser.add_argument("--contributor-id", default="wei_cui_reference")
    parser.add_argument("--contributor-name", default="Wei CUI")
    parser.add_argument(
        "--contribution-name",
        default="csr_masked_self_attention_ranker_v1_reference_submission",
    )
    parser.add_argument("--contribution-version", default="phase1-reference")
    parser.add_argument(
        "--training-statement",
        default=(
            "Reference CSR Transformer ranker trained on the current Phase 1 "
            "Transformer-ready SuiteSparse subset."
        ),
    )
    parser.add_argument(
        "--acknowledge-terms",
        dest="acknowledge_terms",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-acknowledge-terms",
        dest="acknowledge_terms",
        action="store_false",
    )
    args = parser.parse_args()

    data = build_csr_policy_model_submission_from_files(
        args.model_artifact,
        args.acceptance_summary,
        args.acceptance_rows,
        contributor_id=args.contributor_id,
        contributor_name=args.contributor_name,
        contribution_name=args.contribution_name,
        contribution_version=args.contribution_version,
        training_statement=args.training_statement,
        contributor_terms_acknowledged=args.acknowledge_terms,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "submission": output / "csr_policy_model_submission_manifest.json",
        "rows": output / "csr_policy_model_submission_rows.jsonl",
        "summary": output / "csr_policy_model_submission_summary.json",
        "schema": output / "csr_policy_model_submission_schema.json",
        "model_card": output / "MODEL_CARD.md",
        "report": output / "csr_policy_model_submission_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_policy_model_submission_manifest(data, paths["submission"])
    write_csr_policy_model_submission_rows(data["rows"], paths["rows"])
    write_csr_policy_model_submission_summary(data, paths["summary"])
    write_csr_policy_model_submission_schema(data, paths["schema"])
    write_csr_policy_model_submission_model_card(data, paths["model_card"])
    write_csr_policy_model_submission_report(data, paths["report"])
    summary = data["summary"]
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_policy_model_submission",
            command="scripts/tss_csr_policy_model_submission.py",
            tracked_files=CORE_CSR_POLICY_MODEL_SUBMISSION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "submission_ready": summary["submission_ready"],
                "shadow_submission_ready": summary["shadow_submission_ready"],
                "runtime_promotion_ready": summary["runtime_promotion_ready"],
                "model_id": summary["model_id"],
                "accepted_for_shadow": summary["accepted_for_shadow"],
                "accepted_for_runtime_promotion": summary[
                    "accepted_for_runtime_promotion"
                ],
                "num_packaged_files": summary["num_packaged_files"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"shadow_submission_ready: {summary['shadow_submission_ready']}")
    print(f"runtime_promotion_ready: {summary['runtime_promotion_ready']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
