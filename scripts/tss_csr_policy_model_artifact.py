"""Build the runtime-loadable CSR policy model artifact contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_policy_model_artifact import (
    build_csr_policy_model_artifact_from_files,
    write_csr_policy_model_artifact,
    write_csr_policy_model_artifact_report,
    write_csr_policy_model_artifact_rows,
    write_csr_policy_model_artifact_schema,
    write_csr_policy_model_artifact_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_POLICY_MODEL_ARTIFACT_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
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
        "--quality-gate-summary",
        default=(
            "runs/phase1_csr_transformer_quality_gate/"
            "csr_transformer_quality_gate_summary.json"
        ),
    )
    parser.add_argument(
        "--replay-summary",
        default=(
            "runs/phase1_csr_transformer_model_replay/"
            "csr_transformer_model_replay_summary.json"
        ),
    )
    parser.add_argument("--out", default="runs/phase1_csr_policy_model_artifact")
    args = parser.parse_args()

    data = build_csr_policy_model_artifact_from_files(
        args.model,
        args.tensors,
        args.request_index,
        args.quality_gate_summary,
        args.replay_summary,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "artifact": output / "csr_policy_model_artifact.json",
        "rows": output / "csr_policy_model_artifact_rows.jsonl",
        "summary": output / "csr_policy_model_artifact_summary.json",
        "schema": output / "csr_policy_model_artifact_schema.json",
        "report": output / "csr_policy_model_artifact_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_policy_model_artifact(data, paths["artifact"])
    write_csr_policy_model_artifact_rows(data["rows"], paths["rows"])
    write_csr_policy_model_artifact_summary(data, paths["summary"])
    write_csr_policy_model_artifact_schema(data, paths["schema"])
    write_csr_policy_model_artifact_report(data, paths["report"])
    summary = data["summary"]
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_policy_model_artifact",
            command="scripts/tss_csr_policy_model_artifact.py",
            tracked_files=CORE_CSR_POLICY_MODEL_ARTIFACT_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "artifact_ready": summary["artifact_ready"],
                "adapter": summary["adapter"],
                "model_id": summary["model_id"],
                "model_loaded": summary["model_loaded"],
                "replay_exact": summary["replay_exact"],
                "guard_required": summary["guard_required"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
