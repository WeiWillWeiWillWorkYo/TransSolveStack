"""Build the formal CSR Transformer training entrypoint artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_transformer_training_entrypoint import (
    build_csr_transformer_training_entrypoint_from_files,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_TRANSFORMER_TRAINING_ENTRYPOINT_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-dir", default="runs/phase1_csr_transformer_ready")
    parser.add_argument(
        "--quality-gate-summary",
        default=(
            "runs/phase1_csr_transformer_quality_gate/"
            "csr_transformer_quality_gate_summary.json"
        ),
    )
    parser.add_argument("--out", default="runs/phase1_csr_transformer_training_entrypoint")
    parser.add_argument("--model-family", default="csr_transformer_policy_v1")
    args = parser.parse_args()

    export = build_csr_transformer_training_entrypoint_from_files(
        args.ready_dir,
        args.quality_gate_summary,
        args.out,
        model_family=args.model_family,
    )
    summary = export["summary"]
    paths = dict(export["paths"])
    paths["manifest"] = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_transformer_training_entrypoint",
            command="scripts/tss_csr_transformer_training_entrypoint.py",
            tracked_files=CORE_CSR_TRANSFORMER_TRAINING_ENTRYPOINT_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "training_entrypoint_ready": summary["training_entrypoint_ready"],
                "model_training_required": summary["model_training_required"],
                "model_trained": summary["model_trained"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "transformer_connectable": summary["transformer_connectable"],
                "num_model_requests": summary["num_model_requests"],
                "num_tensor_requests": summary["num_tensor_requests"],
                "current_quality_gate_runtime_eligible": summary[
                    "current_quality_gate_runtime_eligible"
                ],
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
