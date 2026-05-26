"""Export CSR learned-selector numeric training arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.policies.csr_tensor_export import (
    build_csr_tensor_export,
    write_csr_tensor_arrays,
    write_csr_tensor_report,
    write_csr_tensor_request_index,
    write_csr_tensor_schema,
    write_csr_tensor_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_TRAINING_TENSOR_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requests",
        default="runs/phase1_csr_model_contract/csr_model_requests.jsonl",
    )
    parser.add_argument(
        "--targets",
        default="runs/phase1_csr_model_contract/csr_model_targets.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_csr_training_tensors")
    args = parser.parse_args()

    export = build_csr_tensor_export(args.requests, args.targets)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "arrays": output / "csr_training_tensors.json",
        "request_index": output / "csr_training_request_index.jsonl",
        "summary": output / "csr_training_tensor_summary.json",
        "schema": output / "csr_training_tensor_schema.json",
        "report": output / "csr_training_tensor_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_tensor_arrays(export, paths["arrays"])
    write_csr_tensor_request_index(export, paths["request_index"])
    write_csr_tensor_summary(export.summary, paths["summary"])
    write_csr_tensor_schema(export.schema, paths["schema"])
    write_csr_tensor_report(export, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_training_tensor_export",
            command="scripts/tss_csr_training_tensors.py",
            tracked_files=CORE_CSR_TRAINING_TENSOR_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": export.summary.status,
                "schema_version": export.summary.schema_version,
                "storage_format": export.summary.storage_format,
                "num_requests": export.summary.num_requests,
                "num_global_candidates": export.summary.num_global_candidates,
                "matrix_feature_dim": export.summary.matrix_feature_dim,
                "candidate_feature_dim": export.summary.candidate_feature_dim,
                "model_required": export.summary.model_required,
                "runtime_selector_changed": export.summary.runtime_selector_changed,
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
