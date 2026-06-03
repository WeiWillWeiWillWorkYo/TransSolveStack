"""Build the minimal external CSR Transformer training package manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_transformer_training_package import (  # noqa: E402
    build_csr_transformer_training_package_from_files,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_TRANSFORMER_TRAINING_PACKAGE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--handoff-dir",
        default="runs/phase1_csr_transformer_handoff_bundle",
    )
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_transformer_training_package",
    )
    args = parser.parse_args()

    artifact = build_csr_transformer_training_package_from_files(
        handoff_dir=args.handoff_dir,
        output_dir=args.out,
    )
    summary = artifact["summary"]
    manifest_path = Path(args.out) / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_transformer_training_package",
            command="scripts/tss_csr_transformer_training_package.py",
            tracked_files=CORE_CSR_TRANSFORMER_TRAINING_PACKAGE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "package_ready": summary["package_ready"],
                "model_training_required": summary["model_training_required"],
                "model_trained": summary["model_trained"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "executes_gpu": summary["executes_gpu"],
                "num_input_files": summary["num_input_files"],
                "num_expected_outputs": summary["num_expected_outputs"],
                "num_validation_steps": summary["num_validation_steps"],
                "training_model_requests": summary["training_model_requests"],
                "training_global_candidates": summary[
                    "training_global_candidates"
                ],
                "internal_development_docs_included": summary[
                    "internal_development_docs_included"
                ],
            },
        ),
        manifest_path,
    )
    for key, value in {**artifact["paths"], "manifest": str(manifest_path)}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"package_ready: {summary['package_ready']}")
    print(f"num_input_files: {summary['num_input_files']}")
    print(f"num_validation_steps: {summary['num_validation_steps']}")
    print(f"runtime_selector_changed: {summary['runtime_selector_changed']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
