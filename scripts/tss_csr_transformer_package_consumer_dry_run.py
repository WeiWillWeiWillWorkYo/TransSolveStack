"""Run a consumer-side dry run for the CSR Transformer training package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_transformer_package_consumer_dry_run import (  # noqa: E402
    build_csr_transformer_package_consumer_dry_run_from_files,
)
from transsolvestack.profiling.provenance import (  # noqa: E402
    CORE_CSR_TRANSFORMER_PACKAGE_CONSUMER_DRY_RUN_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir",
        default="runs/phase1_csr_transformer_training_package",
    )
    parser.add_argument(
        "--reference-intake-dir",
        default="runs/phase1_csr_external_model_intake",
    )
    parser.add_argument(
        "--out",
        default="runs/phase1_csr_transformer_package_consumer_dry_run",
    )
    args = parser.parse_args()

    data = build_csr_transformer_package_consumer_dry_run_from_files(
        package_dir=args.package_dir,
        reference_intake_dir=args.reference_intake_dir,
        output_dir=args.out,
    )
    summary = data["summary"]
    output = Path(args.out)
    manifest_path = output / "artifact_manifest.json"
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_transformer_package_consumer_dry_run",
            command="scripts/tss_csr_transformer_package_consumer_dry_run.py",
            tracked_files=CORE_CSR_TRANSFORMER_PACKAGE_CONSUMER_DRY_RUN_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "consumer_ready": summary["consumer_ready"],
                "dry_run_only": summary["dry_run_only"],
                "source_package_ready": summary["source_package_ready"],
                "num_inputs_verified": summary["num_inputs_verified"],
                "num_expected_outputs_resolved": summary[
                    "num_expected_outputs_resolved"
                ],
                "num_validation_steps_resolved": summary[
                    "num_validation_steps_resolved"
                ],
                "reference_shadow_submission_ready": summary[
                    "reference_shadow_submission_ready"
                ],
                "reference_runtime_promotion_ready": summary[
                    "reference_runtime_promotion_ready"
                ],
                "runtime_selector_changed": summary["runtime_selector_changed"],
            },
        ),
        manifest_path,
    )
    for key, value in {**data["paths"], "manifest": str(manifest_path)}.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    print(f"consumer_ready: {summary['consumer_ready']}")
    print(f"dry_run_only: {summary['dry_run_only']}")
    print(f"runtime_selector_changed: {summary['runtime_selector_changed']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
